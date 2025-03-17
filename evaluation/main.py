import os
import torch
import torch.nn.functional as F
from torch.distributed import init_process_group
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig
)
from torch.distributed._shard.sharded_tensor.api import ShardedTensor, Shard
torch.serialization.add_safe_globals([ShardedTensor, Shard])
from functools import partial
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import click
from eval_data import data
import numpy as np
from cpi import CPIS
from guess_on_heldout import score_rollouts_on_eval
from guess_on_puzzles import score_rollouts_on_puzzles
from play_stockfish import play_against_stockfish, make_engine

GS_BUCKET = os.environ.get("GS_BUCKET_CHECKPOINT")

def setup_distributed():
    """Initialize distributed training environment."""
    # Initialize the process group
    init_process_group(backend="nccl")
    # Get rank and world size
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    print(f"Initialized process {rank}/{world_size} on device {device}")
    return rank, world_size, device

def download_checkpoint(run_name, checkpoint_iter):
    """Download checkpoint shards before model initialization."""
    checkpoint_path = f"{GS_BUCKET}/{run_name}/checkpoints/{checkpoint_iter:08d}"
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Create local cache directory
    cache_dir = f'/data/checkpoints/{run_name}/{checkpoint_iter:08d}'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Only rank 0 downloads metadata file and lists shards
    meta_file = os.path.join(cache_dir, "meta.pt")
    shard_list_file = os.path.join(cache_dir, "shard_list.txt")
    
    if rank == 0:
        # Download meta file if it doesn't exist
        if not os.path.exists(meta_file):
            print(f"Rank 0 downloading meta file to {meta_file}")
            creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            auth_flag = f"-o 'Credentials:gs_service_key_file={creds_path}'" if creds_path else ""
            os.system(f"gsutil -q {auth_flag} cp {checkpoint_path}/meta.pt {meta_file}")
        
        # List all shard files in the bucket if list doesn't exist
        if not os.path.exists(shard_list_file):
            creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            auth_flag = f"-o 'Credentials:gs_service_key_file={creds_path}'" if creds_path else ""
            list_cmd = f"gsutil -q {auth_flag} ls {checkpoint_path}/model_shard_*"
            shard_files = os.popen(list_cmd).read().strip().split('\n')
            shard_files = sorted([f for f in shard_files if f.strip()])
            
            # Save the list of shard files for all ranks to use
            with open(shard_list_file, 'w') as f:
                for shard in shard_files:
                    f.write(f"{shard}\n")
    
    # Wait for rank 0 to finish downloading meta file and listing shards
    torch.distributed.barrier()
    
    # Read the shard list
    with open(shard_list_file, 'r') as f:
        shard_files = [line.strip() for line in f.readlines()]
    
    if rank >= len(shard_files):
        raise ValueError(f"Rank {rank} is out of range for available shards ({len(shard_files)})")
    
    # Get the shard assigned to this rank
    remote_shard_path = shard_files[rank]
    shard_filename = os.path.basename(remote_shard_path)
    local_shard_path = os.path.join(cache_dir, shard_filename)
    
    # Download the shard if it doesn't exist
    if not os.path.exists(local_shard_path):
        print(f"Rank {rank} downloading shard {shard_filename}")
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        auth_flag = f"-o 'Credentials:gs_service_key_file={creds_path}'" if creds_path else ""
        os.system(f"gsutil -q {auth_flag} cp {remote_shard_path} {local_shard_path}")
    
    # Wait for all ranks to finish downloading
    torch.distributed.barrier()
    
    return cache_dir, shard_filename



def init_model_with_fsdp(model_name, tokenizer, rank, device):
    """Initialize model with FSDP wrapping."""
    # Create base model
    model_config = AutoConfig.from_pretrained(model_name, local_files_only=False)
    model_config.vocab_size = len(tokenizer)
    model_config.use_cache = False
    model_config.max_position_embeddings = 1024
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(model_config)
    if rank == 0:
        print(model)
    
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16
    )

    # Auto wrapping policy to shard at transformer layer level
    wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision_policy,
        device_id=device,
        use_orig_params=True
    )
    
    if rank == 0:
        print('Model initialized with FSDP and converted to bfloat16')
    return model

def load_checkpoint_fsdp(model, cache_dir, shard_filename):
    """Load checkpoint directly with FSDP."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Load the shard
    local_shard_path = os.path.join(cache_dir, shard_filename)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = torch.load(local_shard_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        
    # Synchronize all processes after loading
    torch.distributed.barrier()
    print(f"Rank {rank} loaded checkpoint shard {shard_filename} successfully")


@click.command()
@click.option('--model-name', default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help='Model architecture to use')
@click.option('--tokenizer', default=None, help='Tokenizer to use')
@click.option('--run-name', default='sfchess/sideshow/llama8b-r1_smallerlr', help='Name of the training run to load weights from')
@click.option('--checkpoint', default=34044, type=int, help='Checkpoint number to load')
@click.option('--seq-len', default=512, help='Maximum sequence length')
@click.option('--batch-size', default=16, help='Batch size')
@click.option('--cpi', type=str, required=True)
@click.option('--which', type=str, required=True)
@click.option('--sf-depth', default=1, help='Stockfish depth')
@click.option('--sf-games', default=64, help='Number of Stockfish games to play')
@click.option('--sf-batch-size', default=64, help='Number of Stockfish games to play per batch')
def main(model_name, tokenizer, run_name, checkpoint, seq_len, batch_size, cpi, which, sf_depth, sf_games, sf_batch_size):
    """Evaluate model on chess positions."""
    # Initialize distributed environment
    rank, world_size, device = setup_distributed()
    
    # Download checkpoint files first
    print("\nDownloading checkpoint files...")
    cache_dir, shard_filename = download_checkpoint(run_name, checkpoint)
    
    # Initialize tokenizer and model
    print("\nInitializing model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name if tokenizer is None else tokenizer)
    model = init_model_with_fsdp(model_name, tokenizer, rank, device)
    
    # Load checkpoint
    print("\nLoading checkpoint into model...")
    load_checkpoint_fsdp(model, cache_dir, shard_filename)
    model.eval()

    cpi = CPIS[cpi]

    if which == 'heldout':
        if rank == 0:
            print("Evaluating on heldout set...")
        prompts, unrolls, results = score_rollouts_on_eval(model, tokenizer, cpi, rank, world_size, greedy=True, seq_len=seq_len, batch_size=batch_size, device=device)
        if rank == 0:
            print('\n\n'.join(unrolls))
            print(results)
    elif which == 'puzzle':
        if rank == 0:
            print("Evaluating on puzzle set...")
        prompts, unrolls, results = score_rollouts_on_puzzles(model, tokenizer, cpi, rank, world_size, greedy=True, seq_len=seq_len, batch_size=batch_size, device=device)
        if rank == 0:
            print('\n\n'.join(unrolls))
            print(results)
    elif which == 'stockfish':
        if rank == 0:
            print("Evaluating on stockfish...")
        model_engine = make_engine(model, tokenizer, cpi, greedy=True, seq_len=seq_len, batch_size=batch_size, device=device)
        results = play_against_stockfish(model_engine, sf_depth, sf_batch_size, sf_games)
        if rank == 0:
            print(results)
    else:
        print(f"Unknown evaluation type: {which}")

    # Cleanup
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()

    # torchrun --standalone --nproc_per_node=8 main.py --cpi fen_english_move --which stockfish