import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from prepare_data import get_tokenizer, Promptifier
import click
import tempfile
import shutil
from eval_data import data
import numpy as np
# Monkeypatch ShardedTensor to allow loading from checkpoint shards on one GPU
from torch.distributed._shard.sharded_tensor import ShardedTensor

GS_BUCKET = os.environ.get("GS_BUCKET_CHECKPOINT")

def patched_setstate(self, state):
    if isinstance(state, tuple):
        self._local_shards = state[0]
        self._metadata = state[1] if len(state) > 1 else None
    elif isinstance(state, dict):
        self._local_shards = state["_local_shards"]
        self._metadata = state.get("metadata")
    else:
        raise RuntimeError("Unexpected state type: " + str(type(state)))
ShardedTensor.__setstate__ = patched_setstate
def sharded_tensor_to_tensor(st):
    # Sort shards by the starting index along dimension 0
    shards = sorted(st._local_shards, key=lambda shard: shard.metadata.shard_offsets[0])
    return torch.cat([shard.tensor for shard in shards], dim=0)


def init_model(model_name, tokenizer, device='cuda'):
    """Initialize model without FSDP wrapping."""
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.use_cache = False
    model_config.vocab_size = len(tokenizer)
    model = AutoModelForCausalLM.from_config(model_config)
    model = model.to(dtype=torch.bfloat16, device=device)
    return model

def load_checkpoint(model, run_name, checkpoint_iter, device='cuda'):
    """Load sharded checkpoint from GCS by first consolidating the shards."""
    # Create temporary directory for checkpoint download
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download all checkpoint shards
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        auth_flag = f"-o 'Credentials:gs_service_key_file={creds_path}'" if creds_path else ""
        bucket_path = f"{GS_BUCKET}/{run_name}/checkpoints/{checkpoint_iter:08d}"
        
        # First list all files in the checkpoint directory
        list_cmd = f"gsutil -q {auth_flag} ls {bucket_path}/model_shard_*.pt"
        shard_list = os.popen(list_cmd).read().strip().split('\n')
        if not shard_list or shard_list[0] == '':
            raise RuntimeError(f"No checkpoint shards found at {bucket_path}")
        
        # Download each shard
        print("Downloading checkpoint shards...")
        for shard_path in shard_list:
            os.system(f"gsutil -q {auth_flag} cp {shard_path} {tmp_dir}/")
        # Download meta info
        os.system(f"gsutil -q {auth_flag} cp {bucket_path}/meta.pt {tmp_dir}/")

        # Load and consolidate shards
        print("Consolidating shards...")
        consolidated_state_dict = {}
        
        # Load each shard and merge
        shard_files = sorted([f for f in os.listdir(tmp_dir) if f.startswith('model_shard_')])
        print(f"Found {len(shard_files)} shards")
        
        for shard_file in shard_files:
            print(f"Loading shard {shard_file}")
            shard_dict = torch.load(os.path.join(tmp_dir, shard_file), map_location=device)
            for key, value in shard_dict.items():
                if isinstance(value, ShardedTensor):
                    value = sharded_tensor_to_tensor(value)
                if key not in consolidated_state_dict:
                    consolidated_state_dict[key] = value
                else:
                    consolidated_state_dict[key] = torch.cat([consolidated_state_dict[key], value], dim=0)
                print(f"Consolidated state dict {key} shape: {consolidated_state_dict[key].shape}")

        if not consolidated_state_dict:
            raise RuntimeError("No state dict could be loaded from checkpoint shards")
            
        # Load the consolidated state dict
        try:
            model.load_state_dict(consolidated_state_dict)
            print("Checkpoint loaded successfully")
        except Exception as e:
            print("\nError loading checkpoint. State dict keys in checkpoint:")
            print("\n".join(sorted(consolidated_state_dict.keys())))
            print("\nModel state dict keys:")
            print("\n".join(sorted(model.state_dict().keys())))
            raise e

def get_move_probs(model, tokenizer, fen, legal_moves, seq_len=100, device='cuda'):
    """Predict scores for all legal moves."""
    promptifier = Promptifier(tokenizer)
    
    # Prepare sequences for each legal move
    X = []
    Y = []
    masks = []
    for move in legal_moves:
        tokens, mask = promptifier(fen, move, pad_to=seq_len + 1)
        X.append(tokens[:-1])
        Y.append(tokens[1:])
        masks.append(mask[1:])
    
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.long, device=device)
    Y = torch.tensor(Y, dtype=torch.long, device=device)
    masks = torch.tensor(masks, dtype=torch.long, device=device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids=X)
        logits = torch.log_softmax(outputs.logits, dim=-1).float()
        losses = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), reduction='none').view(Y.shape)
        response_losses = torch.where(masks > 0, losses, 0.).sum(dim=1)
        
    # Convert losses to scores (lower loss = higher score)
    probs = np.exp(-response_losses.cpu().numpy())
    return probs

@click.command()
@click.option('--model-name', default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help='Model architecture to use')
@click.option('--tokenizer', default=None, help='Tokenizer to use')
@click.option('--run-name', required=True, help='Name of the training run to load weights from')
@click.option('--checkpoint-iter', required=True, type=int, help='Iteration number of checkpoint to load')
@click.option('--seq-len', default=100, help='Maximum sequence length')
@click.option('--device', default='cuda', help='Device to run on (cuda or cpu)')
def main(model_name, tokenizer, run_name, checkpoint_iter, seq_len, device):
    """Evaluate model on chess positions from eval_data."""
    # Initialize tokenizer and model
    print("\nInitializing model...")
    tokenizer = get_tokenizer(name=model_name if tokenizer is None else tokenizer)
    model = init_model(model_name, tokenizer, device)
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    load_checkpoint(model, run_name, checkpoint_iter, device)
    
    model.eval()
    
    print("\nEvaluating positions...\n")
    
    for i, (fen, best_move, legal_moves) in enumerate(data):
        
        # Get scores for all legal moves
        probs = get_move_probs(model, tokenizer, fen, legal_moves, seq_len, device)
        selected_move = legal_moves[np.argmax(probs)]
        is_correct = selected_move == best_move
        print(f"\nPosition {i+1}: " + (f"\033[94mcorrect\033[0m" if is_correct else f"\033[91mincorrect\033[0m"))
        print(f" {fen}   best={best_move} selected={selected_move} (legality={probs.sum():.2%})")

        # Sort moves by probability
        moves_with_probs = list(zip(legal_moves, probs))
        moves_with_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Print model's evaluation
        print("Move    Probability")
        print("-" * 20)
        for move, prob in moves_with_probs:
            indicator = "* " if move == best_move else "  "
            print(f"{indicator}{move:<6} {prob:>8.2%}")
        print("-" * 50)

        break

if __name__ == "__main__":
    main() 