import os
import json
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
import argparse
import random
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Import both model implementations
import sys
sys.path.append("DeepSeek-V3/inference")
sys.path.append(".")  # Add current directory to path for importing models.py
from v3.inference.model import ModelArgs as OfficialModelArgs
from v3.inference.model import Transformer as OfficialTransformer
from v3.inference.generate import generate as official_generate

# Import the modified model
from model import ModelArgs as ModifiedModelArgs
from model import Transformer as ModifiedTransformer
from generate import pp_generate
from batched_generate import batched_generate
from models import setup_pp
from utils import DistributedInfo

# Import tokenizer
from transformers import AutoTokenizer


def setup_distributed():
    """Initialize distributed environment."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("Not using distributed mode")
        return False
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    
    print(f"Initialized process group: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    return True


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load model configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_official_model(config: Dict[str, Any], mesh: DeviceMesh) -> OfficialTransformer:
    """Create and initialize the official DeepSeek model."""
    args = OfficialModelArgs(**config)
    model = OfficialTransformer(args, mesh)
    return model


def create_modified_model(config: Dict[str, Any], mesh: DeviceMesh) -> ModifiedTransformer:
    """Create and initialize the modified DeepSeek model."""
    args = ModifiedModelArgs(**config)
    model = ModifiedTransformer(args, mesh, grad_checkpointing=False)
    return model


def copy_weights(official_model: OfficialTransformer, modified_model: ModifiedTransformer):
    """Copy weights from official model to modified model."""
    # Get state dict from official model
    official_state_dict = official_model.state_dict()
    
    # Load state dict into modified model
    # We need to handle potential differences in state dict keys
    modified_state_dict = modified_model.state_dict()
    
    # Print some debugging info about the state dicts
    if dist.get_rank() == 0:
        print(f"Official model has {len(official_state_dict)} parameters")
        print(f"Modified model has {len(modified_state_dict)} parameters")
        
        # Find keys that are in one but not the other
        official_keys = set(official_state_dict.keys())
        modified_keys = set(modified_state_dict.keys())
        
        only_in_official = official_keys - modified_keys
        only_in_modified = modified_keys - official_keys
        
        if only_in_official:
            print(f"Keys only in official model: {only_in_official}")
        if only_in_modified:
            print(f"Keys only in modified model: {only_in_modified}")
    
    # Copy matching parameters
    for key in official_state_dict:
        if key in modified_state_dict:
            modified_state_dict[key].copy_(official_state_dict[key])
    
    # Load the updated state dict
    modified_model.load_state_dict(modified_state_dict, strict=False)
    
    return modified_model


def check_model_parity(official_model: OfficialTransformer, modified_model: ModifiedTransformer, config: Dict[str, Any]):
    """Check if both models produce the same output for the same input."""
    # Set both models to evaluation mode
    official_model.eval()
    modified_model.eval()

    official_model.to('cuda')
    modified_model.to('cuda')
    
    # Create random input
    batch_size = 2
    seq_len = 32
    vocab_size = config["vocab_size"]
    
    # Use the same seed for reproducibility
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
    
    # Forward pass through both models
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Official model forward pass
            official_output = official_model(input_ids)
            
            # Modified model forward pass
            modified_output, _ = modified_model(input_ids)
    
    # Check if outputs are the same
    print(f"checking {official_output.size()=} == {modified_output.size()=}")
    is_close = torch.allclose(official_output.float(), modified_output.float(), rtol=1e-5, atol=1e-5)
    
    # Calculate difference statistics
    if not is_close:
        abs_diff = torch.abs(official_output - modified_output)
        max_diff = torch.max(abs_diff).item()
        mean_diff = torch.mean(abs_diff).item()
        
        if dist.get_rank() == 0:
            print(f"Max absolute difference: {max_diff}")
            print(f"Mean absolute difference: {mean_diff}")
            
            # Print some sample values
            sample_idx = (0, 0)  # First batch, first position
            print(f"Sample official output: {official_output[sample_idx]}")
            print(f"Sample modified output: {modified_output[sample_idx]}")
    
    return is_close


def logits_parity(config_path: str, seed: int = 42) -> bool:
    """
    Check if the official and modified models produce the same logits.
    
    Args:
        config_path (str): Path to the model configuration file.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        
    Returns:
        bool: True if the models produce the same logits, False otherwise.
    """
    # Set random seed
    set_seed(seed)
    
    # Setup distributed environment
    is_distributed = setup_distributed()
    
    # Set default device
    device = torch.device("cuda")
    
    # Load model configuration
    config = load_config(config_path)
    
    # Create device mesh for tensor parallelism
    if is_distributed:
        world_size = dist.get_world_size()
        mesh = init_device_mesh('cuda', (world_size,), mesh_dim_names=("tp",))
        tp_mesh = mesh["tp"]
    else:
        mesh = None
        tp_mesh = None
    
    # Create models
    if dist.get_rank() == 0:
        print("Creating official model...")
    official_model = create_official_model(config, tp_mesh)
    official_model.to(device)
    
    if dist.get_rank() == 0:
        print("Creating modified model...")
    modified_model = create_modified_model(config, tp_mesh)
    modified_model.to(device)
    
    # Initialize modified model with official model weights
    if dist.get_rank() == 0:
        print("Copying weights from official model to modified model...")
    modified_model = copy_weights(official_model, modified_model)
    
    # Check model parity
    if dist.get_rank() == 0:
        print("Checking model parity...")
    is_same = check_model_parity(official_model, modified_model, config)
    
    if dist.get_rank() == 0:
        if is_same:
            print("✅ Models produce identical outputs!")
        else:
            print("❌ Models produce different outputs!")
    
    # Clean up
    if is_distributed:
        dist.destroy_process_group()
    
    return is_same


def unroll_parity(
    config_path: str, 
    tokenizer_path: str,
    prompts: List[str],
    max_new_tokens: int = 50,
    pp_size: int = 2,
    seed: int = 1
) -> bool:
    """
    Check if the official and modified models produce the same generated text.
    
    Args:
        config_path (str): Path to the model configuration file.
        tokenizer_path (str): Path to the tokenizer.
        prompts (List[str]): List of prompts to generate text from.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 50.
        pp_size (int, optional): Number of pipeline stages. Defaults to 2.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        
    Returns:
        bool: True if the models produce the same generated text, False otherwise.
    """
    # Set random seed
    set_seed(seed)
    
    # Setup distributed environment
    is_distributed = setup_distributed()
    
    # Set default device
    device = torch.device("cuda")
    
    # Load model configuration
    config = load_config(config_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Create device mesh for tensor parallelism and pipeline parallelism
    if is_distributed:
        world_size = dist.get_world_size()
        
        # Create a 1D mesh for simplicity
        dp_size = 1
        tp_size = 1
        mesh = init_device_mesh('cuda', (dp_size, pp_size, tp_size), mesh_dim_names=("dp", "pp", "tp"))
        tp_mesh = mesh["tp"]
        pp_mesh = mesh["pp"]
        dp_mesh = mesh["dp"]
        
        # Create a distributed info object for setup_pp
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_rank = int(os.environ.get("RANK", "0"))
        local_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        pp_rank = rank % pp_size
        tp_rank = rank // pp_size
        node_rank = world_rank // local_size
        node_size = world_size // local_size
        
        di = DistributedInfo(
            world_size=world_size,
            world_rank=rank,
            local_rank=local_rank,
            local_size=local_size,
            node_rank=node_rank,
            node_size=node_size,
            tp_size=1,  # We're using a 1D mesh for simplicity
            tp_rank=tp_rank,
            pp_size=pp_size,
            pp_rank=pp_rank,
            dp_size=1,  # No data parallelism for this test
            dp_rank=0,
            tp_mesh=tp_mesh,
            pp_mesh=pp_mesh,  # Using the same mesh for PP
            dp_mesh=dp_mesh,  # Using the same mesh for DP
            device_mesh=mesh
        )
    else:
        mesh = None
        tp_mesh = None
        di = None
        print("Warning: Distributed environment not set up. Pipeline parallelism requires distributed setup.")
        return False
    
    # Create models
    if dist.get_rank() == 0:
        print("Creating official model...")
    official_model = create_official_model(config, tp_mesh)
    official_model.to(device)
    
    if dist.get_rank() == 0:
        print("Creating modified model...")
    modified_model = create_modified_model(config, tp_mesh)
    modified_model.to(device)
    
    # Initialize modified model with official model weights
    if dist.get_rank() == 0:
        print("Copying weights from official model to modified model...")
    modified_model = copy_weights(official_model, modified_model)
    
    # Set up pipeline parallelism for modified model using the imported setup_pp
    if dist.get_rank() == 0:
        print("Setting up pipeline parallelism...")
    
    # Use the imported setup_pp function
    pp_schedule, modified_model, has_first_stage, has_last_stage, adjust_loss_fn, create_eval_schedule = setup_pp(
        modified_model, di, n_microbatches=1, use_random_init=True, pre_inited=True
    )
    
    # Tokenize prompts
    tokenized_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    
    # Generate text with official model
    official_texts = []
    if dist.get_rank() == 0:
        print("Generating text with official model...")
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                official_outputs = official_generate(
                    official_model,
                    prompt_tokens=tokenized_prompts,
                    max_new_tokens=max_new_tokens,
                    eos_id=tokenizer.eos_token_id,
                    temperature=0.0  # Greedy decoding
                )
        
        official_texts = [tokenizer.decode(output) for output in official_outputs]
        print("Official model outputs:")
        for i, text in enumerate(official_texts):
            print(f"Prompt {i+1}: {prompts[i]}")
            print(f"Output {i+1}: {text}")
            print("-" * 40)
    
    # Synchronize to ensure all ranks have the official model outputs
    dist.barrier()
    
    # Generate text with modified model using pipeline parallelism
    if dist.get_rank() == 0:
        print("Generating text with modified model...")
    
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            modified_outputs = batched_generate(
                model=modified_model,
                tokenizer=tokenizer,
                schedule=pp_schedule,
                create_eval_schedule=create_eval_schedule,
                has_first_stage=has_first_stage,
                has_last_stage=has_last_stage,
                tokenized_contexts=tokenized_prompts,
                max_new_tokens=max_new_tokens,
                temperature=0.0,  # Greedy decoding
                greedy=True,
                token_output=True,  # Return tokens instead of text
                only_completion=True
            )
    
    # Compare outputs on the last stage
    is_same = False
    if has_last_stage:
        modified_texts = [tokenizer.decode(output) for output in modified_outputs]
        
        print("Modified model outputs:")
        for i, text in enumerate(modified_texts):
            print(f"Prompt {i+1}: {prompts[i]}")
            print(f"Output {i+1}: {text}")
            print("-" * 40)
        
        if dist.get_rank() == 0:
            # Compare outputs
            is_same = all(off == mod for off, mod in zip(official_texts, modified_texts))
            
            if is_same:
                print("✅ Models produce identical generated text!")
            else:
                print("❌ Models produce different generated text!")
                # Print differences
                for i, (off, mod) in enumerate(zip(official_texts, modified_texts)):
                    if off != mod:
                        print(f"Difference in output {i+1}:")
                        print(f"Official: {off}")
                        print(f"Modified: {mod}")
                        print("-" * 40)
    
    # Clean up
    if is_distributed:
        dist.destroy_process_group()
    
    return is_same


def main():
    parser = argparse.ArgumentParser(description="DeepSeek Model Parity Check")
    parser.add_argument("--config", type=str, default="configs/config_2B.json", 
                        help="Path to model configuration file")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer (required for unroll parity)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--check-logits", action="store_true", 
                        help="Check logits parity")
    parser.add_argument("--check-unroll", action="store_true", 
                        help="Check unroll parity")
    parser.add_argument("--pp-size", type=int, default=2, 
                        help="Number of pipeline stages for unroll parity")
    parser.add_argument("--max-new-tokens", type=int, default=50, 
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--prompt", type=str, action="append", default=[],
                        help="Prompt for text generation (can be specified multiple times)")
    args = parser.parse_args()
    
    if args.check_logits:
        logits_parity(args.config, args.seed)
    
    if args.check_unroll:
        if not args.tokenizer:
            print("Error: --tokenizer is required for unroll parity check")
            return
        
        prompts = args.prompt if args.prompt else [
            "Once upon a time,",
        ]
        
        unroll_parity(
            args.config,
            args.tokenizer,
            prompts,
            args.max_new_tokens,
            args.pp_size,
            args.seed
        )


if __name__ == "__main__":
    main()
