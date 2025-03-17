import argparse
import json
import os
import datetime
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from tqdm import tqdm

from safetensors.torch import safe_open, save_file
from model import Transformer, ModelArgs

mapping = {
    "embed": 0,
    "attn_norm": None,
    "ffn_norm": None,
    "wq": 0,
    "wq_a": None,
    "q_norm": None,
    "wq_b": 0,
    "wkv_a": None,
    "kv_norm": None,
    "wkv_b": 0,
    "wo": 1,
    "gate": None,
    "w1": 0,
    "w2": 1,
    "w3": 0,
    "norm": None,
    "head": 0,
    "scale": None,
}

def split_layers(n_layers, tp, pp):
    base = n_layers // pp
    remainder = n_layers % pp
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(pp)]
    return chunk_sizes


def shard_model(model_config: str, tp_size: int, pp_size: int, output_dir: str, upload_to_gcs: bool = False, run_name: str = None):
    """
    Initialize a model locally by reading model_config, and shard the model along the pipeline and tensor dimensions. Save the shards to output_dir, and optionally upload to GCS.
    """
    with open(model_config, "r") as f:
        model_args = ModelArgs(**json.load(f))


    model = Transformer(model_args)
    # split the model
    n_local_experts = model_args.n_routed_experts // tp_size
    layer_chunk_sizes = split_layers(model_args.n_layers, tp_size, pp_size)

    layer_idx = 0
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model config to output directory
    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump(vars(model_args), f, indent=2)
    
    # Process each pipeline stage
    for pp_rank in tqdm(range(pp_size), desc="Processing pipeline stages"):
        need_embed = pp_rank == 0
        need_norm = pp_rank == pp_size - 1
        need_head = pp_rank == pp_size - 1
        layer_start = sum(layer_chunk_sizes[:pp_rank])
        
        # Create state dictionaries for each tensor parallel rank
        state_dicts = [{} for _ in range(tp_size)]
        
        # Process each layer in the current pipeline stage
        for layer_idx in range(layer_chunk_sizes[pp_rank]):
            src_layer = layer_idx + layer_start
            src_layer_name = f"layers.{src_layer}"
            layer_name = f"layers.{layer_idx}"
            
            # Get the layer from the model
            layer = model.layers[src_layer]
            
            # Process the layer's state dict
            layer_state_dict = layer.state_dict()
            
            for name, param in layer_state_dict.items():
                full_name = f"{layer_name}.{name}"
                
                # Extract the key for mapping
                key = name.split(".")[-2] if len(name.split(".")) > 1 else name
                
                if key in mapping:
                    dim = mapping[key]
                    
                    # Handle special cases
                    if key == "embed" and not need_embed:
                        continue
                    if key == "norm" and not need_norm:
                        continue
                    if key == "head" and not need_head:
                        continue
                    
                    # Shard the parameter for each tensor parallel rank
                    for i in range(tp_size):
                        new_param = param
                        
                        # Handle experts
                        if "experts" in name and "shared_experts" not in name:
                            idx = int(name.split(".")[-3])
                            if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                                continue
                        # Handle tensor parallelism
                        elif dim is not None:
                            assert param.size(dim) % tp_size == 0, f"Dimension {dim} of parameter {name} with shape {param.shape} must be divisible by {tp_size}"
                            shard_size = param.size(dim) // tp_size
                            new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                        
                        state_dicts[i][full_name] = new_param
        
        # Process embedding, norm, and head if needed
        if need_embed:
            embed = model.embed
            embed_state_dict = embed.state_dict()
            
            for name, param in embed_state_dict.items():
                full_name = f"embed.{name}"
                
                for i in range(tp_size):
                    dim = mapping["embed"]
                    shard_size = param.size(dim) // tp_size
                    new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    state_dicts[i][full_name] = new_param
        
        if need_norm:
            norm = model.norm
            norm_state_dict = norm.state_dict()
            
            for name, param in norm_state_dict.items():
                full_name = f"norm.{name}"
                
                for i in range(tp_size):
                    state_dicts[i][full_name] = param
        
        if need_head:
            head = model.head
            head_state_dict = head.state_dict()
            
            for name, param in head_state_dict.items():
                full_name = f"head.{name}"
                
                for i in range(tp_size):
                    dim = mapping["head"]
                    shard_size = param.size(dim) // tp_size
                    new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    state_dicts[i][full_name] = new_param
        
        # Save state dictionaries for each tensor parallel rank
        for tp_rank in range(tp_size):
            save_path = os.path.join(output_dir, f"model-pp{pp_size}-tp{tp_size}-pp_rank-{pp_rank}-tp_rank-{tp_rank}.safetensors")
            save_file(state_dicts[tp_rank], save_path)
            print(f"Saved shard to {save_path}")
    
    # Upload to GCS if requested
    if upload_to_gcs and run_name:
        upload_to_gcs_bucket(output_dir, run_name, "model")
        print(f"Uploaded model to GCS bucket: gs://manifestai-chessr1-checkpoints/{run_name}/model")


def upload_to_gcs_bucket(src_dir, run_name, dir):
    """Upload checkpoint to GCS."""
    os.environ["GSUTIL_PARALLEL_THREAD_COUNT"] = "8"
    os.environ["GSUTIL_PARALLEL_PROCESS_COUNT"] = "64"
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    auth_flag = f"-o 'Credentials:gs_service_key_file={creds_path}'" if creds_path else ""
    bucket_path = f"gs://manifestai-chessr1-checkpoints/{run_name}/{dir}"
    os.system(f"cd {os.path.dirname(src_dir)} && gsutil -q {auth_flag} -m cp -r {os.path.basename(src_dir)}/* {bucket_path}/ > /dev/null")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True, help="Path to the model configuration file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the sharded model")
    parser.add_argument("--tp_size", type=int, required=True, help="Tensor parallelism size")
    parser.add_argument("--pp_size", type=int, required=True, help="Pipeline parallelism size")
    parser.add_argument("--upload_to_gcs", action="store_true", help="Whether to upload the sharded model to GCS")
    parser.add_argument("--run_name", type=str, help="Run name for GCS upload, required if upload_to_gcs is True")
    
    args = parser.parse_args()
    
    if args.upload_to_gcs and not args.run_name:
        parser.error("--run_name is required when --upload_to_gcs is set")
    
    shard_model(
        model_config=args.model_config,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        output_dir=args.output_dir,
        upload_to_gcs=args.upload_to_gcs,
        run_name=args.run_name
    )
