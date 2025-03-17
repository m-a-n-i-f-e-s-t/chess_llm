import os
import shutil
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file


mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
}

def split_layers(n_layers, tp, pp):
    base = n_layers // pp
    remainder = n_layers % pp
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(pp)]
    return chunk_sizes


def main(hf_ckpt_path, save_path, n_experts, tp, pp, n_layers):
    """
    Converts and saves model checkpoint files into a specified format.

    Args:
        hf_ckpt_path (str): Path to the directory containing the input checkpoint files.
        save_path (str): Path to the directory where the converted checkpoint files will be saved.
        n_experts (int): Total number of experts in the model.
        tp (int): Tensor parallelism factor.
        pp (int): Pipeline parallelism factor.
        n_layers (int): Total number of layers in the model.

    Returns:
        None
    """
    torch.set_num_threads(16)
    n_local_experts = n_experts // tp
    layer_chunk_sizes = split_layers(n_layers, tp, pp)

    layer_idx = 0

    with open(os.path.join(hf_ckpt_path, "model.safetensors.index.json"), "r") as f:
        model_index = json.load(f)

    for pp_rank in tqdm(range(pp), desc="Processing pipeline stages"):
        need_embed = pp_rank == 0
        need_norm = pp_rank == pp - 1
        need_head = pp_rank == pp - 1
        layer_start = sum(layer_chunk_sizes[:pp_rank])

        state_dicts = [{} for _ in range(tp)]

        files_to_load = set()

        for layer_idx in range(layer_chunk_sizes[pp_rank]):
            src_layer = layer_idx + layer_start
            src_layer_name = f"model.layers.{src_layer}"
            layer_name = f"model.layers.{layer_idx}"
            files_to_load |= set([model_index['weight_map'][key] for key in model_index['weight_map'].keys() if src_layer_name in key])

        for file_path in tqdm(files_to_load, desc="Processing files for pp rank %d" % pp_rank):
            file_path = os.path.join(hf_ckpt_path, file_path)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    assert "model.layers.61" not in name, "model.layers.61 shouldn't be here, set n_layers to 60 instead"
                    param: torch.Tensor = f.get_tensor(name)
                    if name.startswith("model."):
                        name = name[len("model."):]
                    name = name.replace("self_attn", "attn")
                    name = name.replace("mlp", "ffn")
                    name = name.replace("weight_scale_inv", "scale")
                    name = name.replace("e_score_correction_bias", "bias")
                    name = name.replace(src_layer_name, layer_name)
                    key = name.split(".")[-2]
                    assert key in mapping, f"Key {key} not found in mapping"
                    new_key, dim = mapping[key]
                    if new_key == "embed" and not need_embed:
                        continue
                    if new_key == "norm" and not need_norm:
                        continue
                    if new_key == "head" and not need_head:
                        continue
                    name = name.replace(key, new_key)
                    for i in range(tp):
                        new_param = param
                        if "experts" in name and "shared_experts" not in name:
                            idx = int(name.split(".")[-3])
                            if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                                continue
                        elif dim is not None:
                            assert param.size(dim) % tp == 0, f"Dimension {dim} must be divisible by {tp}"
                            shard_size = param.size(dim) // tp
                            new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                        state_dicts[i][name] = new_param

        os.makedirs(save_path, exist_ok=True)

        for tp_rank in range(tp):
            save_file(state_dicts[tp_rank], os.path.join(save_path, f"model-pp{pp}-tp{tp}-pp_rank-{pp_rank}-tp_rank-{tp_rank}.safetensors"))

    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--n-layers", type=int, required=True)

    args = parser.parse_args()
    assert args.n_experts % args.tp == 0, "Number of experts must be divisible by tensor parallelism"
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.tp, args.pp, args.n_layers)
