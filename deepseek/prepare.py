import os
import torch
import shutil
from tqdm import tqdm
from pathlib import Path
from glob import glob
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

def download_raw_weights(model="deepseek-ai/DeepSeek-R1", save_path="/data/model"):
    """ Downloads the raw weights from huggingface.

        Args:
            model: model name
            save_path: path to save the model weights

        Returns:
            model_path: path to the directory containing model weights and configs
    """
    model_path = Path(save_path) / model.replace("/", "_")
    os.system(f"huggingface-cli download {model} {model_path}")
    return model_path


def convert_weights(model_path: str, save_path: str, n_experts: int, mp: int, tp_ranks: list[int]):
    """ Split weights from monolithic weights to different tp ranks

        Args:
            model_path: path to the model weights
            save_path: path to save the converted weights
            n_experts: number of experts
            mp: number of model parallel ranks (total tp ranks)
            tp_ranks: list of tp ranks

        Returns:
            None
    """
    torch.set_num_threads(16)
    n_local_experts = n_experts // mp
    state_dicts = [{} for _ in range(len(tp_ranks))]

    for file in tqdm(glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "model.layers.61" in name:
                    continue
                param: torch.Tensor = f.get_tensor(name)
                if name.startswith("model."):
                    name = name[len("model."):]
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                key = name.split(".")[-2]
                assert key in mapping, f"Key {key} not found in mapping"
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)
                for i in tp_ranks:
                    new_param = param
                    if "experts" in name and "shared_experts" not in name:
                        idx = int(name.split(".")[-3])
                        if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                            continue
                    elif dim is not None:
                        assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
                        shard_size = param.size(dim) // mp
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    state_dicts[i][name] = new_param

    os.makedirs(save_path, exist_ok=True)

    for i in tp_ranks:
        save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))

    for file_path in glob(os.path.join(model_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)
