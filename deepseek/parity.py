import os
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as F
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
import time
from deepseek.model import Transformer, ModelArgs
from typing import Union

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

def diff(a, b, rtol=1e-3, atol=1e-5, assert_close=True, verbose=True):
    """ A diff function that helps debug numerical issues

    Args:
        a: torch.Tensor
        b: torch.Tensor
        rtol: float
        atol: float
        assert_close: bool
        verbose: bool
    Returns:
        bool: True if a and b are close, False otherwise
    """
    equal = torch.allclose(a, b, rtol=rtol, atol=atol)
    error_max = torch.max(torch.abs(a - b))
    error_hist = torch.histc(torch.abs(a - b), bins=100, min=0, max=1)
    
    # Calculate absolute error
    abs_diff = torch.abs(a - b)
    total_elements = a.numel()
    
    # Calculate relative error where b is non-zero
    b_nonzero = b != 0
    rel_diff = torch.zeros_like(abs_diff)
    rel_diff[b_nonzero] = abs_diff[b_nonzero] / torch.abs(b[b_nonzero])
    
    if verbose:
        print(f"Max absolute error: {error_max.item()}")
        print(f"Tensors are {'close' if equal else 'different'} according to torch.allclose")
        
        # Calculate thresholds for relative error table
        rel_thresholds = torch.logspace(
            torch.log10(torch.tensor(rtol)), 
            0.0, 
            steps=10
        )
        
        # Calculate thresholds for absolute error table
        abs_thresholds = torch.logspace(
            torch.log10(torch.tensor(atol)), 
            0.0, 
            steps=10
        )
        
        # Print relative error table
        print("\nRelative Error Table:")
        print("---------------------")
        print(f"{'Threshold':<12} {'% matched':<12} {'Element Count':<12}")
        print("-" * 36)
        for threshold in rel_thresholds:
            count = (rel_diff <= threshold).sum().item()
            percentage = 100.0 * count / total_elements
            print(f"{threshold.item():<12.6f} {percentage:<12.2f} {count:<12}")
        
        # Print absolute error table
        print("\nAbsolute Error Table:")
        print("---------------------")
        print(f"{'Threshold':<12} {'% matched':<12} {'Element Count':<12}")
        print("-" * 36)
        for threshold in abs_thresholds:
            count = (abs_diff <= threshold).sum().item()
            percentage = 100.0 * count / total_elements
            print(f"{threshold.item():<12.6f} {percentage:<12.2f} {count:<12}")
        
        # Print some examples of largest errors
        if not equal:
            n_samples = min(5, total_elements)
            print("\nLargest Errors:")
            flat_indices = torch.argsort(abs_diff.flatten(), descending=True)[:n_samples]
            for i in range(n_samples):
                idx = flat_indices[i]
                multi_idx = torch.unravel_index(idx, a.shape)
                multi_idx_str = ', '.join(map(str, [idx.item() for idx in multi_idx]))
                print(f"Index [{multi_idx_str}]: a={a[multi_idx].item()}, b={b[multi_idx].item()}, "
                      f"abs_diff={abs_diff[multi_idx].item()}, rel_diff={rel_diff[multi_idx].item()}")
    
    if assert_close:
        assert equal, f"Tensors are not close! Max absolute error: {error_max.item()}"
    
    return equal

def get_tensor_by_path(base_obj, attr_path):
    """
    Dynamically access a tensor in a PyTorch model using an attribute path string.
    
    Args:
        base_obj: The base object (e.g., model)
        attr_path: String path to the attribute (e.g., "embed.weight.grad")
    
    Returns:
        The tensor at the specified path
    """
    obj = base_obj
    for attr in attr_path.split('.'):
        obj = getattr(obj, attr)
    return obj

# "x_before_attn", "h_embedding" is failing
grads_to_test = {
    "replicate_act": [],
    "replicate_weight": ["norm.weight.grad", "layers.1.attn.wq_a.weight.grad", "layers.1.ffn.gate.weight.grad", "layers.1.ffn.experts.0.w1.weight.grad"],
    "shard_weight": [("embed.weight.grad", 0), ("layers.1.attn.wq_b.weight.grad", 0), ("layers.1.attn.wkv_b.weight.grad", 0)],
    "shard_act": []
}

def run(rank: int):
    if rank <= 1:
        torch.distributed.init_process_group(
            backend="nccl", init_method="tcp://localhost:12345",
            world_size=2, rank=rank,
        )
    else:
        torch.distributed.init_process_group(
            backend="nccl", init_method="tcp://localhost:12346",
            world_size=1, rank=0,
        )

    vocab_size = 128
    args = ModelArgs(
        vocab_size=vocab_size,
        dim=128,
        inter_dim=512,
        n_layers=16,
        n_dense_layers=1,
        # n_heads=16,
        # n_routed_experts=32,
        # n_shared_experts=1,
        # n_activated_experts=2,
        # n_expert_groups=1,
        q_lora_rank=32,
        kv_lora_rank=32,
        # qk_nope_head_dim=4,
        # qk_rope_head_dim=4,
        # v_head_dim=4,
        # original_seq_len=4,
        # rope_theta=1e6,
        # rope_factor=1.0,
        # beta_fast=0.5,
        # beta_slow=0.5,
        # mscale=1.0,
        train=True,
        # initializer_range=0.02,
        dtype="fp32"
    )
    if rank <= 1:
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("tp",))
    else:
        mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("tp",))
    torch.manual_seed(13)
    with torch.device(f"cuda:{rank}"):
        model = Transformer(args, mesh)

    if rank == 2:
        state_dict = model.state_dict()
        torch.save(state_dict, f"model_gold.pth")
    else:
        timeout, start = 60, time.time()
        while time.time() - start < timeout:
            try:
                state_dict = torch.load("model_gold.pth")
                break
            except Exception:
                time.sleep(1)
        if time.time() - start >= timeout:
            raise TimeoutError("Timeout waiting for model_gold.pth")
        rank_state_dict = {}
        for key, value in state_dict.items():
            name = key.split(".")[-2]
            if name in mapping and mapping[name] is not None and ("shared_experts" in key or "expert" not in key):
                dim = mapping[name]
                shard_size = value.shape[dim] // 2
                rank_state_dict[key] = value.narrow(dim, rank * shard_size, shard_size).contiguous()
            else:
                rank_state_dict[key] = value
        model.load_state_dict(rank_state_dict)

    model.zero_grad(set_to_none=True)

    torch.distributed.barrier(group=mesh.get_group("tp"))

    torch.manual_seed(13)
    x = torch.randint(0, vocab_size, (4, 1024), device=f"cuda:{rank}")
    y, moe_loss = model(x)
    loss = torch.nn.functional.cross_entropy(y.reshape(-1, vocab_size), x.reshape(-1)) + moe_loss
    loss.backward()
    
    # print(f"{rank=} {loss=} {model.embed.weight.grad=}")
    if rank < 2:
        # Use barrier to ensure ranks print in order
        if rank == 1:
            dist.barrier(group=mesh.get_group("tp"))
            
        for grad_path in grads_to_test['replicate_act']:
            full_path = f"{grad_path}_world2_rank{rank}.pth"
            gold_path = f"{grad_path}_world1_rank0.pth"
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"File {full_path} not found")
            grad = torch.load(full_path)
            gold_grad = torch.load(gold_path).to(grad.device)
            print(f"==============> Comparing {full_path} on rank {rank} <===============")
            diff(grad, gold_grad, rtol=1e-3, assert_close=True)
            
        if rank == 0:
            dist.barrier(group=mesh.get_group("tp"))
        dist.barrier(group=mesh.get_group("tp"))
        

        for grad_path, dim in grads_to_test['shard_act']:
            full_path = f"{grad_path}_world2_rank{rank}.pth"
            gold_path = f"{grad_path}_world1_rank0.pth"
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"File {full_path} not found")
            grad = torch.load(full_path)
            gold_grad = torch.load(gold_path).to(grad.device)
            
            grads = F.all_gather(grad, group=mesh.get_group("tp"))
            grad = torch.cat(grads, dim=dim)
            if rank == 0:
                print(f"==============> Comparing {full_path} on rank {rank} <===============")
                diff(grad, gold_grad, rtol=1e-3, assert_close=True)

        dist.barrier(group=mesh.get_group("tp"))

        for grad_path in grads_to_test['replicate_weight']:
            if rank == 0:
                grad = get_tensor_by_path(model, grad_path)
                gold_path = f"{grad_path}_gold.pth"
                gold_grad = torch.load(gold_path).to(grad.device)
                if rank == 1:
                    dist.barrier(group=mesh.get_group("tp"))
                print(f"==============> Comparing {grad_path} on rank {rank} <===============")
                diff(grad, gold_grad, rtol=1e-3, assert_close=True)

        dist.barrier(group=mesh.get_group("tp"))
            
        for grad_path, dim in grads_to_test['shard_weight']:
            grad = get_tensor_by_path(model, grad_path)
            grads = F.all_gather(grad, group=mesh.get_group("tp"))
            grad = torch.cat(grads, dim=dim)
            start_t = time.time()
            match = False
            if rank == 0:
                while time.time() - start_t < 10:
                    print("waiting for grad from rank 2")
                    try:
                        gold_grad = torch.load(f"{grad_path}_gold.pth").to(grad.device)
                        torch.set_printoptions(profile="full")
                        print(f"==============> Comparing {grad_path} on rank {rank} <===============")
                        diff(grad, gold_grad, rtol=1e-3, assert_close=True)
                        break
                    except FileNotFoundError:
                        time.sleep(1)
                    except AssertionError:
                        print("grad mismatch!")
                        os.remove(f"{grad_path}_gold.pth")
                        break
                if match:
                    print("match!")
                else:
                    print("timeout!")
                    if os.path.exists(f"{grad_path}_gold.pth"):
                        os.remove(f"{grad_path}_gold.pth")

            
    if rank == 2:
        for grad_path in grads_to_test['shard_weight']:
            if isinstance(grad_path, tuple):
                grad_path, dim = grad_path
            grad = get_tensor_by_path(model, grad_path)
            torch.save(grad, f"{grad_path}_gold.pth")
        for grad_path in grads_to_test['replicate_weight']:
            grad = get_tensor_by_path(model, grad_path)
            torch.save(grad, f"{grad_path}_gold.pth")

    dist.destroy_process_group()


if __name__ == "__main__":
    if os.path.exists("model_gold.pth"):
        os.remove("model_gold.pth")
    for grad_path in [*grads_to_test['shard_weight'], *grads_to_test['shard_act']]:
        if isinstance(grad_path, tuple):
            grad_path, dim = grad_path
        if os.path.exists(f"{grad_path}_gold.pth"):
            os.remove(f"{grad_path}_gold.pth")
    mp.spawn(run, nprocs=3)