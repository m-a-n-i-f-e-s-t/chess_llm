import math
import os
from functools import partial
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Any, Union, Iterable

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed.nn.functional as dist_F
from torch.amp import custom_fwd, custom_bwd
from torch.utils.checkpoint import checkpoint
from torch.distributed.device_mesh import DeviceMesh

from .kernel import act_quant, weight_dequant, fp8_gemm
from .parallel import all_gather


world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb", "fsdp"] = "fsdp"
group = None
NN_REDUCE = False
USE_MLA = True

def rank_print(message, rank=None):
    world_rank = int(os.getenv("RANK", "0"))
    if rank is None or world_rank == rank:
        print(f"[{world_rank}] {message}")


@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8", "fp32"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    balance_factor: float = 1e-5
    bias_update_speed: float = 0.5
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.
    # flags
    train: bool = False
    # init
    initializer_range: float = 0.02

class parallel_embed(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, x, weight, vocab_start_idx=None, vocab_end_idx=None, world_size=None, group=None):
        if world_size is not None and world_size > 1:
            mask = (x < vocab_start_idx) | (x >= vocab_end_idx)
            x = x - vocab_start_idx
            x = torch.where(mask, torch.zeros_like(x), x)
        else:
            mask = None
        y = F.embedding(x, weight)
        if world_size > 1:
            y = torch.where(mask.unsqueeze(-1), torch.zeros_like(y), y)
            dist.all_reduce(y, group=group)
        ctx.save_for_backward(x, weight, mask)
        ctx.world_size = world_size
        ctx.group = group
        ctx.vocab_start_idx = vocab_start_idx
        return y
    
    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, dy):
        x, weight, mask = ctx.saved_tensors
        world_size = ctx.world_size

        dw = torch.zeros_like(weight)
        if world_size is not None and world_size > 1:
            # [b, t]
            valid_mask = ~mask
            
            if valid_mask.any():
                # [valid_num,]
                x_valid = x.masked_select(valid_mask).long()
                # [valid_num, 2]
                flat_indices = valid_mask.nonzero()
                batch_indices = flat_indices[:, 0]
                seq_indices = flat_indices[:, 1]
                # [valid_num, d]
                dy_valid = dy[batch_indices, seq_indices]
                # [valid_num, d]
                dw.index_add_(0, x_valid, dy_valid)
        else:
            # [t,]
            x_flat = x.reshape(-1).long()
            # [t, d]
            dy_reshaped = dy.reshape(-1, dy.size(-1))
            # [v, d]
            dw.index_add_(0, x_flat, dy_reshaped)
        
        # No gradient for the input indices
        return None, dw, None, None, None, None


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        return parallel_embed.apply(x, self.weight, self.vocab_start_idx, self.vocab_end_idx, world_size, group)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and 
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve 
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version 
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    if weight.element_size() > 1:
        if bias is not None and bias.size() > 0:
            return F.linear(x, weight, bias)
        else:
            return F.linear(x, weight)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features))
        else:
            scale_out_features = out_features
            scale_in_features = in_features
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(scale_out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        if self.bias:
            torch.nn.init.zeros_(self.bias)
        if getattr(self, "scale", None) is not None:
            torch.nn.init.ones_(self.scale)
    

class column_parallel_linear(torch.autograd.Function):
    """
    Column parallel semantics:

    Y = X @ W^T + b
    = X @ [W1^T, W2^T] + [b1, b2] # column parallel
    = [X @ W1^T + b1, X @ W2^T + b2]
    = [Y1, Y2]

    dX = dY @ W
    = [dY1, dY2] @ [W1, W2]
    = dY1 @ W1 + dY2 @ W2 # reduction

    dW^T = X^T @ dY
    = X.T @ [dY1, dY2]
    = [X.T @ dY1, X.T @ dY2] # parallel
    dW = dY.T @ X
    = [dY1.T @ X; dY2.T @ X] # parallel

    db = dY.sum(dim=0)
    db1 = dY1.sum(dim=0)
    db2 = dY2.sum(dim=0)
    """
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, x, weight, bias, world_size, group, log=False):
        ctx.log = log
        ctx.world_size = world_size
        ctx.group = group
        ctx.save_for_backward(x, weight, bias)
        return linear(x, weight, bias)
    
    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, dy):
        world_size = ctx.world_size
        group = ctx.group
        x, weight, bias = ctx.saved_tensors

        dx = dy @ weight
        if world_size > 1:
            work = dist.all_reduce(dx, group=group, async_op=True)
        db = dy.sum(dim=0) if bias is not None else None
        dw = torch.matmul(dy.transpose(-1, -2), x)
        if world_size > 1:
            work.wait()
        return dx, dw, db, None, None, None
        


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        return column_parallel_linear.apply(x, self.weight, self.bias, world_size, group)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        if self.bias:
            torch.nn.init.zeros_(self.bias)
        if getattr(self, "scale", None) is not None:
            torch.nn.init.ones_(self.scale)


class row_parallel_linear(torch.autograd.Function):
    """
    Row parallel semantics:

    Y = X @ W^T + b
    = [X1, X2] @ [W1^T; W2^T] + b # row parallel
    = X1 @ W1 + X2 @ W2 + b

    dX = dY @ W
    = dY @ [W1, W2]
    = [dY @ W1, dY @ W2] # parallel
    dX1 = dY @ W1
    dX2 = dY @ W2

    dW^T = X.T @ dY
    = [X1.T; X2.T] @ dY
    = [X1.T @ dY; X2.T @ dY] # parallel
    dW = dY.T @ X
    = dY.T @ [X1, X2]
    = [dY.T @ X1, dY.T @ X2] # parallel
    dW1 = dY.T @ X1
    dW2 = dY.T @ X2

    db = dY.sum(dim=0)
    """
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, x, weight, bias, world_size, group):
        ctx.save_for_backward(x, weight, bias)
        y = linear(x, weight)
        if world_size > 1:
            dist.all_reduce(y, group=group)
        if bias is not None:
            y += bias
        return y
    
    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, dy):
        x, weight, bias = ctx.saved_tensors

        dx = dy @ weight
        dw = torch.matmul(dy.transpose(-1, -2), x)
        db = dy.sum(dim=0) if bias is not None else None

        return dx, dw, db, None, None


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        return row_parallel_linear.apply(x, self.weight, self.bias, world_size, group)
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        if self.bias:
            torch.nn.init.zeros_(self.bias)
        if getattr(self, "scale", None) is not None:
            torch.nn.init.ones_(self.scale)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        o = F.rms_norm(x, (self.dim,), self.weight, self.eps)
        return o

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.is_train = args.train
        self.attn_impl = attn_impl
        assert self.n_local_heads > 0, f"n_local_heads must be greater than 0, got {self.n_local_heads}"
        if self.attn_impl == "fsdp":
            self.c_attn = ColumnParallelLinear(self.dim, self.n_heads * (2*(self.qk_nope_head_dim + self.qk_rope_head_dim) + self.v_head_dim))
            self.c_proj = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
            self.softmax_scale = self.qk_head_dim ** -0.5
        else:
            if self.q_lora_rank == 0:
                self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
            else:
                self.wq_a = Linear(self.dim, self.q_lora_rank)
                self.q_norm = RMSNorm(self.q_lora_rank)
                self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
            self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
            self.kv_norm = RMSNorm(self.kv_lora_rank)
            self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
            self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
            self.softmax_scale = self.qk_head_dim ** -0.5
            if args.max_seq_len > args.original_seq_len:
                mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
                self.softmax_scale = self.softmax_scale * mscale * mscale

        if not self.is_train:
            self.register_kv_cache()
        else:
            self.has_kv_cache = False

    def register_kv_cache(self):
        if self.attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(self.max_batch_size, self.max_seq_len, self.n_local_heads, self.qk_head_dim, device='cuda'), persistent=False)
            self.register_buffer("v_cache", torch.zeros(self.max_batch_size, self.max_seq_len, self.n_local_heads, self.v_head_dim, device='cuda'), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(self.max_batch_size, self.max_seq_len, self.kv_lora_rank, device='cuda'), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(self.max_batch_size, self.max_seq_len, self.qk_rope_head_dim, device='cuda'), persistent=False)
        self.has_kv_cache = True

    def clear_kv_cache(self):
        if self.attn_impl == "naive":
            if hasattr(self, "k_cache"):
                self.register_buffer("k_cache", None, persistent=False)
            if hasattr(self, "v_cache"):
                self.register_buffer("v_cache", None, persistent=False)
        else:
            if hasattr(self, "kv_cache"):
                self.register_buffer("kv_cache", None, persistent=False)
            if hasattr(self, "pe_cache"):
                self.register_buffer("pe_cache", None, persistent=False)
        self.has_kv_cache = False

    def reset_parameters(self):
        if self.attn_impl == "fsdp":
            self.c_attn.reset_parameters()
            self.c_proj.reset_parameters()
        else:
            if self.q_lora_rank == 0:
                self.wq.reset_parameters()
            else:
                self.wq_a.reset_parameters()
                self.q_norm.reset_parameters()
                self.wq_b.reset_parameters()
            self.wkv_a.reset_parameters()
            self.kv_norm.reset_parameters()
            self.wkv_b.reset_parameters()
            self.wo.reset_parameters()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        
        bsz, seqlen, _ = x.size()

        if self.attn_impl == "fsdp":
            qkv = self.c_attn(x).view(bsz, seqlen, self.n_local_heads, -1)
            q, k, v = torch.split(qkv, [self.qk_head_dim, self.qk_head_dim, self.v_head_dim], dim=-1)
            q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            k_nope, k_pe = torch.split(k, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            q_pe = apply_rotary_emb(q_pe, freqs_cis)
            k_pe = apply_rotary_emb(k_pe, freqs_cis)
            q = torch.cat([q_nope, q_pe], dim=-1).transpose(-2, -3)
            k = torch.cat([k_nope, k_pe], dim=-1).transpose(-2, -3)
            v = v.transpose(-2, -3)
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, scale=self.softmax_scale)
            o = o.transpose(-2, -3).contiguous().view(bsz, seqlen, self.n_local_heads, self.v_head_dim)
            o = self.c_proj(o.flatten(2))
            return o

        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            # wq_a grad replicated, x grad replicated
            q = self.wq_a(x)
            # q_norm grad replicated, q grad replicated
            q = self.q_norm(q)
            # q grad replicated, wq_b grad sharded
            q = self.wq_b(q)
        # q grad sharded, q_pe grad sharded
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        # q grad sharded, q_pe grad sharded
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # q_pe grad sharded
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        # wkv_a grad replicated, x grad replicated
        kv = self.wkv_a(x)
        # kv grad replicated
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            if not self.is_train:
                self.k_cache[:bsz, start_pos:end_pos] = k
                self.v_cache[:bsz, start_pos:end_pos] = v
                scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
            else:
                scores = torch.einsum("bshd,bthd->bsht", q, k) * self.softmax_scale
        elif attn_impl == "absorb":
            # wkv_b grad sharded
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            # wkv_b grad sharded
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            # q_nope grad sharded, wkv_b grad sharded
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            if not self.is_train:
                self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
                self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
                scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                        torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
            else:
                # kv_norm grad replicated, kv grad replicated
                kv = self.kv_norm(kv)
                # kv grad replicated, k_pe grad replicated
                kv = reduce_in_backward.apply(kv, world_size, group)
                kv, k_pe = pass_through.apply(kv, k_pe)
                k_pe = reduce_in_backward.apply(k_pe, world_size, group)
                # q_nope grad sharded, kv grad sharded, q_pe grad sharded, k_pe grad sharded
                scores = (torch.einsum("bshc,btc->bsht", q_nope, kv) + 
                          torch.einsum("bshr,btr->bsht", q_pe, k_pe.squeeze(2))) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            if not self.is_train:
                x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
            else:
                x = torch.einsum("bsht,bthd->bshd", scores, v)
        else:
            if not self.is_train:
                x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            else:
                x = torch.einsum("bsht,btc->bshc", scores, kv)
        #                           x grad sharded, wkv_b grad sharded
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        #           x grad sharded, wo grad replicated
        x = self.wo(x.flatten(2))
        return x


def get_sinusoidal_embeddings(position, dim, device):
    """Generate sinusoidal positional embeddings."""
    # position is [B, T, nh]
    T = position.shape[1]
    div_term = (2. * math.pi) / (float(T) ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)).view(1, 1, 1, -1)
    sinusoid_inp = position.unsqueeze(-1) * div_term
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    return sin, cos # [B, T, nh, d]

def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), axis=-1)
    return x.view(*x.shape[:-2], x.shape[-2] * x.shape[-1])

def apply_rotary_position_embeddings(x, sincos):
    _, T, _, _ = x.shape
    sin, cos = sincos
    sin = sin.repeat_interleave(2, dim=3)[:, :T]
    cos = cos.repeat_interleave(2, dim=3)[:, :T]
    return ((x * cos) + (rotate_every_two(x) * sin)).to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        #         self.dim = args.dim
        # self.n_heads = args.n_heads
        # self.n_local_heads = args.n_heads // world_size
        # self.q_lora_rank = args.q_lora_rank
        # self.kv_lora_rank = args.kv_lora_rank
        # self.qk_nope_head_dim = args.qk_nope_head_dim
        # self.qk_rope_head_dim = args.qk_rope_head_dim
        # self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        # self.v_head_dim = args.v_head_dim
        # self.max_batch_size = args.max_batch_size
        # self.max_seq_len = args.max_seq_len
        # self.is_train = args.train
        # self.attn_impl = attn_impl
        self.n_head = config.n_heads
        self.n_embd = config.dim
        self.dropout = 0.0
        self.attention_kernel = 'sdpa'
        self.head_size = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.qhead_ratio = 1
        # key, query, value projections for all heads, but in a batch
        self.qkv_size = (self.qhead_ratio + 2) * self.n_head * self.head_size
        self.c_attn = ColumnParallelLinear(self.n_embd, self.qkv_size, bias=True)
        # output projection
        self.c_proj = RowParallelLinear(self.qhead_ratio * self.n_head * self.head_size, self.n_embd, bias=True)
        # regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        h = self.n_head // world_size
        hq = self.qhead_ratio * h
        d = self.head_size

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkvg = self.c_attn(x)

        qkv = qkvg[...,:self.qkv_size]
        q, k, v  = qkv.split([hq*d, h*d, h*d], dim=2)
        q = q.view(B, T, hq, d)
        k = k.view(B, T, h, d)
        v = v.view(B, T, h, d)

        # apply rotary position embeddings
        r = torch.arange(T, dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(2) # [1, T, 1]
        sincos = get_sinusoidal_embeddings(r, d, q.device)
        q = apply_rotary_position_embeddings(q, sincos)
        k = apply_rotary_position_embeddings(k, sincos)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        q = q.transpose(1, 2) # (B, nh, T, hs)
        k = k.transpose(1, 2) # (B, nh, T, hs)
        v = v.transpose(1, 2) # (B, nh, T, hs)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                                attn_mask=None,
                                                                dropout_p=0,
                                                                is_causal=True,
                                                                scale=1.0 / d**0.5,
                                                                enable_gqa=False)
        y = y.transpose(1, 2) # (B, T, nh, hs)
        y = y.contiguous().view(B, T, hq * d) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x)), 0.0

    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.w3.reset_parameters()


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias_update_speed = args.bias_update_speed
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.balance_factor = args.balance_factor
        self.register_buffer('bias', torch.zeros(args.n_routed_experts))
        self.register_buffer('expert_imbalance', torch.zeros(args.n_routed_experts), persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = linear(x, self.weight) # [batch_size * seq_len, n_routed_experts]
        tokens = x.size(0)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32) # [batch_size * seq_len, n_routed_experts]
        else:
            scores = scores.sigmoid() # [batch_size * seq_len, n_routed_experts]
        original_scores = scores.clone()
        if self.bias is not None:
            scores = scores + self.bias
        export_load = scores.detach().sum(dim=0) / tokens
        self.expert_imbalance = export_load - export_load.mean()
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1) # [tokens, n_groups, n_routed_experts/n_groups]
            if self.bias is None:
                group_scores = scores.amax(dim=-1) # [tokens, n_groups]
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1) # [tokens, n_groups] sum of top 2 scores for each group
            indices = group_scores.topk(self.topk_groups, dim=-1)[1] # [tokens, topk_groups] selected groups
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1] # [batch_size * seq_len, self.topk]
        weights = original_scores.gather(1, indices) # [batch_size * seq_len, self.topk]
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        f = torch.scatter(
            torch.zeros(scores.shape, dtype=indices.dtype, device=indices.device),
            dim=1,
            index=indices,
            src=torch.ones_like(indices)
        )
        p = original_scores.sum(dim=0) / tokens
        f = f.sum(dim=0) / tokens * self.n_routed_experts / self.n_activated_experts
        moe_loss = (p * f).sum() * self.balance_factor
        return weights.type_as(x), indices, moe_loss
    
    def balance_expert(self):
        """
        Balance each expert's load by adjusting the bias for the gate.
        Intended to be called by all ranks at each training iteration.

        Deepseek doesn't disclose the exact formula for balancing the expert load,
        so here we are using the following heuristic.

        b_i <- b_i - bias_update_speed * expert_imbalance_i
        """
        if self.bias is not None:
            self.bias -= self.bias_update_speed * self.expert_imbalance

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.w3.reset_parameters()


class reduce_in_backward(torch.autograd.Function):
    """
    Reduce the input tensor in the backward pass.
    """
    @staticmethod
    def forward(ctx, x, world_size, group):
        ctx.world_size = world_size
        ctx.group = group
        return x
    
    @staticmethod
    def backward(ctx, dy):
        if ctx.world_size > 1:
            dist.all_reduce(dy, group=ctx.group)
        return dy, None, None


class divide_in_backward(torch.autograd.Function):
    """
    Divide the gradient by the world size in the backward pass.
    """
    @staticmethod
    def forward(ctx, x, world_size, group):
        ctx.world_size = world_size
        ctx.group = group
        return x
    
    @staticmethod
    def backward(ctx, dy):
        if ctx.world_size > 1:
            dy /= ctx.world_size
        return dy, None, None
        

class pass_through(torch.autograd.Function):
    """
    Pass through the input tensor in the forward pass and backward pass.
    This is useful to ensure order of execution in the backward pass.
    """
    @staticmethod
    def forward(ctx, x, y):
        return x, y
    
    @staticmethod
    def backward(ctx, dx, dy):
        return dx, dy

class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    MoE semantics

    Y = \sum_{i=1}^{n} G(x, i) * P(x, i) * E(x, i)
    G(x, i) = \sigma(W_g[i] * x + b_g[i])
    P(x, i) = G(x, i) + bias(i) in topk(G(x, :), k)
    E(x, i) = W_e[i] * x + b_e[i]

    dE(x, i) = dY * P(x, i) * G(x, i)
    dG(x, i) = dY * E(x, i) * P(x, i)

    dx = \sum_{i=1}^{n} (dE(x, i) * ∂(E(x, i))/∂x + dG(x, i) * ∂(G(x, i))/∂x)

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.balance_factor = args.balance_factor
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
            torch.Tensor: MOE loss.
        """
        shape = x.size()
        x = x.view(-1, self.dim) # [tokens, dim]
        weights, indices, moe_loss = self.gate(x) # [tokens, n_activated_experts], [tokens, n_activated_experts], [1]
        weights = reduce_in_backward.apply(weights, world_size, group)
        weights, x = pass_through.apply(weights, x)
        x = reduce_in_backward.apply(x, world_size, group)  
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i) # [tokens], [tokens]
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        x = divide_in_backward.apply(x, world_size, group)
        z = self.shared_experts(x)[0]
        if world_size > 1:
            dist.all_reduce(y, group=group)
        z = z + y
        return z.view(shape), moe_loss

    def reset_parameters(self):
        self.shared_experts.reset_parameters()
        self.gate.reset_parameters()
        for expert in self.experts:
            expert.reset_parameters()


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        if USE_MLA:
            self.attn = MLA(args)
        else:
            self.attn = CausalSelfAttention(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, moe_loss: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            moe_loss (torch.Tensor): MoE loss.
            expert_load (Optional[torch.Tensor]): Expert load.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
            torch.Tensor: MoE loss.
        """
        if USE_MLA:
            x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        else:
            x = x + self.attn(self.attn_norm(x))
        y, y_moe_loss = self.ffn(self.ffn_norm(x))
        return x + y, moe_loss + y_moe_loss
    
    def reset_parameters(self):
        self.attn.reset_parameters()
        self.ffn.reset_parameters()
        self.attn_norm.reset_parameters()
        self.ffn_norm.reset_parameters()


class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """
    def __init__(self, args: ModelArgs, mesh: DeviceMesh, grad_checkpointing: bool = True):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        global world_size, rank, group
        # world_size = dist.get_world_size() if dist.is_initialized() else 1
        # rank = dist.get_rank() if dist.is_initialized() else 0
        if mesh is not None:
            world_size = mesh.size()
            rank = mesh.get_rank() % world_size
            group = mesh.get_group("tp")
        else:
            world_size = 1
            rank = 0
            group = None
        Linear.dtype = args.dtype # setting default dtype for Linear
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size)
        self.is_train = args.train
        self.grad_checkpointing = grad_checkpointing
        self.embed.weight = self.head.weight # tie the weights
        self.initializer_range = args.initializer_range
        self.n_dense_layers = args.n_dense_layers
        self.n_layers = args.n_layers
        self.dim = args.dim
        self.vocab_size = args.vocab_size
        self.args = args
        self.init_freqs_cis()
        self.init_weights()

    def init_freqs_cis(self):
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.args), persistent=False)

    def _forward(self, input_ids: torch.Tensor, moe_loss: torch.Tensor = None, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Transformer model.

        Args:
            input_ids (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            moe_loss (torch.Tensor, optional): MoE loss tensor with shape (1,). Defaults to None.
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        tokens = input_ids
        moe_loss = torch.zeros(1, device=tokens.device, dtype=torch.float32) if moe_loss is None else moe_loss
        seqlen = tokens.size(1)
        # if isinstance(start_pos, torch.Tensor):
            # start_pos = int(start_pos.item())
        h = self.embed(tokens) if self.embed else tokens
        # rank = torch.distributed.get_rank()
        # if activated_layers == 0:
        #     return h, moe_loss
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for i, layer in enumerate(self.layers):
            # true_layer = (rank // 8) * 2 + i
            # if true_layer + 1 > activated_layers:
            #     continue
            if self.grad_checkpointing and self.is_train and h.requires_grad:
                h, moe_loss = checkpoint(
                    layer,
                    h,
                    start_pos,
                    freqs_cis,
                    moe_loss,
                    mask,
                    use_reentrant=False)
            else:
                h, moe_loss = layer(h, start_pos, freqs_cis, moe_loss, mask)
            # if true_layer + 1 == activated_layers:
                # return h, moe_loss, start_pos
        h = self.norm(h) if self.norm is not None else h
        if not isinstance(start_pos, torch.Tensor):
            start_pos = torch.tensor(start_pos, device=h.device, dtype=torch.float32)
        if self.head:
            logits = self.head(h)
            if world_size > 1:
                # all_gather is used here instead of dist.all_reduce to make sure the backward pass is correct
                all_logits = all_gather(logits, group=group)
                # we can use dist.all_reduce here because the gradient of moe_loss is the same across
                # all ranks, so we save an extra communication
                dist.all_reduce(moe_loss, group=group)
                logits = torch.cat(all_logits, dim=-1)
            return logits, moe_loss
        return h, moe_loss
    
    def forward(self, input_ids: torch.Tensor, moe_loss: torch.Tensor = None, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.grad_checkpointing:
            return checkpoint(self._forward, input_ids, moe_loss, start_pos, use_reentrant=False)
        else:
            return self._forward(input_ids, moe_loss, start_pos)
    
    def init_weights(self):
        def _init_weights(module):
            if isinstance(module, (Linear, ParallelEmbedding, Gate)):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, (RMSNorm)):
                torch.nn.init.ones_(module.weight)
            if hasattr(module, 'scale') and module.scale is not None:
                torch.nn.init.ones_(module.scale)
        
        self.apply(_init_weights)

    def gradient_checkpointing_enable(self):
        self.grad_checkpointing = True

    def reset_parameters(self):
        def _reset_parameters(module):
            module.reset_parameters()
        self.apply(_reset_parameters)

    def train(self, mode=True):
        super().train(mode)
        if not self.is_train:
            # clear kv cache
            for layer in self.layers:
                layer.attn.clear_kv_cache()
                layer.attn.is_train = True
                layer.attn.has_kv_cache = False
        self.is_train = mode

    def eval(self):
        super().eval()
        self.is_train = False
        # register kv cache for decoding
        for layer in self.layers:
            if layer.attn.has_kv_cache:
                layer.attn.clear_kv_cache()
            layer.attn.register_kv_cache()
            layer.attn.is_train = False
            layer.attn.has_kv_cache = True


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    print(model(x).size())
