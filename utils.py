import os
import time
import datetime
import threading
import torch
import torch.nn as nn
import torch.distributed as dist
import getpass
from dataclasses import dataclass
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.pipelining.schedules import _PipelineSchedule
from contextlib import contextmanager
from typing import Iterable, Optional, Callable, Any
import logger

GS_BUCKET = os.environ.get("GS_BUCKET_CHECKPOINT")
LOG_CABIN_URL = os.environ.get("LOG_CABIN_URL", "http://localhost:8080")

@dataclass
class DistributedInfo:
    world_rank: int  # global rank across all GPUs 
    world_size: int # total number of GPUs
    node_rank: int # rank of the node in the cluster
    node_size: int # number of nodes in the cluster
    local_rank: int # rank of the GPU within the node
    local_size: int # number of GPUs per node, usually 8
    dp_rank: int # rank of the GPU within the data parallel group, equal to world_rank for pure (fs)dp, will be 0 for pure tp
    pp_rank: int # rank of the GPU within the pipeline parallel group, equal to world_rank for pure (fs)dp, will be 0 for pure dp
    tp_rank: int # rank of the GPU within the tensor parallel group, 0 for pure (fs)dp
    dp_size: int # number of GPUs that forms a data parallel group, equal to world_size for pure (fs)dp
    pp_size: int # number of GPUs that forms a pipeline parallel group, equal to world_size for pure (fs)dp, used for splitting models
    real_pp_size: int # number of GPUs that forms a pipeline parallel group, equal to world_size for pure (fs)dp
    tp_size: int # number of GPUs that forms a tensor parallel group, 1 for pure (fs)dp
    real_dp_rank: int # the real dp rank, useful for HYBRID_SHARD
    model_init_rank: int # rank for initializing the model, models initialized on the same rank will be the same
    device_mesh: DeviceMesh # mesh of all GPUs, can be 2-D if both dp and tp are used
    dp_mesh: DeviceMesh # 1-D mesh of all GPUs in the data parallel group
    pp_mesh: DeviceMesh # 1-D mesh of all GPUs in the pipeline parallel group
    tp_mesh: DeviceMesh # 1-D mesh of all GPUs in the tensor parallel group

@dataclass
class TrainState:
    run_name: str
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    schedule: Optional[_PipelineSchedule]
    has_first_stage: bool
    has_last_stage: bool
    tokenizer: Any
    di: DistributedInfo
    config: dict
    device: torch.device
    get_batch: Callable
    compute_loss: Callable
    start_time: float
    iter_timer: float
    iter_num: int
    next_eval_at: int
    last_checkpoint_time: float
    interval_between_evals: int
    upload_thread: threading.Thread
    unrolls_dir: str
    ptype: torch.dtype
    checkpoint_dir: str
    datasets_dir: str
    loss_scaling_factor: float
    loss_eval_batches: int
    chess_eval_fraction: float
    puzzle_eval_fraction: float
    chat_eval_fraction: float


def rank_print(message, rank=None):
    world_rank = int(os.environ.get("RANK", "0"))
    if rank is None or world_rank == rank:
        print(f"[{world_rank}] {message}")


def local_print(message):
    """ Only print if local_rank is 0 """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_rank = int(os.environ.get("RANK", "0"))
    if local_rank == 0:
        print(f"[{world_rank}] {message}")


def upload_to_gcs(src_dir, run_name, dir, delete=False):
    """Upload checkpoint to GCS in a separate thread."""
    os.environ["GSUTIL_PARALLEL_THREAD_COUNT"] = "8"
    os.environ["GSUTIL_PARALLEL_PROCESS_COUNT"] = "64"
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    auth_flag = f"-o 'Credentials:gs_service_key_file={creds_path}'" if creds_path else ""
    bucket_path = f"{GS_BUCKET}/{run_name}/{dir}"
    os.system(f"cd {os.path.dirname(src_dir)} && gsutil -q {auth_flag} -m cp -r {os.path.basename(src_dir)}/* {bucket_path}/ > /dev/null {'&& rm -rf ' + src_dir if delete else ''}")


def calculate_params(model: torch.nn.Module):
    """Calculate the number of parameters in the model."""
    num_params = sum(p.numel() for p in model.parameters())
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return num_params, param_size


@contextmanager
def timing(name, print):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name} took {end - start:.6f} seconds")

# This is a hot-patch to torch.distributed.fsdp.fully_sharded_data_parallel._get_grad_norm
# so that it doesn't require all gradients to be in the same dtype. It's converted to float32
# before being returned anyway.
def _get_grad_norm(
    params: Iterable[nn.Parameter],
    norm_type: float,
    zero: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Return the gradient norm of parameters ``param`` s, where the gradients are viewed as a single vector.

    The returned norm is in FP32 even if parameters/gradients are in a low precision. This is because the downstream
    use of this return value is a reduction across ranks.
    """
    params_with_grad = [param for param in params if param.grad is not None]
    if len(params_with_grad) == 0:
        # Reuse a tensor for zero to avoid a GPU sync
        return zero
    grads = [param.grad for param in params_with_grad]
    # Compute the gradient norm in FP32, where we treat the gradients as a
    # single vector
    grad_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                torch.linalg.vector_norm(grad.detach().to(torch.float32), norm_type, dtype=torch.float32)
                for grad in grads
            ],
        ),
        norm_type,
        dtype=torch.float32,
    )
    return grad_norm.to(device=device)
torch.distributed.fsdp.fully_sharded_data_parallel._get_grad_norm = _get_grad_norm


def init_distributed_groups(dp_size=None, pp_size=None, tp_size=None, **kwargs) -> DistributedInfo:
    """ Takes care of the torch distribution setup, and returns a DistributedInfo object. 

        It's recommended to specify all of dp_size, pp_size, and tp_size. If not,
        all ranks in the world will first be allocated to the rightmost dimension that's unspecified.

    Args:
        dp_size: The size of the data parallel group. If None, the default is the world size divided by tp_size * pp_size.
        pp_size: The size of the pipeline parallel group. If None, the default is world_size / (dp_size * tp_size).
        tp_size: The size of the tensor parallel group. If None, the default is 1.

    Returns:
        A DistributedInfo object.
    """

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    world_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(f"{local_rank=} {local_size=} {world_rank=} {world_size=}")
    pp_rank = kwargs.get('pp_rank', None)
    node_rank = world_rank // local_size
    node_size = world_size // local_size
    hybrid_shard = kwargs.get('hybrid_shard', False)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=60))

    assert sum(x is not None for x in [dp_size, pp_size, tp_size]) >= 2, "At least two of dp_size, pp_size, or tp_size must be provided"
    
    real_pp_size = -1 if pp_rank is not None else pp_size
    grid = torch.arange(world_size).view((dp_size or -1, real_pp_size or -1, tp_size or -1))
    dp_size = grid.size(0)
    pp_size = grid.size(1) if pp_size is None else pp_size
    real_pp_size = grid.size(1)
    tp_size = grid.size(2)
    del grid

    rank_print(f"Initializing device mesh with dp_size: {dp_size}, pp_size: {pp_size}, tp_size: {tp_size}, real_pp_size: {real_pp_size}")
    device_mesh = init_device_mesh("cuda", (dp_size, real_pp_size, tp_size), mesh_dim_names=("dp", "pp", "tp"))
    dp_mesh = device_mesh["dp"]
    pp_mesh = device_mesh["pp"]
    tp_mesh = device_mesh["tp"]
    dp_rank = world_rank // (pp_size * tp_size)
    pp_rank = (world_rank // tp_size) % pp_size if pp_rank is None else pp_rank
    tp_rank = world_rank % tp_size
    if hybrid_shard:
        # In hybrid sharding, we are interpreting the tp_rank as part of dp_rank
        real_dp_rank = dp_rank * (pp_size * tp_size) + tp_rank
        model_init_rank = pp_rank
    else:
        real_dp_rank = dp_rank
        model_init_rank = tp_rank + pp_rank * tp_size

    rank_print(f"Initialized process group: {world_rank}/{world_size} on {node_rank}/{node_size} nodes (local: {local_rank}/{local_size}), dp_size: {dp_size}, pp_size: {pp_size}, real_pp_size: {real_pp_size}, tp_size: {tp_size}")
    rank_print(f"dp_rank: {dp_rank}, pp_rank: {pp_rank}, tp_rank: {tp_rank}")

    di = DistributedInfo(world_rank, world_size, node_rank, node_size, local_rank, local_size, dp_rank, pp_rank, tp_rank,
                         dp_size, pp_size, real_pp_size, tp_size, real_dp_rank, model_init_rank, device_mesh, dp_mesh, pp_mesh, tp_mesh)
    return di


def logger_init(config: dict,
                model_config: dict,
                di: DistributedInfo,
                should_log_to_server: bool = False,
                **kwargs):
    """Initialize logging for a new run.
    
    Args:
        config (dict): The configuration dictionary.
        model_config (dict): The model configuration dictionary.
        di (DistributedInfo): The distributed information.
        **kwargs: Additional keyword arguments, will be added to the info dictionary.
    """
    
    def sanitize_dict(d: dict) -> dict:
        return {k: v if isinstance(v, (int, float, str)) else str(v) for k, v in d.items()}
    
    user = getpass.getuser()
    user = "jacob" if user not in ["jacob", "sean"] else user

    training_config = {
        "per_gpu_batch_size": config['batch_size'],
    } | di.__dict__ | kwargs

    info = {
        **config,
        "model_init": "random" if config['use_random_init'] else "pretrained",
        "model_config": sanitize_dict(model_config),
        "training_config": sanitize_dict(training_config),
    }

    run_name = logger.init(
        name=config['run_name'],
        info=info,
        server_url=None if config['disable_remote_logging'] or not should_log_to_server else f"{LOG_CABIN_URL}/{user}/api"
    )
    if di.local_rank == 0:
        print(f"Starting training run: \033[94m{run_name}\033[0m")
        print("\n\033[1m=== Configuration ===\033[0m")
        for k, v in info.items():
            if not isinstance(v, dict):
                print(f"\033[94m{k}:\033[0m {v}")
        for k, v in info.items():
            if isinstance(v, dict):
                print(f"\n\033[95m{k.replace('_',' ').title()}:\033[0m")
                for sub_k, sub_v in v.items():
                    print(f"  \033[96m{sub_k.replace('_',' ').title()}:\033[0m {sub_v}")
        print()


@contextmanager
def pp_eval(schedule: _PipelineSchedule):
    """ Context manager for evaluating the model in pipeline parallel mode.
    
    This is necessary because pipeline parallel schedules have a custom loss function
    that is not compatible with the default loss function used elsewhere in the code.
    Also, torch's pipeline parallel implementation requires the shape of the output of
    each stage to be known in advance, which is accomplished by running a shape inference
    before the actual execution. During our evaluation, we might evaluate the model on
    a different shape than the one used during training, so we need to make sure that
    the cached shape information is cleared.
    """
    ori_loss_fn = schedule._loss_fn
    ori_n_microbatches = schedule._n_microbatches
    ori_has_backward = schedule._has_backward
    ori_stage_initialized = schedule._stage_initialized
    ori_stage_inputs_meta = schedule._stage.inputs_meta
    ori_stage_outputs_meta = schedule._stage._outputs_meta
    schedule._stage.clear_runtime_states()
    schedule._stage_initialized = False
    schedule._stage.inputs_meta = None
    schedule._stage._outputs_meta = None
    schedule._loss_fn = None
    schedule._n_microbatches = 1
    schedule._has_backward = False
    yield
    schedule._loss_fn = ori_loss_fn
    schedule._n_microbatches = ori_n_microbatches
    schedule._has_backward = ori_has_backward
    schedule._stage_initialized = ori_stage_initialized
    schedule._stage.inputs_meta = ori_stage_inputs_meta
    schedule._stage._outputs_meta = ori_stage_outputs_meta


def print_cuda_memory_usage(prefix: str, rank: int = None):
    """Prints the current CUDA memory usage."""
    allocated_memory = torch.cuda.memory_allocated() / (1024**3)
    reserved_memory = torch.cuda.memory_reserved() / (1024**3)
    max_memory = torch.cuda.max_memory_allocated() / (1024**3)

    rank_print(f"\n\033[1m=== {prefix} CUDA Memory Usage ===\033[0m", rank=rank)
    rank_print(f"{prefix} Allocated: {allocated_memory:.2f} GB", rank=rank)
    rank_print(f"{prefix} Reserved: {reserved_memory:.2f} GB", rank=rank)
    rank_print(f"{prefix} Max Allocated: {max_memory:.2f} GB", rank=rank)
    rank_print(f"{prefix} Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", rank=rank)
