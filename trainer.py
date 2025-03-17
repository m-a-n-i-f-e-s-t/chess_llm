import time
import threading
import os
import math
import gc
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup
from torch.nn import functional as F
from contextlib import contextmanager
from torch.distributed.pipelining.schedules import _PipelineSchedule
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, OptimStateKeyType
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp._common_utils import TrainingState
from torch.distributed.fsdp.fully_sharded_data_parallel import _get_grad_norm
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from tqdm import tqdm
from typing import Callable, Any, Union, Iterable, Optional, List
from utils import upload_to_gcs, rank_print, TrainState, pp_eval, local_print, timing, DistributedInfo
from unroll_on_eval import score_rollouts_on_eval, score_rollouts_on_puzzles
from sample import chat_sample
from contextlib import nullcontext
import logger

loss_eval_schedule = None
GS_BUCKET = os.environ.get("GS_BUCKET_CHECKPOINT")

def loss_eval(ts: TrainState, create_eval_schedule: Callable[[], _PipelineSchedule]) -> Callable[[torch.nn.Module], dict]:
    """
    Evaluates the model on training batch and return loss.
    """
    global loss_eval_schedule
    if loss_eval_schedule is None:
        loss_eval_schedule = create_eval_schedule()
    loss_total = torch.zeros(1, device=ts.device, dtype=ts.ptype)
    response_total = torch.zeros(1, device=ts.device, dtype=torch.int64)
    loop_range = range(ts.config['loss_eval_batches'])
    for i in tqdm(loop_range, desc=f"Evaluation") if ts.di.local_rank == 0 else loop_range:
        X, Y, response_mask, n_responses = ts.get_batch(ts.iter_num + i, with_response_mask=True)
        if loss_eval_schedule is None:
            loss, _ = ts.compute_loss(ts.model, X, Y)
            loss_total += torch.where(response_mask != 0, loss, 0.0).sum()
            response_total += n_responses
        else:
            with pp_eval(loss_eval_schedule):
                ts.model.train()
                if ts.has_first_stage:
                    output = loss_eval_schedule.step(X, moe_loss=torch.tensor(0.0, device=X.device), start_pos=0)
                else:
                    output = loss_eval_schedule.step()
                if output is not None: # only last stage returns output
                    assert ts.has_last_stage, "Only last stage should return output"
                    logits, _ = output # ignore moe loss during eval
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), reduction='none')
                    loss = loss.view(Y.shape)
                    loss_total += torch.where(response_mask != 0, loss, 0.0).sum()
                    response_total += n_responses

    # Sync loss across nodes
    if ts.di.world_size > 1:
        dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(response_total, op=dist.ReduceOp.SUM)

    loss = ts.loss_scaling_factor * loss_total / response_total
    
    return dict(
        loss_per_response=loss.item()
    )


def eval_and_log(ts: TrainState, create_eval_schedule: Callable[[], _PipelineSchedule]):
    """ Entrypoint for evaluating the model during training.
    """
    eval_start = time.time()
    ts.model.eval()
    eval_cpi = ts.config['eval_cpi']
    cpis = ts.config['cpi'].split(',')
    assert eval_cpi in cpis, f"eval_cpi {eval_cpi} must be in {cpis}"

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=ts.ptype) if ts.config['deepseek'] else nullcontext():
            with timing("loss_eval", local_print):
                # get loss eval
                loss_eval_results = loss_eval(ts, create_eval_schedule)
            with timing("chess_eval", local_print):
                # get chess eval
                eval_unrolls, chess_eval_results = score_rollouts_on_eval(ts.model, create_eval_schedule, ts.tokenizer, eval_cpi, greedy=True,
                                                fraction=ts.config['chess_eval_fraction'],
                                                seq_len=ts.config['chess_eval_seq_len'],
                                                dp_rank=ts.di.dp_rank,
                                                dp_size=ts.di.dp_size,
                                                dp_group=ts.di.dp_mesh.get_group(),
                                                has_first_stage=ts.has_first_stage,
                                                has_last_stage=ts.has_last_stage,
                                                batch_size=ts.config['chess_eval_batch_size'])
            with timing("puzzle_eval", local_print):
                puzzle_unrolls, puzzle_eval_results = score_rollouts_on_puzzles(ts.model, create_eval_schedule, ts.tokenizer, eval_cpi, greedy=True,
                                                    fraction=ts.config['puzzle_eval_fraction'],
                                                    seq_len=ts.config['chess_eval_seq_len'],
                                                    dp_rank=ts.di.dp_rank,
                                                    dp_size=ts.di.dp_size,
                                                    dp_group=ts.di.dp_mesh.get_group(),
                                                    has_first_stage=ts.has_first_stage,
                                                    has_last_stage=ts.has_last_stage,
                                                    batch_size=ts.config['chess_eval_batch_size'])
            with timing("chat_eval", local_print):
                chat_examples = chat_sample(ts, create_eval_schedule)
    
    eval_duration = time.time() - eval_start
    
    # Only one rank across all GPUs should log to server, but all local_ranks==0 should log to terminal
    if ts.di.local_rank==0 and ts.di.pp_rank == ts.di.pp_size - 1:
        logger.log("eval", {
            "total_hours": (time.time() - ts.start_time) / 3600,
            "iter": ts.iter_num,
            **loss_eval_results,
            **chess_eval_results,
            **puzzle_eval_results,
            "duration": eval_duration
        })
        # Convert tokens back to text and save
        iter_unrolls_dir = f"{ts.unrolls_dir}/{ts.iter_num:08d}"
        os.makedirs(iter_unrolls_dir, exist_ok=True)
        GREEN = "\033[92m"; RESET = "\033[0m"
        print(f"{RESET}Example unrolled text: {GREEN}" + eval_unrolls[0] + f"{RESET}")
        for i, example in enumerate(chat_examples):
            print(f"{RESET}Example chat {i}: {GREEN}" + example + f"{RESET}")
        for name, unrolls in zip(['eval_unrolls', 'puzzle_unrolls'], [eval_unrolls, puzzle_unrolls]):
            unroll_file = f"{iter_unrolls_dir}/{name}.txt"
            with open(unroll_file, "w", encoding='utf-8') as f:
                f.write("\n\n".join(unrolls))
            unroll_thread = threading.Thread(
                target=upload_to_gcs,
                args=(iter_unrolls_dir, ts.run_name, f"{name}/{ts.iter_num:08d}"),
                daemon=True
            )
            unroll_thread.start()

    # Sync all processes after evaluation
    rank_print(f"Syncing all processes after evaluation")
    dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()

    # Calculate next evaluation iteration using log spacing
    ts.next_eval_at = ts.iter_num + min(ts.interval_between_evals, ts.config['max_eval_interval'])
    ts.interval_between_evals = int(ts.interval_between_evals * ts.config['eval_spacing'])
    ts.model.train()
    torch.cuda.empty_cache()


def checkpoint_and_upload(ts: TrainState):
    """ Entrypoint for checkpointing and uploading the model during training.
    """
    iter_checkpoint_dir = f"{ts.checkpoint_dir}/{ts.iter_num:08d}"
    if ts.di.local_rank == 0:
        # Create checkpoint directory
        os.makedirs(iter_checkpoint_dir, exist_ok=True)
        # Wait for any previous upload to complete before starting new checkpoint
        if ts.upload_thread is not None and ts.upload_thread.is_alive():
            if ts.di.world_rank == 0:
                rank_print("Waiting for previous upload to complete...")
            ts.upload_thread.join()
    
    dist.barrier()
    rank_print(f"\033[38;5;208mSaving sharded checkpoint to {iter_checkpoint_dir}\033[0m")

    # Save sharded model state
    ctx = FSDP.state_dict_type(ts.model, StateDictType.SHARDED_STATE_DICT) if isinstance(ts.model, FSDP) else nullcontext()
    with ctx:
        sharded_state_dict = ts.model.state_dict()
        torch.save(sharded_state_dict, f"{iter_checkpoint_dir}/model_shard_tp{ts.di.tp_rank}-of-{ts.di.tp_size}_pp{ts.di.pp_rank}-of-{ts.di.pp_size}_dp{ts.di.dp_rank}-of-{ts.di.dp_size}.pt")
    if isinstance(ts.optimizer, FSDP):
        optim_state = FSDP.full_optim_state_dict(ts.model, ts.optimizer)
    else:
        optim_state = ts.optimizer.state_dict()
    torch.save(optim_state, f"{iter_checkpoint_dir}/optim_shard_tp{ts.di.tp_rank}-of-{ts.di.tp_size}_pp{ts.di.pp_rank}-of-{ts.di.pp_size}_dp{ts.di.dp_rank}-of-{ts.di.dp_size}.pt")
    # Save scheduler and metadata on local rank 0 only
    if ts.di.world_rank == 0:
        torch.save({
            'scheduler_state_dict': ts.scheduler.state_dict(),
            'iter_num': ts.iter_num,
            'config': ts.config
        }, f"{iter_checkpoint_dir}/meta.pt")
    
    # Sync nodes after checkpoint save
    dist.barrier()
    rank_print(f"[Rank {ts.di.world_rank}]  \033[38;5;208m Checkpoint saved successfully.\033[0m")

    if ts.di.local_rank == 0:
        # Start background upload thread
        ts.upload_thread = threading.Thread(
            target=upload_to_gcs,
            args=(iter_checkpoint_dir, ts.run_name, f"checkpoints/{ts.iter_num:08d}", True),
            daemon=True
        )
        rank_print(f"Starting upload thread for checkpoint {ts.iter_num}")
        ts.upload_thread.start()

    dist.barrier()
    ts.last_checkpoint_time = time.time()


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, resume_from: str, ts: TrainState, config: dict):
    """ Load a checkpoint from gcloud
    """
    di = ts.di
    checkpoints_dir = f'/data/checkpoint/{resume_from}/checkpoints'
    checkpoint_url = f'{GS_BUCKET}/{resume_from}/checkpoints'
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    auth_flag = f"-o 'Credentials:gs_service_key_file={creds_path}'" if creds_path else ""

    # check if the checkpoint exists
    ls_cmd = f"gsutil -q {auth_flag} ls {checkpoint_url}/"
    rank_print(f"Checking if checkpoint {checkpoints_dir} exists")
    checkpoints = os.popen(ls_cmd).read().strip().split('\n')
    if len(checkpoints) == 0:
        raise ValueError(f"Checkpoint {checkpoints_dir} does not exist")
    checkpoints_list = '\n'.join(checkpoints)
    local_print(f"Found {len(checkpoints)} checkpoints in {checkpoint_url}: {checkpoints_list}")
    latest_checkpoint = checkpoints[-1].strip('/').split('/')[-1]
    rank_print(f"Using checkpoint {checkpoint_url}/{latest_checkpoint}")

    # download the sharded checkpoint
    os.makedirs(checkpoints_dir, exist_ok=True)
    model_checkpoint = f"model_shard_tp{di.tp_rank}-of-{di.tp_size}_pp{di.pp_rank}-of-{di.pp_size}_dp{di.dp_rank}-of-{di.dp_size}.pt"
    optim_checkpoint = f"optim_shard_tp{di.tp_rank}-of-{di.tp_size}_pp{di.pp_rank}-of-{di.pp_size}_dp{di.dp_rank}-of-{di.dp_size}.pt"
    if not os.path.exists(f"{checkpoints_dir}/{model_checkpoint}"):
        rank_print(f"Downloading checkpoint {checkpoint_url}/{latest_checkpoint}/{model_checkpoint} to {checkpoints_dir}/{model_checkpoint}")
        os.system(f"gsutil {auth_flag} -m cp {checkpoint_url}/{latest_checkpoint}/{model_checkpoint} {checkpoints_dir}/{model_checkpoint}")
    else:
        rank_print(f"Checkpoint {checkpoints_dir}/{model_checkpoint} already exists")
    if not os.path.exists(f"{checkpoints_dir}/{optim_checkpoint}"):
        rank_print(f"Downloading checkpoint {checkpoint_url}/{latest_checkpoint}/{optim_checkpoint} to {checkpoints_dir}/{optim_checkpoint}")
        os.system(f"gsutil {auth_flag} -m cp {checkpoint_url}/{latest_checkpoint}/{optim_checkpoint} {checkpoints_dir}/{optim_checkpoint}")
    else:
        rank_print(f"Checkpoint {checkpoints_dir}/{optim_checkpoint} already exists")

    # load the checkpoint
    model_state_dict = torch.load(f"{checkpoints_dir}/{model_checkpoint}", map_location=torch.device("cuda"), weights_only=False)
    # optim_state_dict = torch.load(f"{checkpoints_dir}/{optim_checkpoint}", map_location=torch.device("cuda"), weights_only=False)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT) if isinstance(model, FSDP) else nullcontext():
        model.load_state_dict(model_state_dict)
    # if torch.distributed.get_rank() % 8 == 0:
    #     print(f"{optimizer.state_dict()['param_groups'].keys()=}")
    #     print(f"{optimizer.state_dict()['state'].keys()=}")
    # rekeyed_optim_state_dict = FSDP.rekey_optim_state_dict(optim_state_dict, OptimStateKeyType.PARAM_NAME, model=model, optim_input=optimizer.state_dict()['state'], optim=optimizer)
    # optim_state_to_load = FSDP.flatten_sharded_optim_state_dict(optim_state_dict, model, optim=optimizer)
    # optimizer.load_state_dict(optim_state_to_load)
    ts.iter_num = config['resume_from_iter']
    ts.next_eval_at = ts.iter_num + min(ts.interval_between_evals, ts.config['max_eval_interval'])
    ts.last_checkpoint_time = time.time()
    rank_print("Checkpoint resumed successfully")
    
    return model, optimizer, ts


def fsdp_grad_norm(model: FSDP, norm_type: float = 2.0, error_if_nonfinite: bool = False) -> torch.Tensor:
    """ Compute the gradient norm of an FSDP model. Borrowed from torch.distributed.fsdp.fully_sharded_data_parallel.py
    """
    _lazy_init(model, model)
    if not model._is_root:
        raise RuntimeError(
            "`clip_grad_norm_()` should only be called on the root FSDP instance"
        )
    if model._zero_scalar is None:
        model._zero_scalar = torch.tensor(0.0, device=model.compute_device)
    model._assert_state(TrainingState.IDLE)
    # If every FSDP instance uses `NO_SHARD`, then we can directly use
    # the normal `nn.utils` one targeting local gradients
    all_no_shard = all(
        not handle.uses_sharded_strategy for handle in model._all_handles
    )
    if all_no_shard:
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        total_norm = torch.nn.utils.get_total_norm(
            grads, norm_type, error_if_nonfinite
        )
    # Otherwise, there exists some FSDP instance using a sharded strategy,
    # where sharded and non-sharded parameters must be handled separately
    norm_type = float(norm_type)
    sharded_params_set = set()
    nonsharded_params_set = set()  # `NO_SHARD` or not FSDP-managed
    # Make sure to compute the local norm using lists for deterministic
    # iteration order and hence deterministic total norm computation
    sharded_params = []
    nonsharded_params = []
    grads: List[torch.Tensor] = []
    for handle in model._all_handles:
        if handle.uses_sharded_strategy:
            target_set = sharded_params_set
            target_list = sharded_params
        else:
            target_set = nonsharded_params_set
            target_list = nonsharded_params
        if handle._use_orig_params:
            for param in handle.flat_param._params:
                if param not in target_set:
                    target_set.add(param)
                    target_list.append(param)
                    if param.grad is not None:
                        grads.append(param.grad)
        else:
            if handle.flat_param not in target_set:
                target_set.add(handle.flat_param)
                target_list.append(handle.flat_param)
                if handle.flat_param.grad is not None:
                    grads.append(handle.flat_param.grad)
    for param in model.parameters():
        not_fsdp_managed = (
            param not in sharded_params_set and param not in nonsharded_params_set
        )
        if not_fsdp_managed:
            nonsharded_params_set.add(param)
            nonsharded_params.append(param)
            if param.grad is not None:
                grads.append(param.grad)
    # Compute local norms (forced to be in FP32)
    local_sharded_norm = _get_grad_norm(
        sharded_params, norm_type, model._zero_scalar, model.compute_device
    )
    local_nonsharded_norm = (
        _get_grad_norm(
            nonsharded_params, norm_type, model._zero_scalar, model.compute_device
        )
        if nonsharded_params
        else None
    )
    # Reconstruct the total gradient norm depending on the norm type
    if norm_type == math.inf:
        total_norm = (
            torch.maximum(local_sharded_norm, local_nonsharded_norm)
            if local_nonsharded_norm is not None
            else local_sharded_norm
        )
        dist.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.MAX, group=model.process_group
        )
    else:
        total_norm = local_sharded_norm**norm_type
        dist.all_reduce(total_norm, group=model.process_group)
        # All-reducing the local non-sharded norm would count it an extra
        # world-size-many times
        if local_nonsharded_norm is not None:
            total_norm += local_nonsharded_norm**norm_type
        total_norm = total_norm ** (1.0 / norm_type)
    if model.cpu_offload.offload_params:
        total_norm = total_norm.cpu()
    return total_norm


# Borrowed from https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py
@torch.no_grad()
def clip_grad_norm_(
    model: torch.nn.Module,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
    groups: Optional[Iterable[ProcessGroup]] = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        group: process group to reduce gradient norm across.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    # if not isinstance(model, FSDP):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
            grads, norm_type, error_if_nonfinite, foreach
        )
    # else:
    #     total_norm = fsdp_grad_norm(model, norm_type, error_if_nonfinite)

    assert not isinstance(total_norm, DTensor), "total_norm is a DTensor, this should not happen"

    if groups is not None:
        for group in groups:
            if math.isinf(norm_type):
                dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=group)
            else:
                total_norm **= norm_type
                dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=group)
                total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(model.parameters(), max_norm, total_norm, foreach)
    return total_norm