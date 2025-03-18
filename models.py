import os
import torch
import json
import gc
import time
import tempfile
from functools import partial
from torch.nn import functional as F
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torch.distributed.pipelining.microbatch import TensorChunkSpec, _Replicate, sum_reducer
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers import AutoModelForCausalLM, AutoConfig
from deepseek.model import Transformer as DeepSeekModel, ModelArgs, Block, Gate, RMSNorm, ParallelEmbedding
from safetensors.torch import load_file, load_model
from utils import timing, rank_print, DistributedInfo, print_cuda_memory_usage

microbatch_start = 0
GS_BUCKET = os.environ.get("GS_BUCKET_CHECKPOINT")
TS_MAPPING = {
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

def print_deepseek_model_sizes(model: DeepSeekModel, di: DistributedInfo):
    def estimate_model_size(module, depth=0, max_depth=3):
        total_bytes = 0
        for name, param in module.named_parameters(recurse=True):
            # Assuming model uses bfloat16 (2 bytes per parameter)
            bytes = param.numel() * {torch.bfloat16: 2, torch.float32: 4, torch.float16: 2, torch.int8: 1}[param.dtype]
            total_bytes += bytes
        
        if depth < max_depth:
            for name, child in module.named_children():
                child_bytes = estimate_model_size(child, depth + 1, max_depth)
                if child_bytes > 0:
                    gb = child_bytes / (1024**3)
                    print("  " * depth + f"{name}: {gb:.2f} GB")
        
        return total_bytes

    if di.local_rank == 0:
        print("\n\033[1m=== Model Size ===\033[0m")
        total_bytes = estimate_model_size(model)
        total_gb = total_bytes / (1024**3)
        print(f"\nTotal model size: {total_gb:.2f} GB")
        print(f"model has {len(model.layers)} layers")


def compute_loss(model, X, Y):
    # Do this myself because I don't trust the model's internal loss function
    if isinstance(model, DeepSeekModel) or (isinstance(model, FSDP) and isinstance(model.module, DeepSeekModel)):
        logits, moe_loss = model(X)
        logits = torch.log_softmax(logits, dim=-1).float()
    else:
        outputs, moe_loss = model(input_ids=X), torch.zeros(1, device=X.device, dtype=torch.float32)
        logits = torch.log_softmax(outputs.logits, dim=-1).float()
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), reduction='none')
    loss = loss.view(Y.shape)
    return loss, moe_loss


def loss_fn(output, target, response_mask: torch.Tensor, gradient_accumulation: int, n_responses: int, loss_scaling_factor: float):
    """ Given a model output and a target, compute the loss for training.
    """
    global microbatch_start
    if isinstance(output, tuple):
        logits, moe_loss = output[0], output[1]
    else:
        logits = output
        moe_loss = torch.zeros(1, device=logits.device, dtype=torch.float32)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction='none')
    loss = loss.view(target.shape)
    loss = loss_scaling_factor * loss / (gradient_accumulation * n_responses)
    moe_loss = loss_scaling_factor * moe_loss / (gradient_accumulation * n_responses)
    microbatch = loss.size(0) # pp might split the batch into multiple microbatches
    # rank_print(f"{microbatch_start=} \n {microbatch=} \n {response_mask=} \n {loss=} \n {moe_loss=} \n {target=} \n {logits=}")
    loss = torch.where(response_mask[microbatch_start:microbatch_start+microbatch] != 0, loss, 0.0).sum() + moe_loss.sum()
    microbatch_start += microbatch
    return loss


def setup_pp(model: torch.nn.Module,
             di: DistributedInfo,
             n_microbatches: int,
             use_random_init: bool = True,
             pre_inited: bool = False,
             resume_from: str = None,
             initialize_device: str = 'cuda'):
    """ Sets up the pipeline parallel group.
    """
    if di.pp_size == 1:
        return model

    n_layers = model.n_layers

    # Basic sanity check
    assert di.pp_size <= n_layers, "too many pipeline stages for too few layers"

    # Compute how many layers each rank should handle
    base = n_layers // di.pp_size
    remainder = n_layers % di.pp_size

    # List of chunk sizes—some ranks get an extra layer until the remainder is exhausted
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(di.pp_size)]

    # Compute starting and ending indices for this rank
    start_layer_idx = sum(chunk_sizes[:di.pp_rank])
    end_layer_idx = start_layer_idx + chunk_sizes[di.pp_rank]

    # Prune the layers in-place for this pipeline rank
    model.layers = model.layers[start_layer_idx:end_layer_idx]
    if di.pp_rank == 0:
        del model.head
        del model.norm
        model.head = None
        model.norm = None
    elif di.pp_rank < di.pp_size - 1:
        del model.embed
        del model.norm
        del model.head
        model.embed = None
        model.norm = None
        model.head = None
    else:
        del model.embed
        model.embed = None
    gc.collect()
    torch.cuda.empty_cache()

    if not pre_inited:
        model.to_empty(device=torch.device(initialize_device))

    if use_random_init:
        # we want to have different seeds for each tp rank
        if not pre_inited:
            with torch.random.fork_rng():
                torch.manual_seed(1337 + di.model_init_rank)
                model.init_weights()
        with torch.device(initialize_device):
            model.init_freqs_cis()
    elif resume_from is None:
        init_dir = os.environ.get("INIT_DIR", '/data/checkpoint/init')
        os.makedirs(init_dir, exist_ok=True)
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        auth_flag = f"-o 'Credentials:gs_service_key_file={creds_path}'" if creds_path else ""
        checkpoint_name = f"model-pp{di.pp_size}-tp{di.tp_size}-pp_rank-{di.pp_rank}-tp_rank-{di.tp_rank}{'-redo' if di.pp_rank == 29 else ''}.safetensors"
        bucket_path = f"{GS_BUCKET}/deepseek-r1-original/bf16/pp{di.pp_size}tp{di.tp_size}/{checkpoint_name}"

        # check if the checkpoint exists
        ls_cmd = f"gsutil -q {auth_flag} ls {bucket_path}"
        rank_print(f"Checking if checkpoint {bucket_path} exists")
        checkpoints = os.popen(ls_cmd).read().strip().split('\n')
        if len(checkpoints) == 0:
            raise ValueError(f"Checkpoint {bucket_path} does not exist")
        
        # download the checkpoint
        if not os.path.exists(f"{init_dir}/{checkpoint_name}"):
            rank_print(f"Downloading checkpoint {bucket_path} to {init_dir}")
            os.system(f"gsutil -q {auth_flag} cp {bucket_path} {init_dir}/{checkpoint_name}")
        else:
            rank_print(f"Checkpoint {init_dir}/{checkpoint_name} already exists")

        # load the checkpoint
        rank_print(f"Loading checkpoint {init_dir}/{checkpoint_name}")
        # model.load_state_dict(load_file(f"{init_dir}/{checkpoint_name}"))
        load_model(model, f"{init_dir}/{checkpoint_name}")
        rank_print("Checkpoint loaded successfully")
        with torch.device("cuda"):
            model.init_freqs_cis()

    if di.local_rank == 0:
        rank_print("After pruning model, model size is:")
        print_deepseek_model_sizes(model, di)

    print(f"pp_size: {di.pp_size}, real_pp_size: {di.real_pp_size}, pp_rank: {di.pp_rank}, group_size: {di.pp_mesh.get_group('pp').size()}")
    stage = PipelineStage(
        model,
        di.pp_rank,
        di.real_pp_size,
        torch.device(f"cuda:{di.local_rank}"),
        group=di.pp_mesh.get_group("pp")
    )

    def dummy_loss_fn(output, target):
        assert False, "This should never be called"


    # This defines how the inputs are chunked into microbatches
    args_chunk_spec = [TensorChunkSpec(0)]
    kwargs_chunk_spec = {
        "moe_loss": _Replicate,
        "start_pos": _Replicate,
        # "activated_layers": _Replicate,
    }

    # This defines how outputs from different microbatches are merged
    # We are going to gather all the logits from all the microbatches and sum the moe losses
    output_merge_spec = [TensorChunkSpec(0), sum_reducer]

    schedule = ScheduleGPipe(
        stage,
        n_microbatches=n_microbatches,
        loss_fn=dummy_loss_fn,
        args_chunk_spec=args_chunk_spec,
        kwargs_chunk_spec=kwargs_chunk_spec,
        output_merge_spec=output_merge_spec,
    )

    has_first_stage = di.pp_rank == 0
    has_last_stage = di.pp_rank == di.pp_size - 1

    def adjust_loss_fn(response_mask: torch.Tensor, gradient_accumulation: int, n_responses: int, loss_scaling_factor: float):
        global microbatch_start
        microbatch_start = 0
        _loss_fn = partial(loss_fn, response_mask=response_mask, gradient_accumulation=gradient_accumulation, n_responses=n_responses, loss_scaling_factor=loss_scaling_factor)
        schedule._loss_fn = _loss_fn

    def create_eval_schedule():
        stage = PipelineStage(
            model,
            di.pp_rank,
            di.pp_size,
            torch.device(f"cuda:{di.local_rank}"),
            group=di.pp_mesh.get_group("pp")
        )
        schedule = ScheduleGPipe(
            stage,
            n_microbatches=1,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
        )
        return schedule

    return schedule, model, has_first_stage, has_last_stage, adjust_loss_fn, create_eval_schedule


def get_model(model_name: str,
              di: DistributedInfo,
              use_random_init: bool = True,
              model_config_path: str = None,
              initialize_device: str = 'cuda',
              **kwargs) -> tuple[torch.nn.Module, dict]:
    """
    Get a model from a given name.

    Args:
        model_name: The name of the model to get. If it contains "deepseek", we will use the deepseek model.
        use_random_init: Whether to use random initialization.
        model_config_path: The path to the model config file.
        tp_mesh: The tensor parallel mesh.
        dp_mesh: The data parallel mesh.
        config: The config dictionary containing the whole invocation command line arguments.
        **kwargs: Additional keyword arguments.
            local_files_only: Whether to use local files only, only works for hf models
            tokenizer_vocab_size: The vocabulary size of the tokenizer, only works for hf models and random initted deepseek models
            print_model_sizes: Whether to print the model sizes
            batch_size: The batch size of the model, only works for deepseek models
            seq_len: The sequence length of the model, only works for deepseek models
            compile: Whether to compile the model using torch.compile

    Returns:
        model: The model.
        model_config: The model config.
    """
    is_deepseek = kwargs.get("deepseek", False)
    tokenizer_vocab_size = kwargs.get("tokenizer_vocab_size", None)
    local_files_only = kwargs.get("local_files_only", False)
    compile = kwargs.get("compile", False)
    seq_len = kwargs['seq_len']
    init_model_path = kwargs.get("init_model_path", None)
    consistent_init = kwargs.get("consistent_init", False)
    run_name = kwargs.get("run_name", None)

    if not is_deepseek:
        hf_token = os.environ.get("HF_AUTH_TOKEN", None)
        if not use_random_init:
            with timing("Loading model", rank_print):
                with torch.device("cuda"):
                    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False, local_files_only=local_files_only, token=hf_token)
            if model.config.vocab_size != tokenizer_vocab_size:
                raise ValueError(f"Model vocabulary size ({model.config.vocab_size}) does not match tokenizer vocabulary size ({tokenizer_vocab_size})")
            if seq_len > model.config.max_position_embeddings:
                raise ValueError(f"Sequence length ({seq_len}) exceeds model's maximum position embeddings ({model.config.max_position_embeddings})")
            model_config = model.config
        else:
            with timing("Loading model config", rank_print):
                model_config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only, token=hf_token)
                model_config.use_cache = False
                model_config.vocab_size = tokenizer_vocab_size
                model_config.max_position_embeddings = seq_len
            with timing("Loading model", rank_print):
                with torch.device("cuda"):
                    with torch.random.fork_rng():
                        torch.manual_seed(1337 + di.tp_rank + di.pp_rank)
                        model = AutoModelForCausalLM.from_config(model_config)
    else:
        if model_config_path is None:
            raise ValueError("model_config_path must be provided if --deepseek is True")
        if not os.path.exists(model_config_path):
            raise ValueError(f"deepseek_config file does not exist: {model_config_path}")
        with open(model_config_path, "r") as f:
            with timing("Loading model config", rank_print):
                model_config = json.load(f)
                # LLAMA vocab size is 128256, but their tokenizer is 128000, weird but this fixes it
                vocab_size = max(model_config.get("vocab_size", 0), tokenizer_vocab_size)
                print(f"adjusted vocab size: {vocab_size}")
            model_config = ModelArgs(**(model_config | {"train": True, "vocab_size": vocab_size}))
            if di.local_rank == 0:
                print(f"Model config: {model_config}")
                with torch.device("meta"):
                    with timing("Loading model", rank_print):
                        model = DeepSeekModel(model_config, di.tp_mesh)
                print_deepseek_model_sizes(model, di)
        
        if use_random_init:
            # if consistent_init is True, we need to make sure that the model is initialized the same way on all ranks, we do that by initializing a gold model and saving its state_dict
            if consistent_init:
                os.makedirs(f"/tmp/{run_name}", exist_ok=True)
                with torch.random.fork_rng():
                    torch.manual_seed(1337 + di.model_init_rank)
                    with torch.device("cuda"):
                        init_model_path = f"/tmp/{run_name}/consistent_init.pt"
                        init_model_done = f"/tmp/{run_name}/consistent_init_done"
                        if di.local_rank == 0:
                            gold_model = DeepSeekModel(model_config, None)
                            state_dict = gold_model.state_dict()
                            torch.save(state_dict, init_model_path)
                            rank_print(f"Consistent initialization saved to {init_model_path}")
                            os.system(f"touch {init_model_done}")
                            del gold_model
                            torch.cuda.empty_cache()
                        while not os.path.exists(init_model_done):
                            time.sleep(5)

            with torch.random.fork_rng():
                torch.manual_seed(1337 + di.model_init_rank)
                # only initialize the model on cuda if we are not using pipeline parallel
                # otherwise, initialize on meta to avoid OOM
                with torch.device(initialize_device):
                    with timing("Loading model", rank_print):
                        model = DeepSeekModel(model_config, di.tp_mesh)
                    if initialize_device != "meta":
                        model.init_freqs_cis()

            # load a full model from checkpoint, only useful for debugging
            if init_model_path is not None:
                init_state_dict = torch.load(init_model_path, map_location=f"cuda:{di.local_rank}")
                model_state_dict = {}
                for key in init_state_dict:
                    if "shared_experts" in key or "experts" not in key:
                        attr = key.split(".")[-2]
                        if attr not in TS_MAPPING or TS_MAPPING[attr] is None:
                            model_state_dict[key] = init_state_dict[key]
                        else:
                            dim = TS_MAPPING[attr]
                            shard_size = init_state_dict[key].size(dim) // di.tp_size
                            model_state_dict[key] = init_state_dict[key].narrow(dim, di.tp_rank * shard_size, shard_size)
                    else:
                        parts = key.split(".")
                        expert_num = int(parts[parts.index("experts") + 1])
                        num_experts_per_rank = model_config.n_routed_experts // di.tp_size
                        if expert_num < di.tp_rank * num_experts_per_rank or expert_num >= (di.tp_rank + 1) * num_experts_per_rank:
                            continue
                        model_state_dict[key] = init_state_dict[key]
                model.load_state_dict(model_state_dict)
                with torch.device("cuda"):
                    model.init_freqs_cis()
                rank_print(f"Loaded model from {init_model_path}")

        else:
            with torch.device("meta"):
                model = DeepSeekModel(model_config, di.tp_mesh)
    if compile:
        model = torch.compile(model)

    return model, model_config


def setup_fsdp(model: torch.nn.Module,
               di: DistributedInfo,
               ptype: torch.dtype,
               **kwargs):
    """ Sets up FSDP for the model.
    """
    is_deepseek = isinstance(model, DeepSeekModel)

    if is_deepseek:
        params = [p for p in model.parameters()]
        total_norm = torch.nn.utils.get_total_norm(params, 2.0)
        rank_print(f"Total param norm: {total_norm}")

    # Setup mixed precision
    mixed_precision_policy = MixedPrecision(
        param_dtype=ptype,
        reduce_dtype=torch.float32,
        buffer_dtype=ptype
    )

    # Auto wrapping policy to shard at transformer layer level
    wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer, GPT2Block, Qwen2DecoderLayer, Block},
    )

    
    # FSDP settings
    with timing("Setting up FSDP", rank_print):
        ignored_modules = [module for name, module in model.named_modules() if isinstance(module, (Gate, RMSNorm, ParallelEmbedding)) or name in ("head",)]

        if is_deepseek:
            pg = di.dp_mesh.get_group("dp")
            sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            if di.tp_size > 1 and di.dp_size > 1:
                pg = (di.tp_mesh.get_group(), di.dp_mesh.get_group())
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
                print(f"Using hybrid sharding with {di.tp_size} sharding dim and {di.dp_size} replicate dim")
            else:
                pg = di.dp_mesh.get_group("dp")
                sharding_strategy = ShardingStrategy.FULL_SHARD
                print(f"Using full sharding with {di.dp_size} replicate dim")

        ignored_modules = [module for name, module in model.named_modules() if isinstance(module, (Gate, RMSNorm, ParallelEmbedding)) or name in ("head",)]
        model = FSDP(
            model,
            process_group=pg,
            use_orig_params=True,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=sharding_strategy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=False),
            auto_wrap_policy=wrap_policy,
            ignored_modules=ignored_modules,
            device_id=di.local_rank,
        )
    model.gradient_checkpointing_enable()
    return model


def expert_balance(model: DeepSeekModel):
    """ Balances the expert load of the model.
    """
    n_layers = model.n_layers
    n_dense_layers = model.n_dense_layers
    if n_layers - n_dense_layers == 0:
        return torch.zeros(1, device='cuda')
    expert_imbalance = torch.zeros(n_layers - n_dense_layers, device='cuda')
    for i, layer in enumerate([block for block in model.layers if hasattr(block.ffn, 'gate')]):
        expert_imbalance[i] = layer.ffn.gate.expert_imbalance.std()
        layer.ffn.gate.balance_expert()
    return expert_imbalance

