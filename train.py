import time
import os
import logger
import click
import torch
import torch.distributed as dist
from contextlib import nullcontext
from prepare_data import get_tokenizer
from utils import rank_print, timing, init_distributed_groups, logger_init, TrainState, calculate_params
from models import get_model, setup_fsdp, compute_loss, expert_balance, setup_pp
from data import setup_data
from trainer import eval_and_log, checkpoint_and_upload, clip_grad_norm_, load_checkpoint
import numpy as np
import os
import torch.nn.functional as F
from verifier import verify_unroll, check_params, verify_activation

@click.command()
@click.option('--run-name', default=None, help='Name for this run. If None, an anonymous name will be generated')
@click.option('--disable-remote-logging', is_flag=True, default=False, help='Disable remote logging')
@click.option('--model-name', default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help='Model to load (or use as architecture for random init)')
@click.option('--use-random-init', is_flag=True, help='Use random initialization instead of loading weights')
@click.option('--batch-size', default=128, help='Batch size per dp rank')
@click.option('--gradient-accumulation', default=1, help='Number of steps to accumulate gradients')
@click.option('--seq-len', default=160, help='Sequence length for training')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--weight-decay', default=0.1, help='Weight decay')
@click.option('--warmup-iters', default=1000, help='Number of warmup iterations')
@click.option('--max-iters', default=600_000, help='Maximum number of training iterations')
@click.option('--log-every', default=10, help='Log every N iterations')
@click.option('--adam-beta1', default=0.9, help='Adam beta1 parameter')
@click.option('--adam-beta2', default=0.95, help='Adam beta2 parameter')
@click.option('--adam-eps', default=1e-8, help='Adam epsilon parameter')
@click.option('--local-files-only', is_flag=True, default=False, help='Use only local files, no downloads')
@click.option('--disable-eval', is_flag=True, default=False, help='Disable evaluation')
@click.option('--min-eval-interval', default=400, help='Minimum iterations between evaluations')
@click.option('--max-eval-interval', default=20_000, help='Maximum iterations between evaluations')
@click.option('--eval-spacing', default=1.3, help='Spacing factor between evaluations')
@click.option('--chess-eval-seq-len', default=80, help='Sequence length for chess-specific evaluation')
@click.option('--chess-eval-batch-size', default=20, help='Batch size for chess-specific evaluation')
@click.option('--checkpoint-every-hours', default=None, type=float, help='Save checkpoint every N hours')
@click.option('--simple-tokenization', is_flag=True, default=False, help='Use simple tokenization (no BPE)')
@click.option('--deepseek', is_flag=True, default=False, help='Use deepseek model')
@click.option('--tp-size', type=int, help='Size of the tensor parallel group, default to world size divided by dp_size * pp_size', default=None)
@click.option('--dp-size', type=int, help='Size of the data parallel group, default to world size divided by tp_size * pp_size', default=None)
@click.option('--pp-size', type=int, help='Size of the pipeline parallel group, default to world size divided by dp_size * tp_size', default=None)
@click.option('--starting-eval-at', type=int, help='Start evaluation at this iteration', default=0)
@click.option('--n-microbatches', type=int, help='Number of microbatches to use for pipeline parallel', default=1)
@click.option('--deepseek-config', type=str, help='Path to the deepseek config file', default=None)
@click.option('--compile', is_flag=True, default=False, help='Compile the model using torch.compile')
@click.option('--debug', is_flag=True, default=False, help='Debug mode, will calculate gradient norm and expect dl')
@click.option('--moe-loss-weight', type=float, help='Weight of the moe loss', default=1.0)
@click.option('--tokenizer', type=str, help='Tokenizer to use', default=None)
@click.option('--cpi', type=str, help='CPI to use, use comma to separate multiple CPIs', default='fen_move')
@click.option('--cpi-ratio', type=str, help='Ratio of different CPI to use, use comma to separate multiple ratios', default='1.0')
@click.option('--save-init-model', is_flag=True, default=False, help='Save the initialized model to disk, only for debugging purposes')
@click.option('--eval-cpi', type=str, help='CPI to use for evaluation, use comma to separate multiple CPIs', default='fen_english_move')
@click.option('--init-model-path', type=str, help='Load the initialized model from disk, only for debugging purposes', default=None)
@click.option('--sync-loss', is_flag=True, default=False, help='Sync the loss across all dp ranks, only for debugging purposes')
@click.option('--loss-eval-batches', type=int, help='Number of batches to evaluate on', default=10)
@click.option('--chess-eval-fraction', type=float, help='Fraction of training steps to do chess eval', default=0.0)
@click.option('--puzzle-eval-fraction', type=float, help='Fraction of training steps to do puzzle eval', default=0.0)
@click.option('--chat-eval-fraction', type=float, help='Fraction of training steps to do chat eval', default=0.0)
@click.option('--data-prep-seconds', type=int, help='Number of seconds to wait for data to be downloaded', default=600)
@click.option('--verify-unroll', is_flag=True, default=False, help='Verify the unroll of the model')
@click.option('--pp-rank', type=int, help='Pipeline rank to use, only for debugging purposes', default=None)
@click.option('--check-param', is_flag=True, default=False, help='Check the parameters of the model')
@click.option('--verify-activation', is_flag=True, default=False, help='Verify the activation of the model')
@click.option('--activated-layers', type=int, help='Number of layers to activate, only for verification purposes', default=1)
@click.option('--official-config-path', type=str, help='Path to the official model config', default=None)
@click.option('--first-checkpoint-iter', type=int, help='First checkpoint iteration', default=30)
@click.option('--consistent-init', is_flag=True, default=False, help='Use consistent initialization across ranks')
@click.option('--mixture-type', type=str, help='Mixture type to use, choose from within_batch, across_batch', default='within_batch')
@click.option('--hybrid-shard', is_flag=True, default=False, help='Use hybrid FSDP sharding')
@click.option('--resume-from', type=str, help='Resume from a checkpoint, only working for deepseek pp enabled models', default=None)
@click.option('--resume-from-iter', type=int, help='The iteration number to resume from', default=None)
@click.option('--measure-runtime-only', is_flag=True, default=False, help='Measure runtime only')
@click.option('--initialize-device', type=str, help='Device to initialize the model on, default is cuda, can be cpu or meta', default='cuda')
def train(**config):
    """Train a language model using FSDP."""

    # Initialize distributed training
    di = init_distributed_groups(**config)
    dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(1337 + di.real_dp_rank)
    
    # Load tokenizer first so we can get its vocab size
    with timing("Loading tokenizer", rank_print):
        tokenizer_name = config['tokenizer'] if config['tokenizer'] is not None else config['model_name']
        tokenizer = get_tokenizer(name=tokenizer_name, local_files_only=config['local_files_only'])
        tokenizer_vocab_size = len(tokenizer)
        rank_print(f"tokenizer.vocab_size: {tokenizer.vocab_size}", 0)


    # Setup data
    rank_print("Setting up data")
    checkpoints_dir, datasets_dir, unrolls_dir, get_batch, data_files = setup_data(di, config)


    rank_print("Setting up model")
    model, model_config = get_model(
        config['model_name'],
        di,
        config['use_random_init'],
        config['deepseek_config'],
        tokenizer_vocab_size=tokenizer_vocab_size,
        batch_size=config['batch_size'],
        seq_len=config['seq_len'],
        deepseek=config['deepseek'],
        init_model_path=config['init_model_path'],
        consistent_init=config['consistent_init'],
        run_name=config['run_name'],
        initialize_device=config['initialize_device']
    )
    # training dtype
    ptype = torch.bfloat16

    # save initialized model to disk
    if config['save_init_model']:
        checkpoint_path = f"{config['run_name']}_init.pth"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        rank_print(f"Initialized model saved to {checkpoint_path}")

    # Calculate effective batch size
    per_dp_rank_batch_size = config['batch_size']
    gradient_accumulation = config['gradient_accumulation']
    global_batch_size = per_dp_rank_batch_size * di.dp_size * gradient_accumulation

    # Setup PP
    is_pp = di.pp_size > 1
    if is_pp:
        rank_print("Setting up PP")
        pp_schedule, model, has_first_stage, has_last_stage, adjust_loss_fn, create_eval_schedule = setup_pp(model, di, config['n_microbatches'], config['use_random_init'], config['init_model_path'] is not None or config['consistent_init'], config['resume_from'], config['initialize_device'])
    else:
        has_first_stage = True
        has_last_stage = True
        pp_schedule = None
        create_eval_schedule = lambda: None

    if config['check_param']:
        check_params(model, di)
        torch.distributed.destroy_process_group()
        import sys; sys.exit()

    if config['verify_activation']:
        verify_activation(model, pp_schedule, tokenizer, "Once upon a time,", di, config['activated_layers'], official_config_path=config['official_config_path'])
        torch.distributed.destroy_process_group()
        import sys; sys.exit()

    # Setup FSDP
    # TODO: deepseek model doesn't work with FSDP yet
    if di.dp_size > 1:
        rank_print("Setting up FSDP")
        model = setup_fsdp(model, di, torch.float32)
    else:
        model.gradient_checkpointing_enable()

    # Finally put the model on the correct device
    model.to_empty(device="cuda")
    model.init_weights()
    with torch.device("cuda"):
        model.init_freqs_cis()

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        eps=config['adam_eps'],
        weight_decay=config['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-4,
        end_factor=1.0,
        total_iters=config['warmup_iters']
    )

    # Log configuration details
    should_log_to_server = (di.pp_size == 1 and di.world_rank == 0) or (has_last_stage and di.dp_rank == 0 and di.tp_rank == 0)
    num_params, param_size = calculate_params(model)
    logger_init(config, model_config.__dict__, di, global_batch_size=global_batch_size, should_log_to_server=should_log_to_server, num_params=num_params, param_size=param_size)

    # Identify the fixed scaling factor for the loss, equal to n_responses / unmasked_token
    X, Y, response_mask, n_responses = get_batch(0, with_response_mask=True)

    total_tokens = torch.tensor(sum([len(np.memmap(os.path.expanduser(f'{train_file}'), dtype=np.uint32, mode='r')) for train_file, _ in data_files]), device=di.local_rank) if X is not None else torch.zeros(1, device=di.local_rank)
    dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM, group=di.dp_mesh.get_group())
    mask_nonzero = (response_mask != 0).sum() if response_mask is not None else torch.zeros(1, device=di.local_rank)
    _n_responses = n_responses.clone().detach() if n_responses is not None else torch.zeros(1, device=di.local_rank)
    dist.all_reduce(mask_nonzero, op=dist.ReduceOp.SUM, group=di.dp_mesh.get_group())
    dist.all_reduce(_n_responses, op=dist.ReduceOp.SUM, group=di.dp_mesh.get_group())
    loss_scaling_factor = _n_responses.item() / mask_nonzero.item()
    if di.world_rank == 0:
        print(f"Total_tokens: {total_tokens.item():_}")
        print(f"Example_datapoint: \033[96m\n\t{tokenizer.decode(X[0])}\033[0m")
        print(f"Example_label: \033[96m\n\t{tokenizer.decode(Y[0][response_mask[0] != 0])}\033[0m")
        print(f"Scaling factor: {loss_scaling_factor} ({_n_responses.item()} / {mask_nonzero.item()})")

    if config['verify_unroll']:
        assert config['deepseek'], "Verification of logits is only supported for deepseek models"
        model.eval()
        verify_unroll(model, create_eval_schedule, tokenizer, "Once upon a time,", di)

    # Training loop
    dist.barrier()
    print(f"Rank {di.world_rank} beginning to train")
    model.train()
    ts = TrainState(
        run_name=config['run_name'],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        schedule=pp_schedule if is_pp else None,
        has_first_stage=has_first_stage,
        has_last_stage=has_last_stage,
        tokenizer=tokenizer,
        di=di,
        device=torch.device(f"cuda:{di.local_rank}"),
        config=config,
        get_batch=get_batch,
        compute_loss=compute_loss,
        start_time=time.time(),
        iter_timer=time.time(),
        iter_num=0,
        next_eval_at=config.get('starting_eval_at', 0),
        last_checkpoint_time=time.time(),
        interval_between_evals=config['min_eval_interval'],
        upload_thread=None,
        unrolls_dir=unrolls_dir,
        checkpoint_dir=checkpoints_dir,
        datasets_dir=datasets_dir,
        ptype=ptype,
        loss_scaling_factor=loss_scaling_factor,
        chess_eval_fraction=config['chess_eval_fraction'],
        puzzle_eval_fraction=config['puzzle_eval_fraction'],
        chat_eval_fraction=config['chat_eval_fraction'],
        loss_eval_batches=config['loss_eval_batches']
    )

    if config['resume_from'] is not None:
        load_checkpoint(model, optimizer, config['resume_from'], ts, config)
        dist.barrier()

    # torch.autograd.set_detect_anomaly(True)
    measure_start_time = None
    while True:
        # Evaluation & checkpointing
        if not config['disable_eval'] and (ts.iter_num >= ts.next_eval_at or ts.iter_num == config['max_iters']):
            eval_and_log(ts, create_eval_schedule)

        # Checkpointing
        if config['checkpoint_every_hours'] is not None and ((time.time() - ts.last_checkpoint_time) / 3600 >= config['checkpoint_every_hours'] or ts.iter_num == config['first_checkpoint_iter']):
            checkpoint_and_upload(ts)

        # Compute gradient
        loss_acc = torch.zeros(1, device=ts.device, dtype=torch.float32)
        response_loss_acc = torch.zeros(1, device=ts.device, dtype=torch.float32)
        moe_loss_acc = torch.zeros(1, device=ts.device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        for i in range(gradient_accumulation):
            ctx = model.no_sync() if i < gradient_accumulation - 1 else nullcontext()
            with ctx:
                if not is_pp:
                    with torch.autocast(device_type="cuda", dtype=ptype):
                        losses, moe_loss = compute_loss(model, X, Y)
                        response_losses = ts.loss_scaling_factor * losses / (gradient_accumulation * n_responses)
                        moe_losses = ts.loss_scaling_factor * config['moe_loss_weight'] * moe_loss / (gradient_accumulation * n_responses)
                        response_loss = torch.where(response_mask != 0, response_losses, 0.0).sum()
                        moe_loss = moe_losses.sum()
                        loss = response_loss + moe_loss
                        loss_acc += loss.detach().float()
                        response_loss_acc += response_loss.detach().float()
                        moe_loss_acc += moe_loss.detach().float()
                    loss.backward()
                else:
                    # This is a hack to work around the fact the torch's PP solution doesn't allow easy
                    # manipulation of the final loss and the fact that our loss is adjusted by n_responses,
                    # a value only known once we get the batch.
                    # We also only adjust the loss on the last stage
                    if response_mask is not None:
                        adjust_loss_fn(response_mask, gradient_accumulation, n_responses, ts.loss_scaling_factor)

                    targets, losses = (Y, []) if has_last_stage else (None, None)
                    if has_first_stage:
                        pp_schedule.step(X, target=targets, losses=losses, moe_loss=torch.tensor(0.0, device=ts.device), start_pos=0)
                    elif has_last_stage:
                        # loss is already calculated and backpropped in the pipeline
                        # here we just re-calculate it to report it
                        # TODO: this requires that n_microbatches is 1
                        logits, moe_loss = pp_schedule.step(target=targets, losses=losses)
                        logits = torch.log_softmax(logits, dim=-1).float()
                        response_losses = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), reduction='none').view(Y.shape)
                        response_losses = ts.loss_scaling_factor * response_losses / (gradient_accumulation * n_responses)
                        moe_losses = ts.loss_scaling_factor * config['moe_loss_weight'] * moe_loss / (gradient_accumulation * n_responses)
                        response_loss = torch.where(response_mask != 0, response_losses, 0.0).sum()
                        moe_loss = moe_losses.sum()
                        response_loss_acc += response_loss.detach().float()
                        moe_loss_acc += moe_loss.detach().float()
                    else:
                        pp_schedule.step(target=targets, losses=losses)

                    loss = (
                        torch.mean(torch.stack(losses)).to(ts.device)
                        if has_last_stage
                        else torch.tensor([-1.0], device=ts.device)
                    )
                    loss_acc += loss.detach().float()

            X, Y, response_mask, n_responses = get_batch(ts.iter_num, with_response_mask=True)

        if config['sync_loss'] and di.dp_size > 1:
            dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM) 
            loss_acc = loss_acc / di.dp_size

        # Clip gradients
        norm = clip_grad_norm_(
            model,
            max_norm=1.0,
            groups=[di.tp_mesh.get_group(), di.pp_mesh.get_group()]
        )

        if config['debug']:
            norm = norm.sum()
        
        # Compute inner product between parameters and gradients
        if config['debug']:
            expect_dl = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    expect_dl += (param.grad * param).sum()

        # Optimizer step
        optimizer.step()
        scheduler.step()

        # Expert load balance
        if config['deepseek']:
            expert_imbalance = expert_balance(model)

        # Iter complete
        ts.iter_num += 1
        if config['measure_runtime_only']:
            if ts.iter_num == 10:
                measure_start_time = time.time()
            if ts.iter_num == 60:
                average_time = (time.time() - measure_start_time) / 50
                print(f"Rank {di.local_rank} finished training, time: {average_time}")
                break

        # Logging, we print logs on the first dp rank and on the first tp rank if we are using PP
        if ts.iter_num % config['log_every'] == 0 and (ts.di.local_rank == 0 or should_log_to_server):
            duration = time.time() - ts.iter_timer; ts.iter_timer = time.time()
            log = {
                "total_hours": (time.time() - ts.start_time) / 3600,
                "iter": ts.iter_num,
                "loss": loss_acc.item(),
                "response_loss": response_loss_acc.item(),
                "moe_loss": moe_loss_acc.item(),
                "duration": duration / config['log_every']
            }
            if config['debug']:
                log["grad_norm"] = norm.item()
                log["expect_dl"] = expect_dl.item()
            if config['deepseek']:
                log["expert_imbalance_mean"] = expert_imbalance.mean().item()
                log["expert_imbalance_max"] = expert_imbalance.max().item()
                log["expert_imbalance_max_layer"] = expert_imbalance.argmax().item() + model.n_dense_layers
            logger.log("train", log)

        if ts.iter_num >= config['max_iters']:
            break

    # Wait for all logs to be written
    if should_log_to_server:    
        rank_print("waiting for logs to be written")
        logger.wait_for_completion()

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    train()


