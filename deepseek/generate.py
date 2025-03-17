import os
import json
import time
from argparse import ArgumentParser
from typing import List, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torch.distributed.pipelining.microbatch import TensorChunkSpec, _Replicate, sum_reducer
from torch.distributed.pipelining.schedules import PipelineScheduleSingle
from transformers import AutoTokenizer
import torch.multiprocessing as mp
from safetensors.torch import load_model
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from .model import Transformer, ModelArgs
from tqdm import tqdm


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    model.eval()
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    # TODO: something is wrong with the kv cache, fix it later
    for cur_pos in range(min(prompt_lens), total_len):
        logits, _ = model.forward(tokens[:, 0:cur_pos], moe_loss=torch.tensor(0.0, device='cuda'), start_pos=0)
        logits = logits[:, -1, :]
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


@torch.inference_mode()
def pp_generate(
    schedule: PipelineScheduleSingle,
    decoding_schedule: PipelineScheduleSingle,
    create_eval_schedule: Callable,
    has_first_stage: bool,
    has_last_stage: bool,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    global_rank: int,
    temperature: float = 1.0,
    use_kv_cache: bool = False
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        schedule (PipelineScheduleSingle): The pipeline schedule used for token generation.
        has_first_stage (bool): Whether the first stage is present.
        has_last_stage (bool): Whether the last stage is present.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        global_rank (int): The global rank of the current process.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    def _global_rank(peer_rank, group):
        return (
            peer_rank
            if group is None
            else dist.get_global_rank(group, peer_rank)
        )
    last_stage_peer_rank = schedule._stage.group_size - 1
    last_stage_global_rank = _global_rank(last_stage_peer_rank, schedule._stage.group)
    first_stage_peer_rank = 0
    first_stage_global_rank = _global_rank(first_stage_peer_rank, schedule._stage.group)
    model = schedule._stage.submod
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    first = True
    # TODO: something is wrong with the kv cache, fix it later
    for cur_pos in range(min(prompt_lens), total_len):
        # TODO: enable this switch when kv cache is fixed
        if not use_kv_cache:
            schdule_to_use = create_eval_schedule()
        else:
            schdule_to_use = schedule if first else decoding_schedule
        if has_first_stage:
            if use_kv_cache:
                output = schdule_to_use.step(tokens[:, prev_pos:cur_pos], moe_loss=torch.tensor(0.0, device=tokens.device), start_pos=prev_pos)
            else:
                output = schdule_to_use.step(tokens[:, 0:cur_pos], moe_loss=torch.tensor(0.0, device=tokens.device), start_pos=0)
        else:
            output = schdule_to_use.step()
        first = False
        if output is not None:  # last stage
            logits, _ = output
            logits = logits[:, -1, :]
            if temperature > 0:
                next_token = sample(logits, temperature)
            else:
                next_token = logits.argmax(dim=-1)
            next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
            prev_pos = cur_pos
            # need to send tokens to first stage
            assert has_last_stage, "output should only be present in last stage"
            dist.send_object_list(
                [next_token, prev_pos],
                dst=first_stage_global_rank,
                group=schdule_to_use._stage.group,
                device=schdule_to_use._stage.device
            )
            # broadcast finished to all ranks
            assert global_rank == last_stage_global_rank, "this should be the last stage"
            dist.broadcast_object_list(
                [finished],
                src=global_rank,
                group=schdule_to_use._stage.group,
                device=schdule_to_use._stage.device
            )
        else:  # not last stage
            # receive tokens from last stage
            if has_first_stage:
                objects = [None, None]
                dist.recv_object_list(
                    objects,
                    src=last_stage_global_rank,
                    group=schdule_to_use._stage.group,
                    device=schdule_to_use._stage.device
                )
                next_token, recv_prev_pos = objects
                tokens[:, cur_pos] = next_token
                prev_pos = cur_pos
                assert prev_pos == recv_prev_pos, "prev_pos should be the same"
            dist.broadcast_object_list(
                [finished],
                src=last_stage_global_rank,
                group=schdule_to_use._stage.group,
                device=schdule_to_use._stage.device
            )
        if finished.all():
            break

    completion_tokens = []
    if has_last_stage:
        for i, toks in enumerate(tokens.tolist()):
            toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
            if eos_id in toks:
                toks = toks[:toks.index(eos_id)]
            completion_tokens.append(toks)
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    checkpoint_url: str = "http://deepseek-svc:8000/"
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tensor_path = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    if not os.path.exists(tensor_path):
        os.makedirs(ckpt_path, exist_ok=True)
        print(f"Downloading model {rank}-mp{world_size} from {checkpoint_url}")
        files_needed = [
            f"model{rank}-mp{world_size}.safetensors",
        ]
        if local_rank == 0: # only need to download files on the first local rank
            files_needed.extend([
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "configuration_deepseek.py",
                "generation_config.json",
                "model.safetensors.index.json",
            ])
        for file in files_needed:
            start = time.time()
            if "gs://" in checkpoint_url:
                os.system(f"gsutil -m cp {checkpoint_url}/mp{world_size}/{file} {ckpt_path}/{file}")
            elif "http://" in checkpoint_url:
                os.system(f"wget -q {checkpoint_url}/mp{world_size}/{file} -O {ckpt_path}/{file}")
            else:
                raise ValueError(f"Invalid checkpoint URL: {checkpoint_url}")
            end = time.time()
            print(f"Time taken to download {file}: {end - start} seconds")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
    load_model(model, tensor_path)

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        prompts = prompts[:args.max_batch_size]
        prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
        total_tokens = sum([len(t) for t in prompt_tokens])
        start = time.time()
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        end = time.time()
        total_output_tokens = sum([len(c) for c in completions])
        print(f"Time taken: {end - start} seconds")
        print(f"Total input tokens: {total_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        print(f"Output tokens per second: {total_output_tokens / (end - start)}")
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()


def run(rank: int):
    dist.init_process_group("nccl", init_method="tcp://localhost:12346", world_size=8, rank=rank)
    mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("pp", "tp"))
    pp_mesh = mesh["pp"]
    tp_mesh = mesh["tp"]
    pp_size = pp_mesh.size()
    tp_size = tp_mesh.size()
    pp_rank = (rank // 2) % 4
    tp_rank = rank % 2
    vocab = 4096
    torch.cuda.set_device(rank)
    with torch.device(f"cuda:{rank}"):
        args = ModelArgs(
            vocab_size=vocab,
            dim=1024,
            inter_dim=1024,
            moe_inter_dim=1024,
            n_layers=12,
            n_dense_layers=1,
            n_routed_experts=8,
            n_shared_experts=2,
            n_activated_experts=2,
            max_batch_size=16,
        )
        model = Transformer(args, tp_mesh)
        model.eval()
        model.init_freqs_cis()

    # Basic sanity check
    assert pp_size <= args.n_layers, "too many pipeline stages for too few layers"

    # Compute how many layers each rank should handle
    base = args.n_layers // pp_size
    remainder = args.n_layers % pp_size

    # List of chunk sizes—some ranks get an extra layer until the remainder is exhausted
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(pp_size)]

    # Compute starting and ending indices for this rank
    start_layer_idx = sum(chunk_sizes[:pp_rank])
    end_layer_idx = start_layer_idx + chunk_sizes[pp_rank]

    model.layers = model.layers[start_layer_idx:end_layer_idx]
    if pp_rank == 0:
        del model.head
        del model.norm
        model.head = None
        model.norm = None
    elif pp_rank < pp_size - 1:
        del model.embed
        del model.norm
        del model.head
        model.embed = None
        model.norm = None
        model.head = None
    else:
        del model.embed
        model.embed = None
    torch.cuda.empty_cache()

    def create_eval_schedule():
        stage = PipelineStage(
            model,
            pp_rank,
            pp_size,
            torch.device(f"cuda:{rank}"),
            group=pp_mesh.get_group("pp")
        )
        schedule = ScheduleGPipe(
            stage,
            n_microbatches=1,
            args_chunk_spec=[TensorChunkSpec(0)],
            kwargs_chunk_spec={
                "moe_loss": _Replicate,
                "start_pos": _Replicate,
            },
            output_merge_spec=[TensorChunkSpec(0), sum_reducer, _Replicate]
        )
        return schedule
    
    eval_schedule = create_eval_schedule()
    decode_schedule = create_eval_schedule()
    has_first_stage = pp_rank == 0
    has_last_stage = pp_rank == pp_size - 1

    prompt_tokens = torch.randint(0, vocab, (16, 160,)).tolist()
    outputs1 = pp_generate(
        schedule=eval_schedule,
        decoding_schedule=decode_schedule,
        create_eval_schedule=create_eval_schedule,
        has_first_stage=has_first_stage,
        has_last_stage=has_last_stage,
        prompt_tokens=prompt_tokens,
        max_new_tokens=128,
        eos_id=0,
        global_rank=rank,
        temperature=0.0,
        use_kv_cache=True
    )
    start = time.time()
    outputs1 = pp_generate(
        schedule=eval_schedule,
        decoding_schedule=decode_schedule,
        create_eval_schedule=create_eval_schedule,
        has_first_stage=has_first_stage,
        has_last_stage=has_last_stage,
        prompt_tokens=prompt_tokens,
        max_new_tokens=1024,
        eos_id=0,
        global_rank=rank,
        temperature=0.0,
        use_kv_cache=True
    )
    kv_cache_time = time.time() - start
    start = time.time()
    outputs2 = pp_generate(
        schedule=eval_schedule,
        decoding_schedule=decode_schedule,
        create_eval_schedule=create_eval_schedule,
        has_first_stage=has_first_stage,
        has_last_stage=has_last_stage,
        prompt_tokens=prompt_tokens,
        max_new_tokens=1024,
        eos_id=0,
        global_rank=rank,
        temperature=0.0,
        use_kv_cache=False
    )
    no_kv_cache_time = time.time() - start

    if pp_rank == pp_size - 1 and tp_rank == 0:
        print("="*10 + "with kv cache " + "="*10)
        print(outputs1[0])
        print(f"Time taken: {kv_cache_time=} seconds")
        print("="*10 + "without kv cache " + "="*10)
        print(outputs2[0])
        print(f"Time taken: {no_kv_cache_time=} seconds")
        # check if outputs1 and outputs2 are the same, element by element
        diff = 0
        for o1, o2 in zip(outputs1, outputs2):
            diff += sum(o1 != o2)
        print(f"Diff: {diff}/{len(outputs1[0])}")

    end = time.time()
    total_output_tokens = sum([len(c) for c in outputs1])
    if pp_rank == pp_size - 1:
        print(f"Time taken: {end - start} seconds")
        print(f"Output tokens per second: {total_output_tokens / (end - start)}")



    dist.destroy_process_group()


if __name__ == "__main__":    
    mp.spawn(run, args=(), nprocs=8)