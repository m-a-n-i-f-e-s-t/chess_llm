import torch
from typing import List, Optional, Callable
from torch.distributed.pipelining.schedules import _PipelineSchedule
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from deepseek.model import Transformer as DeepseekModel
from deepseek.generate import pp_generate, generate
from contextlib import nullcontext
from tqdm import tqdm
import os
import time
from utils import local_print, rank_print

def batched_generate(
    model: torch.nn.Module,
    tokenizer,
    schedule: Optional[_PipelineSchedule] = None,
    create_eval_schedule: Optional[Callable[[], _PipelineSchedule]] = None,
    has_first_stage: bool = False,
    has_last_stage: bool = False,
    contexts: Optional[List[str]] = None,
    tokenized_contexts: Optional[List[List[int]]] = None,
    max_new_tokens: int = 100,
    batch_size: Optional[int] = None,
    temperature: float = 1.0,
    greedy: bool = False,
    device: str = 'cuda',
    token_output: bool = False,
    only_completion: bool = True,
    progress_bar: str = "Generating"
) -> List[str]:
    """
    Generate continuations for multiple contexts in batches using model.generate with KV caching.
    """
    decoding_schedule = create_eval_schedule()
    # Tokenize contexts and determine maximum context length
    assert (tokenized_contexts is None) != (contexts is None), "Must provide either contexts or tokenized_contexts"
    if tokenized_contexts is None:
        tokenized_contexts = [tokenizer.encode(ctx) for ctx in contexts]
    if batch_size is None:
        batch_size = len(tokenized_contexts)

    model.eval()
    all_outputs = []
    start = time.time()
    total_tokens = 0
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_rank = int(os.environ.get('RANK', 0))

    # Get rank for distributed training
    rank = getattr(model, 'rank', 0) if schedule is None else int(os.environ.get('LOCAL_RANK', 0))

    # Process contexts in batches
    ctx = FSDP.summon_full_params(model, writeback=False) if isinstance(model, FSDP) else nullcontext()
    with ctx:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Only show progress bar for rank 0
            batch_range = range(0, len(tokenized_contexts), batch_size)
            if rank == 0:
                batch_range = tqdm(batch_range, desc=progress_bar)

            for batch_start in batch_range:
                batch_tokens = tokenized_contexts[batch_start:batch_start+batch_size]
                # Use huggingface generate; caching is handled internally with use_cache=True
                if isinstance(model, DeepseekModel) or (isinstance(model, FSDP) and isinstance(model.module, DeepseekModel)):
                    if schedule is None:
                        outputs = generate(
                            model,
                            prompt_tokens=batch_tokens,
                            max_new_tokens=max_new_tokens,
                            eos_id=tokenizer.eos_token_id,
                            temperature=temperature if not greedy else 0.0,
                        )
                    else:
                        outputs = pp_generate(
                            create_eval_schedule(), # each batch might have different generation length, hence need to create new schedule
                            decoding_schedule=decoding_schedule,
                            create_eval_schedule=create_eval_schedule,
                            has_first_stage=has_first_stage,
                            has_last_stage=has_last_stage,
                            prompt_tokens=batch_tokens,
                            max_new_tokens=max_new_tokens,
                            eos_id=tokenizer.eos_token_id,
                            global_rank=dist.get_rank(),
                            temperature=temperature if not greedy else 0.0
                        )
                    if not only_completion:
                        outputs = [[*inp, *out] for out, inp in zip(outputs, batch_tokens)] if outputs is not None else []
                else:
                    max_ctx_len = max(len(t) for t in batch_tokens)
                    input_ids = torch.zeros((len(batch_tokens), max_ctx_len), dtype=torch.long, device=device)
                    attention_mask = torch.zeros_like(input_ids, device=device)
                    for i, tokens in enumerate(batch_tokens):
                        input_ids[i, -len(tokens):] = torch.tensor(tokens, dtype=torch.long, device=device)
                        attention_mask[i, -len(tokens):] = 1
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=not greedy,
                        temperature=temperature,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    # remove padding
                    outputs = [output[max_ctx_len - len(tokens):] for output, tokens in zip(outputs, batch_tokens)] if outputs is not None else []

                total_tokens += sum([len(output) for output in outputs])

                # Decode each generated sequence
                for i, output in enumerate(outputs):
                    if not token_output:
                        text = tokenizer.decode(output)
                    else:
                        text = output
                    all_outputs.append(text)

        end = time.time()

        if all_outputs:
            local_print(f"[{world_rank}] Time taken: {end - start} seconds")
            local_print(f"[{world_rank}] Output tokens per second: {total_tokens / (end - start)}")


    return all_outputs

if __name__ == "__main__":
    model_name = "deepseek-ai/deepseek-coder-6.7b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    contexts = [
        "def fibonacci(n):",
        "# Function to calculate factorial\ndef factorial(n):",
        "class BinaryTree:",
    ]

    continuations = batched_generate(
        model,
        tokenizer,
        contexts,
        max_new_tokens=100,
        temperature=0.8
    )

    print("\nGenerated continuations:")
    print("-" * 40)
    for ctx, cont in zip(contexts, continuations):
        print(f"Context: {ctx}")
        print(f"Generated: {cont}")
        print("-" * 40)
