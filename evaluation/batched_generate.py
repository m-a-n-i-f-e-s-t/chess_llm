import torch
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm

def batched_generate(
    model: torch.nn.Module,
    tokenizer,
    contexts: Optional[List[str]] = None,
    tokenized_contexts: Optional[List[List[int]]] = None,
    max_new_tokens: int = 100,
    batch_size: Optional[int] = None,
    temperature: float = 1.0,
    greedy: bool = False,
    device: str = 'cuda',
    token_output: bool = True,
) -> List[str]:
    """
    Generate continuations for multiple contexts in batches using model.generate with KV caching.
    """
    # Tokenize contexts and determine maximum context length
    assert (tokenized_contexts is None) != (contexts is None), "Must provide either contexts or tokenized_contexts"
    if tokenized_contexts is None:
        tokenized_contexts = [tokenizer.encode(ctx) for ctx in contexts]
    if batch_size is None:
        batch_size = len(tokenized_contexts)

    model.eval()
    all_outputs = []

    # Get rank for distributed training
    rank = getattr(model, 'rank', 0)

    # Process contexts in batches
    with FSDP.summon_full_params(model, writeback=False):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Only show progress bar for rank 0
            batch_range = range(0, len(tokenized_contexts), batch_size)
            if rank == 0:
                batch_range = tqdm(batch_range, desc="Generating")

            for batch_start in batch_range:
                batch_tokens = tokenized_contexts[batch_start:batch_start+batch_size]
                max_ctx_len = max(len(t) for t in batch_tokens)
                # Create padded input tensor and attention mask
                input_ids = torch.zeros((len(batch_tokens), max_ctx_len), dtype=torch.long, device=device)
                attention_mask = torch.zeros_like(input_ids, device=device)
                for i, tokens in enumerate(batch_tokens):
                    input_ids[i, -len(tokens):] = torch.tensor(tokens, dtype=torch.long, device=device)
                    attention_mask[i, -len(tokens):] = 1

                # Use huggingface generate
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=not greedy,
                    temperature=temperature,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )

                # Remove padding tokens
                outputs = [output[max_ctx_len - len(tokens):] for output, tokens in zip(outputs, batch_tokens)]

                # Decode each generated sequence
                for output in outputs:
                    if not token_output:
                        text = tokenizer.decode(output)
                    else:
                        text = output
                    all_outputs.append(text)

    return all_outputs
