from eval_data import data as eval_data
from batched_generate import batched_generate
import torch
import torch.distributed as dist

def unroll_on_eval(data, model, tokenizer, cpi, greedy=True, seq_len=128, batch_size=16, device='cuda'):
    """Unroll the model on the evaluation set.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        cpi: The Chess Prompt Interface to use
        greedy: Whether to use greedy decoding (True) or sampling (False)
        seq_len: Maximum number of new tokens to generate
        batch_size: Number of examples to process in parallel
        device: Device to run evaluation on
        
    Returns:
        List of generated continuations
    """

    # Format each context using the CPI
    tokenized_contexts = []
    for item in data:
        # Create a dictionary with the required fields for the CPI
        example = {"fen": item[0]}
            
        # Get tokens using the CPI (question only)
        tokens = cpi.example_to_tokens(example, tokenizer, question_only=True)
        tokenized_contexts.append(tokens)
    
    responses = batched_generate(
        model=model,
        tokenizer=tokenizer,
        tokenized_contexts=tokenized_contexts,
        batch_size=batch_size,
        max_new_tokens=seq_len-max([len(x) for x in tokenized_contexts]),
        greedy=greedy,
        device=device,
        token_output=True,
    )
    return responses



def score_rollouts_on_eval(model, tokenizer, cpi, rank, world_size, greedy=True, seq_len=128, batch_size=16, device='cuda'):
    my_data = eval_data[rank::world_size]

    # Get unrolls for this device's portion of data
    responses = unroll_on_eval(my_data, model, tokenizer, cpi, greedy=greedy, seq_len=seq_len, batch_size=batch_size, device=device)
    prompts = [my_data[i][0] for i in range(len(my_data))]
    guesses = [cpi.extract_answer(x, tokenizer) for x in responses]
    unrolls = [cpi.decode_strip_ending(x, tokenizer) for x in responses]

    # Get corresponding ground truth responses
    best_moves = [item[1] for item in my_data]
    legal_moves = [item[2] for item in my_data]
    
    # Calculate metrics
    exact_matches = sum(1 for g, m in zip(guesses, best_moves) if g == m)
    legal_guesses = sum(1 for g, ms in zip(guesses, legal_moves) if g in ms)
    total = len(guesses)
    
    to_sync = torch.tensor([exact_matches, legal_guesses, total], device=device)
    dist.all_reduce(to_sync, op=dist.ReduceOp.SUM)
    exact_matches, legal_guesses, total = to_sync.tolist()

    # Calculate final metrics
    best_accuracy = exact_matches / total if total > 0 else 0.0
    legal_accuracy = legal_guesses / total if total > 0 else 0.0
    
    return prompts, unrolls, {
        'best_move_accuracy_eval': best_accuracy,
        'legal_move_accuracy_eval': legal_accuracy
    }
