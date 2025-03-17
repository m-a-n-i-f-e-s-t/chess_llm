from eval_data import data as eval_data
from batched_generate import batched_generate
from prepare_data import get_cpi, get_tokenizer
from utils import local_print, rank_print
import json
import torch


def unroll_on_eval(model, create_eval_schedule, tokenizer, cpi_name, greedy=True, fraction=1.0, seq_len=100, dp_rank=0, dp_size=1, has_first_stage=False, has_last_stage=False, batch_size=8, device='cuda'):
    """Unroll the model on the evaluation set.
    
    Args:
        model: The model to evaluate
        create_eval_schedule: A callable that creates new schedule
        tokenizer: The tokenizer to use
        cpi_name: Name of the Chess Prompt Interface to use
        greedy: Whether to use greedy decoding (True) or sampling (False)
        fraction: Fraction of evaluation data to use
        seq_len: Maximum number of new tokens to generate
        dp_rank: Current process rank for distributed evaluation
        dp_size: Total number of processes for distributed evaluation
        has_first_stage: Whether the model has a first stage
        has_last_stage: Whether the model has a last stage
        batch_size: Number of examples to process in parallel
        device: Device to run evaluation on
        
    Returns:
        List of generated continuations
    """
    unroll_on_eval_schedule = create_eval_schedule()
    # Initialize cpi
    cpi = get_cpi(cpi_name)
    fraction_len = max(int(len(eval_data) * fraction), batch_size) if fraction > 0 else 0
    eval_data_fraction = eval_data[:fraction_len]
    
    # Get my portion of the eval data based on world rank/size
    my_start = (len(eval_data_fraction) // dp_size) * dp_rank
    my_end = (len(eval_data_fraction) // dp_size) * (dp_rank + 1)
    my_eval_data = eval_data_fraction[my_start:my_end]
    rank_print(f"Processing {len(my_eval_data)} eval examples from {my_start} to {my_end} of {fraction_len} examples, fraction {fraction}")

    # Format each context using the CPI
    tokenized_contexts = []
    for item in my_eval_data:
        # Create a dictionary with the required fields for the CPI
        example = {"fen": item[0]}
            
        # Get tokens using the CPI (question only)
        tokens = cpi.example_to_tokens(example, tokenizer, question_only=True)
        tokenized_contexts.append(tokens)
    
    out_tokens = batched_generate(
        model=model,
        tokenizer=tokenizer,
        schedule=unroll_on_eval_schedule,
        create_eval_schedule=create_eval_schedule,
        has_first_stage=has_first_stage,
        has_last_stage=has_last_stage,
        tokenized_contexts=tokenized_contexts,
        batch_size=batch_size,
        max_new_tokens=seq_len,
        greedy=greedy,
        device=device,
        token_output=True,
        only_completion=False,
        progress_bar=f"Unrolling on eval"
    )
    return out_tokens

def keep_until_period(text):
    """Keep text until first period, or return original text if no period found."""
    period_idx = text.find('.')
    if period_idx == -1:
        return text
    return text[:period_idx]


def score_rollouts_on_eval(model, create_eval_schedule, tokenizer, cpi_name, greedy=True, fraction=1.0, seq_len=100, dp_rank=0, dp_size=1, has_first_stage=False, has_last_stage=False, dp_group=None, device='cuda', batch_size=8):
    """Evaluate unrolls against ground truth responses.
    
    Args:
        model: The model to evaluate
        create_eval_schedule: A callable that creates new schedule
        tokenizer: The tokenizer to use
        cpi_name: Name of the Chess Prompt Interface to use
        greedy: Whether to use greedy decoding
        fraction: Fraction of evaluation data to use
        seq_len: Maximum sequence length
        dp_rank: Current process rank
        dp_size: Total number of processes
        has_first_stage: Whether the model has a first stage
        has_last_stage: Whether the model has a last stage
        dp_group: The process group to use for reduction
        device: Device to run on
        
    Returns:
        Dictionary of metrics averaged across devices
    """
    # Get unrolls for this device's portion of data
    unrolls = unroll_on_eval(model, create_eval_schedule, tokenizer, cpi_name, greedy, fraction, seq_len, dp_rank, dp_size, has_first_stage, has_last_stage, batch_size, device)
    
    # Extract the move from the generated text
    cpi = get_cpi(cpi_name)
    guesses = []
    for tokens in unrolls:
        guesses.append(cpi.extract_answer(tokens, tokenizer))
    
    # Get corresponding ground truth responses
    fraction_len = max(int(len(eval_data) * fraction), batch_size) if fraction > 0 else 0
    eval_data_fraction = eval_data[:fraction_len]
    my_start = (len(eval_data_fraction) // dp_size) * dp_rank
    my_end = (len(eval_data_fraction) // dp_size) * (dp_rank + 1)
    my_eval_data = eval_data_fraction[my_start:my_end]
    best_moves = [item[1] for item in my_eval_data]
    legal_moves = [item[2] for item in my_eval_data]
    
    # Calculate metrics
    exact_matches = sum(1 for g, m in zip(guesses, best_moves) if g == m)
    legal_guesses = sum(1 for g, ms in zip(guesses, legal_moves) if g in ms)
    total = len(unrolls)
    
    # Convert to tensors for reduction
    metrics = torch.tensor([exact_matches, legal_guesses, total], device=device)
    
    # Sum across all devices
    if dp_size > 1:
        torch.distributed.all_reduce(metrics, group=dp_group)
    
    # Calculate final metrics
    global_exact_matches, global_legal_guesses, global_total = metrics.tolist()
    best_accuracy = global_exact_matches / global_total if global_total > 0 else 0.0
    legal_accuracy = global_legal_guesses / global_total if global_total > 0 else 0.0
    
    return [cpi.decode_strip_ending(x, tokenizer) for x in unrolls], {
        'best_move_accuracy_eval': best_accuracy,
        'legal_move_accuracy_eval': legal_accuracy
    }

def load_puzzle_data():
    """Load puzzle data from puzzles.json"""
    with open("puzzles.json", "r") as f:
        puzzle_data = json.load(f)
    return [(item["fen"], item["moves"]) for item in puzzle_data]

def score_rollouts_on_puzzles(model, create_eval_schedule, tokenizer, cpi_name, greedy=True, fraction=1.0, seq_len=100, dp_rank=0, dp_size=1, has_first_stage=False, has_last_stage=False, dp_group=None, batch_size=8, device='cuda'):
    """Evaluate unrolls against puzzle solutions.
    
    Args:
        model: The model to evaluate
        create_eval_schedule: A callable that creates new schedule
        tokenizer: The tokenizer to use
        cpi_name: Name of the Chess Prompt Interface to use
        greedy: Whether to use greedy decoding
        fraction: Fraction of puzzle data to use
        seq_len: Maximum sequence length
        dp_rank: Current process rank
        dp_size: Total number of processes
        batch_size: Number of examples to process in parallel
        device: Device to run on
        
    Returns:
        Dictionary of metrics averaged across devices
    """
    # need to create new schedule for each pass because the unroll shapes can be different
    rollout_puzzle_schedule = create_eval_schedule()
    # Load puzzle data
    puzzle_data = load_puzzle_data()
    fraction_len = int(len(puzzle_data) * fraction)
    puzzle_data_fraction = puzzle_data[:fraction_len]
    
    # Get my portion of the puzzle data
    my_start = (len(puzzle_data_fraction) // dp_size) * dp_rank
    my_end = (len(puzzle_data_fraction) // dp_size) * (dp_rank + 1)
    my_puzzle_data = puzzle_data_fraction[my_start:my_end]
    local_print(f"Processing {len(my_puzzle_data)} puzzles from {my_start} to {my_end} of {fraction_len} puzzles, fraction {fraction}")
    
    # Initialize CPI
    cpi = get_cpi(cpi_name)
    
    # Format each puzzle using the CPI
    tokenized_contexts = []
    for fen, moves in my_puzzle_data:
        example = {"fen": fen}
        tokens = cpi.example_to_tokens(example, tokenizer, question_only=True)
        tokenized_contexts.append(tokens)
    
    # Generate responses
    unrolls = batched_generate(
        model=model,
        tokenizer=tokenizer,
        schedule=rollout_puzzle_schedule,
        create_eval_schedule=create_eval_schedule,
        has_first_stage=has_first_stage,
        has_last_stage=has_last_stage,
        tokenized_contexts=tokenized_contexts,
        batch_size=batch_size,
        max_new_tokens=seq_len,
        greedy=greedy,
        device=device,
        token_output=True,
        only_completion=False,
        progress_bar=f"Unrolling on puzzles"
    )
    
    # Extract the move from the generated text
    guesses = [cpi.extract_answer(tokens, tokenizer) for tokens in unrolls]
    
    # Calculate metrics
    best_moves = [max(scores.items(), key=lambda x: x[1])[0] for scores in [moves for _, moves in my_puzzle_data]]
    legal_moves = [list(scores.keys()) for scores in [moves for _, moves in my_puzzle_data]]
    move_scores_achieved = [scores.get(guess, 0) for guess, (_, scores) in zip(guesses, my_puzzle_data)]
    
    exact_matches = sum(1 for g, m in zip(guesses, best_moves) if g == m)
    legal_guesses = sum(1 for g, ms in zip(guesses, legal_moves) if g in ms)
    total = len(guesses)
    total_score = sum(move_scores_achieved)
    
    # Convert to tensors for reduction
    metrics = torch.tensor([exact_matches, legal_guesses, total, total_score], device=device, dtype=torch.float32)
    
    # Sum across all devices
    if dp_size > 1:
        # TODO: deepseek model doesn't work with FSDP because it gets stuck here
        # I'm suspecting it's the summon_full_params that's the culprit
        torch.distributed.all_reduce(metrics, group=dp_group)
    
    # Calculate final metrics
    global_exact_matches, global_legal_guesses, global_total, global_total_score = metrics.tolist()
    best_accuracy = global_exact_matches / global_total if global_total > 0 else 0.0
    legal_accuracy = global_legal_guesses / global_total if global_total > 0 else 0.0
    average_score = global_total_score / global_total if global_total > 0 else 0.0
    
    return [cpi.decode_strip_ending(x, tokenizer) for x in unrolls], {
        'best_move_accuracy_puzzles': best_accuracy,
        'legal_move_accuracy_puzzles': legal_accuracy,
        'score_puzzles': average_score
    }



if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base").to("cuda")
    tokenizer = get_tokenizer("simple")
    
    # Use the same CPI as in training
    cpi_name = "fen_move"
    
    unrolls = unroll_on_eval(model, tokenizer, cpi_name, greedy=True, seq_len=100, dp_rank=0, dp_size=1)
    print(unrolls[:3])  # Print first 3 unrolls as example
