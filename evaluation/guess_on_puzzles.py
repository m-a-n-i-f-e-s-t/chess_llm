import json
import torch
import torch.distributed as dist
from batched_generate import batched_generate

def load_puzzle_data():
    """Load puzzle data from puzzles.json"""
    with open("puzzles.json", "r") as f:
        puzzle_data = json.load(f)
    return [(item["fen"], item["moves"]) for item in puzzle_data]

def score_rollouts_on_puzzles(model, tokenizer, cpi, rank, world_size, greedy=True, seq_len=100, batch_size=8, device='cuda'):
    """Evaluate unrolls against puzzle solutions.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        cpi: The Chess Prompt Interface to use
        rank: Current process rank
        world_size: Total number of processes
        greedy: Whether to use greedy decoding
        seq_len: Maximum sequence length
        batch_size: Number of examples to process in parallel
        device: Device to run on
        
    Returns:
        Tuple of (prompts, unrolls, metrics)
    """
    # Load puzzle data
    puzzle_data = load_puzzle_data()
    
    # Get my portion of the puzzle data
    my_puzzle_data = puzzle_data[rank::world_size]
    
    # Format each puzzle using the CPI
    tokenized_contexts = []
    for fen, moves in my_puzzle_data:
        example = {"fen": fen}
        tokens = cpi.example_to_tokens(example, tokenizer, question_only=True)
        tokenized_contexts.append(tokens)
    
    # Generate responses
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
    
    # Extract the move from the generated text
    guesses = [cpi.extract_answer(tokens, tokenizer) for tokens in responses]
    prompts = [item[0] for item in my_puzzle_data]  # FEN positions
    unrolls = [cpi.decode_strip_ending(x, tokenizer) for x in responses]
    
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
    if world_size > 1:
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    # Calculate final metrics
    global_exact_matches, global_legal_guesses, global_total, global_total_score = metrics.tolist()
    best_accuracy = global_exact_matches / global_total if global_total > 0 else 0.0
    legal_accuracy = global_legal_guesses / global_total if global_total > 0 else 0.0
    average_score = global_total_score / global_total if global_total > 0 else 0.0
    
    return prompts, unrolls, {
        'best_move_accuracy_puzzles': best_accuracy,
        'legal_move_accuracy_puzzles': legal_accuracy,
        'score_puzzles': average_score
    }
