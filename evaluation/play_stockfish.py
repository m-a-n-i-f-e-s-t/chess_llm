import chess.engine
import chess
import random
from batched_generate import batched_generate

import urllib.request
import io
import zipfile
from tqdm import tqdm

import torch
import torch.distributed as dist

def make_engine(model, tokenizer, cpi, greedy, seq_len, batch_size, device):
    def model_engine(boards):
        tokenized_contexts = []
        for board in boards:
            example = {"fen": board.fen()}
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
        moves = [cpi.extract_answer(tokens, tokenizer) for tokens in responses]
        legal_moves = [list(board.legal_moves) for board in boards]
        san_moves = []
        for board, move, legal_moveset in zip(boards, moves, legal_moves):
            try:
                san_moves.append(board.parse_san(move))
            except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError) as e:
                san_moves.append(random.choice(legal_moveset) if legal_moveset else None)
        return san_moves
    return model_engine

# read opening book

def get_openings():
    link = "https://github.com/official-stockfish/books/raw/refs/heads/master/noob_3moves.epd.zip"
    response = urllib.request.urlopen(link)
    zip_data = io.BytesIO(response.read())
    with zipfile.ZipFile(zip_data) as zip_file:
        # Read the .epd file from the zip
        with zip_file.open('noob_3moves.epd') as f:
            openings = f.read().decode('utf-8').splitlines()
    return openings

def play_games(fens, white_engine, black_engine):
    boards = [chess.Board(fen) for fen in fens]
    white_boards = [board for board in boards if board.turn == chess.WHITE]
    black_boards = [board for board in boards if board.turn == chess.BLACK]
    while not all(board.is_game_over() for board in boards):
        if white_boards:
            white_results = white_engine(white_boards)
            white_results = [result if not board.is_game_over() else None for board, result in zip(white_boards, white_results)]
            for board, result in zip(white_boards, white_results):
                if result is not None:
                    board.push(result)
        if black_boards:
            black_results = black_engine(black_boards)
            black_results = [result if not board.is_game_over() else None for board, result in zip(black_boards, black_results)]
            for board, result in zip(black_boards, black_results):
                if result is not None:
                    board.push(result)
        boards = white_boards + black_boards
    return sum([1 if board.outcome().winner == chess.WHITE else 0 if board.outcome().winner == chess.BLACK else 0.5
                for board in boards]) / len(boards)

def play_series(fens, engine1, engine2):
    games1 = play_games(fens, engine1, engine2)
    games2 = play_games(fens, engine2, engine1)
    return (games1 + (1 - games2)) / 2

def random_engine(boards):
    legal_moves = [list(board.legal_moves) for board in boards]
    moves = [random.choice(legal_moves) if legal_moves else None for legal_moves in legal_moves]
    return moves


def play_against_stockfish(opponent : callable, depth : int, batch_size : int, games : int):
    assert games % batch_size == 0

    stockfish_engine = chess.engine.SimpleEngine.popen_uci(r"/root/Stockfish/src/stockfish")
    def stockfish(boards):
        results = [stockfish_engine.play(board, chess.engine.Limit(depth=depth)) for board in boards]
        return [result.move for result in results]
        
    # get random opening
    openings = get_openings()
    score = 0
    for i in tqdm(range(0, games), desc="Playing against Stockfish"):
        fens = [random.choice(openings).strip() for _ in range(batch_size)]
        result = play_series(fens, opponent, stockfish)
        score += result
    stockfish_engine.quit()

    if dist.is_initialized():
        to_sync = torch.tensor([score, games])
        dist.all_reduce(to_sync, op=dist.ReduceOp.SUM)
        score, games = to_sync.tolist()

    return score / games

if __name__ == "__main__":
    print(play_against_stockfish(random_engine, depth=1, batch_size=10, games=100))