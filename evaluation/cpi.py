import random


class ChessPromptInterface:
    def __init__(self, question_maker, cot_maker, answer_maker, answer_delimiter=("<answer>", "</answer>")):
        self.response_idx = 1
        # an example is a dict eg loaded from a jsonl
        self.question_maker = question_maker # function (example, tokenizer) -> list of tokens
        self.cot_maker = cot_maker # function (example, tokenizer) -> list of tokens
        self.answer_maker = answer_maker # function (example, tokenizer) -> list of tokens
        self.answer_delimiter = answer_delimiter

    def example_to_tokens(self, example, tokenizer, question_only=False, with_response_mask=False, pad_to=None):
        answer_start_tokens = tokenizer.encode(self.answer_delimiter[0], add_special_tokens=False)
        answer_end_tokens = tokenizer.encode(self.answer_delimiter[1], add_special_tokens=False)
        tokens = self.question_maker(example, tokenizer)
        if with_response_mask:
            response_mask = [0] * len(tokens)

        if not question_only:
            cot_tokens = self.cot_maker(example, tokenizer)
            answer_tokens = [*answer_start_tokens, *self.answer_maker(example, tokenizer), *answer_end_tokens]
            response_tokens = [*cot_tokens, *answer_tokens]
            tokens += response_tokens
            if with_response_mask:
                response_mask += [self.response_idx] * len(response_tokens)
                self.response_idx += 1
                self.response_idx = max(1, self.response_idx % 2**16)
        
        if pad_to is not None:
            pad_tokens = [tokens[-1]] * (pad_to - len(tokens))
            tokens = [*tokens, *pad_tokens]
            if with_response_mask:
                response_mask += [0] * len(pad_tokens)

        if with_response_mask:
            return tokens, response_mask
        else:
            return tokens

    def extract_answer(self, tokens, tokenizer):
        answer_start_tokens = tokenizer.encode(self.answer_delimiter[0], add_special_tokens=False)
        answer_end_tokens = tokenizer.encode(self.answer_delimiter[1], add_special_tokens=False)
        if not isinstance(tokens, list):
            tokens = tokens.tolist()

        start_idx = -1
        for i in range(len(tokens) - len(answer_start_tokens) + 1):
            if tokens[i:i+len(answer_start_tokens)] == answer_start_tokens:
                start_idx = i + len(answer_start_tokens)
                break
                
        if start_idx == -1:
            return ""
            
        # Find end of answer
        end_idx = len(tokens)
        for i in range(start_idx, len(tokens) - len(answer_end_tokens) + 1):
            if tokens[i:i+len(answer_end_tokens)] == answer_end_tokens:
                end_idx = i
                break
        
        if end_idx == start_idx:
            return ""
                
        return tokenizer.decode(tokens[start_idx:end_idx], skip_special_tokens=True)

    def decode_strip_ending(self, tokens, tokenizer):
        answer_start_tokens = tokenizer.encode(self.answer_delimiter[0], add_special_tokens=False)
        answer_end_tokens = tokenizer.encode(self.answer_delimiter[1], add_special_tokens=False)
        if not isinstance(tokens, list):
            tokens = tokens.tolist()
        
        answer_start_idx = -1
        for i in range(len(tokens) - len(answer_start_tokens) + 1):
            if tokens[i:i+len(answer_start_tokens)] == answer_start_tokens:
                answer_start_idx = i + len(answer_start_tokens)
                break
            
        # Find end of answer
        end_idx = len(tokens)
        for i in range(answer_start_idx, len(tokens) - len(answer_end_tokens) + 1):
            if tokens[i:i+len(answer_end_tokens)] == answer_end_tokens:
                end_idx = i + len(answer_end_tokens)
                break
                        
        return tokenizer.decode(tokens[:end_idx])

def get_top_n_moves(moves, n, shuffle=False, noise=0):
    out = sorted(moves, key=lambda x: int(moves[x]["score"]), reverse=True)[:n]
    if shuffle:
        random.shuffle(out)
    if noise > 0:
        for i in range(len(out)):
            if random.random() < noise:
                out[i] = random.choice(list(moves.keys()))
    return out

CPIS = dict(
        fen_move = ChessPromptInterface(
            question_maker=lambda x, tokenizer: tokenizer.encode(x["fen"]+'</board>'),
            cot_maker=lambda x, tokenizer: [],
            answer_maker=lambda x, tokenizer: tokenizer.encode(x["move"], add_special_tokens=False)),
        fen_thinktok10_move = ChessPromptInterface(
            question_maker=lambda x, tokenizer: tokenizer.encode(x["fen"]+'</board>'),
            cot_maker=lambda x, tokenizer: tokenizer.encode(". . . . . . . . . .</think>", add_special_tokens=False),
            answer_maker=lambda x, tokenizer: tokenizer.encode(x["move"], add_special_tokens=False)),
        fen_top3_move = ChessPromptInterface(
            question_maker=lambda x, tokenizer: tokenizer.encode(x["fen"]+'</board>'),
            cot_maker=lambda x, tokenizer: tokenizer.encode(", ".join(get_top_n_moves(x["moves"], 3, shuffle=True, noise=.05)) + "</think>", add_special_tokens=False),
            answer_maker=lambda x, tokenizer: tokenizer.encode(x["move"], add_special_tokens=False)),
        fen_english_move = ChessPromptInterface(
            question_maker=lambda x, tokenizer: tokenizer.encode('<｜User｜> You are a grandmaster chess player. What should I respond in the following position, given in FEN notation?  ' + x["fen"]+'\n<｜Assistant｜>'),
            cot_maker=lambda x, tokenizer: tokenizer.encode("<think>" + x['cot'] + "</think>", add_special_tokens=False),
            answer_maker=lambda x, tokenizer: tokenizer.encode(x["move"], add_special_tokens=False)),
    )