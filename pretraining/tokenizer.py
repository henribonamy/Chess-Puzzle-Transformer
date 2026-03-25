class FENTokenizer:

    fen_chars = [
        'r','n','b','q','k','p',
        'R','N','B','Q','K','P',
        '.',
        '/',
        'w','b',
        'K','Q','k','q',
        '-',
        'a','b','c','d','e','f','g','h',
        '0','1','2','3','4','5','6','7','8','9'
    ]

    char2id = {c:i for i,c in enumerate(fen_chars)}
    id2char = {i:c for c,i in char2id.items()}

    def __init__(self):
        self.char2id = self.char2id
        self.id2char = self.id2char
        self.vocab_size = len(self.fen_chars)

    def tokenize(self, fen: str) -> str:
        parts = fen.split(" ")
        board_part = parts[0]
        active_player = parts[1]
        castling_availability = parts[2]
        en_passant_target = parts[3]
        halfmove_clock = parts[4]
        fullmove_number = parts[5]

        tokenized_board = ""
        for char in board_part:
            if char.isdigit():
                tokenized_board += "." * int(char)
            else:
                tokenized_board += char

        tokenized_active_player = active_player
        tokenized_castling = castling_availability.ljust(4, ".")

        if en_passant_target == "-":
            tokenized_en_passant = "-."
        else:
            tokenized_en_passant = en_passant_target

        tokenized_halfmove = halfmove_clock.rjust(3, ".")
        tokenized_fullmove = fullmove_number.rjust(3, ".")

        tokenized_fen = (
            tokenized_board
            + tokenized_active_player
            + tokenized_castling
            + tokenized_en_passant
            + tokenized_halfmove
            + tokenized_fullmove
        )

        return tokenized_fen

    def detokenize(self, tokenized_fen: str) -> str:
        board_part = tokenized_fen[:71]
        active_player = tokenized_fen[71]
        castling = tokenized_fen[72:76]
        en_passant = tokenized_fen[76:78]
        halfmove = tokenized_fen[78:81]
        fullmove = tokenized_fen[81:84]

        compressed_board = ""
        ranks = board_part.split("/")
        for rank in ranks:
            compressed_rank = ""
            empty_count = 0
            for char in rank:
                if char == ".":
                    empty_count += 1
                else:
                    if empty_count > 0:
                        compressed_rank += str(empty_count)
                        empty_count = 0
                    compressed_rank += char
            if empty_count > 0:
                compressed_rank += str(empty_count)
            compressed_board += compressed_rank + "/"
        compressed_board = compressed_board.rstrip("/")

        castling_cleaned = castling.replace(".", "")
        if not castling_cleaned:
            castling_cleaned = "-"

        if en_passant == "-.":
            en_passant_cleaned = "-"
        else:
            en_passant_cleaned = en_passant

        halfmove_cleaned = halfmove.replace(".", "")
        if not halfmove_cleaned:
            halfmove_cleaned = "0"

        fullmove_cleaned = fullmove.replace(".", "")
        if not fullmove_cleaned:
            fullmove_cleaned = "1"

        valid_fen = " ".join([
            compressed_board,
            active_player,
            castling_cleaned,
            en_passant_cleaned,
            halfmove_cleaned,
            fullmove_cleaned
        ])

        return valid_fen

    def encode(self, fen: str) -> list[int]:
        tokenized = self.tokenize(fen)
        return [self.char2id[char] for char in tokenized]

    def decode(self, token_ids: list[int]) -> str:
        tokenized = "".join([self.id2char[idx] for idx in token_ids])
        return self.detokenize(tokenized)


def tokenize(fen: str) -> str:
    tokenizer = FENTokenizer()
    return tokenizer.tokenize(fen)
