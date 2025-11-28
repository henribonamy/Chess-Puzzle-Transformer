fen_chars = [
    'r','n','b','q','k','p',      # black pieces
    'R','N','B','Q','K','P',      # white pieces
    '.',                           # empty square
    '/',                           # row separator
    'w','b',                       # side to move
    'K','Q','k','q',               # castling rights
    '-',                           # no en-passant
    '0','1','2','3','4','5','6','7','8'  # half/full move counters
]

char2id = {c:i for i,c in enumerate(fen_chars)}
id2char = {i:c for c,i in char2id.items()}

def tokenize(fen: str) -> str:
    """Tokenizes a FEN string into a fixed-length representation."""
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

    tokenized_halfmove = halfmove_clock.rjust(2, ".")
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
