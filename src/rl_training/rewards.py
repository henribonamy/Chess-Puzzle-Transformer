import math

import torch
import chess
import chess.engine

from replay_buffer import ReplayBuffer


def extract_board_position(fen: str) -> str:
    """Extract the first 4 FEN fields as a canonical board position string."""
    parts = fen.split(" ")
    if len(parts) < 4:
        return fen
    return " ".join(parts[:4])


def is_valid_fen(fen: str) -> bool:
    """Return True if the FEN string represents a legal chess position."""
    try:
        chess.Board(fen)
        return True
    except Exception:
        return False


def is_realistic_piece_count(fen: str) -> bool:
    """Return False if any side has >16 total pieces, >8 pawns, or != 1 king."""
    try:
        board = chess.Board(fen)
    except Exception:
        return False
    for color in (chess.WHITE, chess.BLACK):
        total = len(board.pieces(chess.PAWN, color) | board.pieces(chess.KNIGHT, color) |
                    board.pieces(chess.BISHOP, color) | board.pieces(chess.ROOK, color) |
                    board.pieces(chess.QUEEN, color) | board.pieces(chess.KING, color))
        if total > 16:
            return False
        if len(board.pieces(chess.PAWN, color)) > 8:
            return False
        if len(board.pieces(chess.KING, color)) != 1:
            return False
    return True


def _winning_chance(score: chess.engine.PovScore) -> float:
    """Convert a PovScore to a winning probability in [0, 1] for the side to move."""
    cp = score.relative.score(mate_score=10000)
    return 1.0 / (1.0 + math.exp(-cp / 400.0))


def _check_uniqueness(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int,
    tau_uni: float,
) -> tuple[bool, str, float]:
    """Return (passes_uniqueness, pv_string, uniqueness_score).

    Uniqueness holds when the best move's winning chance exceeds the second best
    by at least tau_uni. The score is w1-w2 in [0, 1].
    """
    try:
        infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=2)
    except Exception as e:
        print(f"WARNING: Stockfish analysis failed ({type(e).__name__}: {e})", flush=True)
        return False, "", 0.0
    if not infos:
        return False, "", 0.0
    pv = " ".join(move.uci() for move in infos[0].get("pv", []))
    if len(infos) < 2:
        return True, pv, 1.0
    w1 = _winning_chance(infos[0]["score"])
    w2 = _winning_chance(infos[1]["score"])
    diff = w1 - w2
    return diff >= tau_uni, pv, diff


def _shallow_top_move(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int,
) -> str:
    """Return the UCI string of the top move at the given shallow depth, or empty string on failure."""
    try:
        infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
        if infos and infos[0].get("pv"):
            return infos[0]["pv"][0].uci()
    except Exception as e:
        print(f"WARNING: Shallow Stockfish analysis failed ({type(e).__name__}: {e})", flush=True)
    return ""


def compute_binary_rewards(
    fens: list[str],
    sequences: torch.Tensor,
    engine: chess.engine.SimpleEngine,
    replay_buffer: ReplayBuffer,
    model: torch.nn.Module,
    tau_board: int = 5,
    tau_pv: int = 3,
    tau_ent: float = 0.5,
    tau_uni: float = 0.3,
    tactical_depth: int = 10,
    shallow_depth: int = 4,
) -> tuple[torch.Tensor, list[tuple[str, str]], list[float], list[bool]]:
    """Return rewards, qualifying positions, uniqueness scores, and counter-intuitive flags.

    Rewards: -2 (illegal), 0 (valid but fails), +1 (qualifying), +2 (qualifying + counter-intuitive).
    """
    scores: list[float] = []
    qualifying: list[tuple[str, str]] = []
    uniqueness_scores: list[float] = []
    counter_flags: list[bool] = []

    with torch.no_grad():
        entropies: list[float] = model.compute_entropy(sequences).tolist()

    for i, fen in enumerate(fens):
        if not is_valid_fen(fen):
            scores.append(-2.0)
            continue

        if not is_realistic_piece_count(fen):
            scores.append(0.0)
            continue

        board = chess.Board(fen)
        if not board.is_valid() or board.is_game_over():
            scores.append(0.0)
            continue

        if entropies[i] < tau_ent:
            scores.append(0.0)
            continue

        unique, pv, uni_score = _check_uniqueness(board, engine, tactical_depth, tau_uni)
        if not unique:
            scores.append(0.0)
            continue

        board_str = extract_board_position(fen)
        if not replay_buffer.is_novel(board_str, pv, tau_board, tau_pv):
            scores.append(0.0)
            continue

        deep_top = pv.split()[0] if pv else ""
        shallow_top = _shallow_top_move(board, engine, shallow_depth)
        is_counter_intuitive = bool(deep_top and shallow_top and deep_top != shallow_top)

        scores.append(2.0 if is_counter_intuitive else 1.0)
        qualifying.append((board_str, pv))
        uniqueness_scores.append(uni_score)
        counter_flags.append(is_counter_intuitive)

    return torch.tensor(scores, dtype=torch.float32), qualifying, uniqueness_scores, counter_flags
