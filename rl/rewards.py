import math

import torch
import chess
import chess.engine

from replay_buffer import ReplayBuffer


_PIECE_VALUES: dict[int, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


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
    """Return False if any side has >16 total pieces, >8 pawns, !=1 king, >2 queens, >2 rooks, or >2 bishops."""
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
        if len(board.pieces(chess.QUEEN, color)) > 1:
            return False
        if len(board.pieces(chess.ROOK, color)) > 2:
            return False
        if len(board.pieces(chess.BISHOP, color)) > 2:
            return False
        if len(board.pieces(chess.KNIGHT, color)) > 2:
            return False
    return True


def _winning_chance(score: chess.engine.PovScore) -> float:
    """Convert a PovScore to a winning probability in [0, 1] for the side to move."""
    cp = score.relative.score(mate_score=10000)
    return 1.0 / (1.0 + math.exp(-cp / 400.0))


def _capture_material(board: chess.Board, move: chess.Move) -> int:
    """Return material value of the piece captured by move (0 if not a capture)."""
    if not board.is_capture(move):
        return 0
    if board.is_en_passant(move):
        return 1
    captured = board.piece_at(move.to_square)
    return _PIECE_VALUES.get(captured.piece_type, 0) if captured else 0


def _is_obvious_move(board: chess.Board, move: chess.Move) -> bool:
    """Return True if the move is obviously good: a winning or equal capture (captured >= attacker)."""
    if not board.is_capture(move):
        return False
    if board.is_en_passant(move):
        return False
    captured = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)
    if not captured or not attacker:
        return False
    return _PIECE_VALUES.get(captured.piece_type, 0) >= _PIECE_VALUES.get(attacker.piece_type, 0)


def _has_eval_reversal(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int,
) -> bool:
    """Return True if the position has an evaluation reversal (one move crosses win/draw threshold)."""
    try:
        infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=2)
    except Exception:
        return False
    if not infos or len(infos) < 2:
        return False
    w1 = _winning_chance(infos[0]["score"])
    w2 = _winning_chance(infos[1]["score"])
    return (w1 > 0.65 and w2 < 0.65) or (w1 > 0.35 and w2 < 0.35)


def _analyse_position(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    tactical_depth: int,
    tau_uni: float,
) -> tuple[bool, bool, bool, bool, str, float, float, float]:
    """Return (unique, eval_reversal, non_obvious, multi_move, pv, gap, w_deep, w2_deep).

    Uniqueness: gap = w_deep - w2_deep >= tau_uni.
    Evaluation reversal: best move crosses win/draw threshold vs all other moves.
    Non-obvious: best move is NOT a winning/equal capture.
    Multi-move: after best move + opponent best reply, position ALSO has eval reversal.
    """
    try:
        infos = engine.analyse(board, chess.engine.Limit(depth=tactical_depth), multipv=2)
    except Exception as e:
        print(f"WARNING: Stockfish analysis failed ({type(e).__name__}: {e})", flush=True)
        return False, False, False, False, "", 0.0, 0.0, 0.5
    if not infos or not infos[0].get("pv"):
        return False, False, False, False, "", 0.0, 0.0, 0.5
    best_move = infos[0]["pv"][0]
    pv = " ".join(move.uci() for move in infos[0]["pv"])
    w_deep = _winning_chance(infos[0]["score"])
    w2_deep = _winning_chance(infos[1]["score"]) if len(infos) >= 2 else 0.0
    gap = w_deep - w2_deep
    if gap < tau_uni:
        return False, False, False, False, pv, gap, w_deep, w2_deep
    wins_when_others_draw  = w_deep > 0.65 and w2_deep < 0.65
    draws_when_others_lose = w_deep > 0.35 and w2_deep < 0.35
    evaluation_reversal = wins_when_others_draw or draws_when_others_lose
    if not evaluation_reversal:
        return True, False, False, False, pv, gap, w_deep, w2_deep
    non_obvious = not _is_obvious_move(board, best_move)
    multi_move = False
    if non_obvious and len(infos[0]["pv"]) >= 2:
        opp_reply = infos[0]["pv"][1]
        board.push(best_move)
        board.push(opp_reply)
        multi_move = _has_eval_reversal(board, engine, max(4, tactical_depth - 4))
        board.pop()
        board.pop()
    return True, True, non_obvious, multi_move, pv, gap, w_deep, w2_deep


def score_single_fen(
    fen: str,
    entropy: float,
    engine: chess.engine.SimpleEngine,
    replay_buffer: ReplayBuffer,
    tau_ent: float,
    tau_uni: float,
    tactical_depth: int,
    tau_board: int,
    tau_pv: int,
) -> tuple[float, bool, str, str, float, dict[str, int]]:
    """Score a single FEN. Returns (reward, is_qualifying, board_str, pv, gap, debug_flags).

    Thread-safe: only uses the passed engine instance.
    """
    debug = {"n_valid": 0, "n_unique": 0, "n_reversal": 0, "n_non_obvious": 0,
             "n_multi": 0, "n_novel": 0, "n_unique_winning": 0, "n_balanced": 0,
             "n_true_puzzles": 0, "gap_novel": 0.0}

    if not is_valid_fen(fen):
        return -2.0, False, "", "", 0.0, debug

    if not is_realistic_piece_count(fen):
        return -1.0, False, "", "", 0.0, debug

    board = chess.Board(fen)
    if not board.is_valid() or board.is_game_over():
        return -1.0, False, "", "", 0.0, debug

    if entropy < tau_ent:
        return 0.0, False, "", "", 0.0, debug

    debug["n_valid"] = 1
    unique, eval_reversal, non_obvious, multi_move, pv, gap, w_deep, w2_deep = (
        _analyse_position(board, engine, tactical_depth, tau_uni)
    )
    if not unique:
        return -0.1, False, "", "", gap, debug

    debug["n_unique"] = 1
    if w_deep >= 0.5:
        debug["n_unique_winning"] = 1
    if eval_reversal:
        debug["n_reversal"] = 1
    if non_obvious:
        debug["n_non_obvious"] = 1
    if multi_move:
        debug["n_multi"] = 1

    board_str = extract_board_position(fen)
    is_novel = replay_buffer.is_novel(board_str, pv, tau_board, tau_pv)
    if is_novel:
        debug["n_novel"] = 1
        debug["gap_novel"] = gap

    if multi_move:
        debug["n_true_puzzles"] = 1

    if not non_obvious:
        return -0.1, False, board_str, pv, gap, debug

    reward = 0.5 * gap + 0.2 * float(eval_reversal) + 0.3 * float(multi_move)

    return reward, is_novel, board_str, pv, gap, debug


def compute_binary_rewards(
    fens: list[str],
    sequences: torch.Tensor,
    engine: chess.engine.SimpleEngine,
    replay_buffer: ReplayBuffer,
    model: torch.nn.Module,
    tau_board: int = 5,
    tau_pv: int = 3,
    tau_ent: float = 0.6,
    tau_uni: float = 0.5,
    tactical_depth: int = 6,
) -> tuple[torch.Tensor, list[tuple[str, str]], list[float], dict[str, int]]:
    """Return rewards, qualifying positions, r_cnt scores, and per-filter debug counts.

    Shaped reward: 0.3*gap + 0.3*reversal + 0.2*non_obvious + 0.2*multi_move
    gives a smooth gradient toward puzzle-like positions instead of sparse tiers.
    debug_counts keys: n_valid, n_unique, n_reversal, n_non_obvious, n_multi, n_novel, n_true_puzzles.
    """
    scores: list[float] = []
    qualifying: list[tuple[str, str]] = []
    r_cnt_scores: list[float] = []
    n_valid = n_unique = n_reversal = n_non_obvious = n_multi = n_novel = 0
    n_unique_winning = 0
    n_balanced = 0
    n_true_puzzles = 0
    gap_novel_sum = 0.0

    with torch.no_grad():
        entropies: list[float] = model.compute_entropy(sequences).tolist()

    for i, fen in enumerate(fens):
        if not is_valid_fen(fen):
            scores.append(-2.0)
            continue

        if not is_realistic_piece_count(fen):
            scores.append(-1.0)
            continue

        board = chess.Board(fen)
        if not board.is_valid() or board.is_game_over():
            scores.append(-1.0)
            continue

        if entropies[i] < tau_ent:
            scores.append(0.0)
            continue

        n_valid += 1
        unique, eval_reversal, non_obvious, multi_move, pv, gap, w_deep, w2_deep = (
            _analyse_position(board, engine, tactical_depth, tau_uni)
        )
        if not unique:
            scores.append(-0.1)
            continue

        n_unique += 1
        if w_deep >= 0.5:
            n_unique_winning += 1
        if eval_reversal:
            n_reversal += 1
        if non_obvious:
            n_non_obvious += 1
        if multi_move:
            n_multi += 1
        board_str = extract_board_position(fen)
        is_novel = replay_buffer.is_novel(board_str, pv, tau_board, tau_pv)
        if is_novel:
            n_novel += 1
            qualifying.append((board_str, pv))
            r_cnt_scores.append(gap)
            gap_novel_sum += gap

        if multi_move:
            n_true_puzzles += 1

        if not non_obvious:
            scores.append(-0.1)
            continue

        reward = 0.5 * gap + 0.2 * float(eval_reversal) + 0.3 * float(multi_move)
        scores.append(reward)

    mean_gap_novel = gap_novel_sum / n_novel if n_novel > 0 else 0.0
    debug_counts = {
        "n_valid": n_valid, "n_unique": n_unique, "n_reversal": n_reversal,
        "n_non_obvious": n_non_obvious, "n_multi": n_multi, "n_novel": n_novel,
        "n_unique_winning": n_unique_winning,
        "n_balanced": n_balanced, "mean_gap_novel": mean_gap_novel,
        "n_true_puzzles": n_true_puzzles,
    }
    return torch.tensor(scores, dtype=torch.float32), qualifying, r_cnt_scores, debug_counts
