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


def _analyse_position(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    tactical_depth: int,
    tau_uni: float,
    tau_cnt: float,
) -> tuple[bool, bool, str, float, float, float, float]:
    """Return (unique, counter_intuitive, pv_string, gap, r_cnt, w_deep, w_shallow).

    Uniqueness (paper Eq. 1): gap >= tau_uni at tactical_depth.
    Counter-intuitiveness: depth-1 best move must differ from depth-tactical best move
    (the position requires deep calculation, not an obvious reply) AND r_cnt >= tau_cnt.
    w_shallow measures how balanced the position looks at depth-1 (closer to 0.5 = more balanced).
    """
    try:
        infos = engine.analyse(board, chess.engine.Limit(depth=tactical_depth), multipv=2)
    except Exception as e:
        print(f"WARNING: Stockfish analysis failed ({type(e).__name__}: {e})", flush=True)
        return False, False, "", 0.0, 0.0, 0.0, 0.5
    if not infos or not infos[0].get("pv"):
        return False, False, "", 0.0, 0.0, 0.0, 0.5
    best_move = infos[0]["pv"][0]
    pv = " ".join(move.uci() for move in infos[0]["pv"])
    w_deep = _winning_chance(infos[0]["score"])
    w2_deep = _winning_chance(infos[1]["score"]) if len(infos) >= 2 else 0.0
    gap = w_deep - w2_deep
    if gap < tau_uni:
        return False, False, pv, gap, 0.0, w_deep, w_deep
    if w_deep < 0.5:
        return True, False, pv, gap, 0.0, w_deep, w_deep
    if board.is_capture(best_move):
        captured = board.piece_at(best_move.to_square)
        if captured and _PIECE_VALUES.get(captured.piece_type, 0) >= 3:
            attacker = board.piece_at(best_move.from_square)
            attacker_val = _PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
            captured_val = _PIECE_VALUES.get(captured.piece_type, 0)
            undefended = not board.is_attacked_by(not board.turn, best_move.to_square)
            clearly_winning = captured_val - attacker_val >= 3
            if undefended or clearly_winning:
                return True, False, pv, gap, 0.0, w_deep, w_deep
    try:
        info_shallow = engine.analyse(board, chess.engine.Limit(depth=1), multipv=1)
    except Exception as e:
        print(f"WARNING: depth-1 analysis failed ({type(e).__name__}: {e})", flush=True)
        return True, False, pv, gap, 0.0, w_deep, w_deep
    if not info_shallow or not info_shallow[0].get("pv"):
        return True, False, pv, gap, 0.0, w_deep, w_deep
    best_move_shallow = info_shallow[0]["pv"][0]
    shallow_mate = info_shallow[0]["score"].relative.mate()
    if shallow_mate is not None and shallow_mate > 0:
        return True, False, pv, gap, 0.0, w_deep, w_deep
    w_shallow = _winning_chance(info_shallow[0]["score"])
    r_cnt = w_deep - w_shallow
    move_disagree = best_move != best_move_shallow
    r_cnt_effective = r_cnt + (0.1 if move_disagree else 0.0)
    counter_intuitive = r_cnt_effective >= 0.0
    return True, counter_intuitive, pv, gap, r_cnt_effective, w_deep, w_shallow


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
    tau_cnt: float = 0.1,
    tactical_depth: int = 6,
) -> tuple[torch.Tensor, list[tuple[str, str]], list[float], dict[str, int]]:
    """Return rewards, qualifying positions, r_cnt scores, and per-filter debug counts.

    Rewards: -2 (illegal), 0 (valid but not qualifying), +1 (unique + counter-intuitive + novel).
    Matches paper Eq. 6–7: uniqueness (Eq. 1) AND counter-intuitiveness (Eq. 5) AND diversity.
    debug_counts keys: n_valid, n_unique, n_counter, n_novel.
    """
    scores: list[float] = []
    qualifying: list[tuple[str, str]] = []
    r_cnt_scores: list[float] = []
    n_valid = n_unique = n_counter = n_novel = 0
    n_unique_winning = 0
    n_balanced = 0
    r_cnt_unique_sum = 0.0
    gap_novel_sum = 0.0

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

        n_valid += 1
        unique, counter_intuitive, pv, gap, r_cnt, w_deep, w_shallow = _analyse_position(
            board, engine, tactical_depth, tau_uni, tau_cnt
        )
        if not unique:
            scores.append(0.0)
            continue

        n_unique += 1
        if w_deep >= 0.5:
            n_unique_winning += 1
            r_cnt_unique_sum += r_cnt
        if counter_intuitive:
            n_counter += 1
        board_str = extract_board_position(fen)
        if not replay_buffer.is_novel(board_str, pv, tau_board, tau_pv):
            scores.append(0.0)
            continue

        n_novel += 1
        # Option A: reward gap directly (uniqueness signal, always non-zero for unique positions)
        gap_reward = min(0.7, gap * 4.0)
        # Option B: bonus for balanced positions (w_shallow < 0.55 = depth-1 doesn't see a clear winner)
        balanced = w_shallow < 0.55
        balance_bonus = 0.3 if balanced else 0.0
        reward = min(1.0, gap_reward + balance_bonus)
        if balanced:
            n_balanced += 1
        scores.append(reward)
        qualifying.append((board_str, pv))
        r_cnt_scores.append(r_cnt)
        gap_novel_sum += gap

    mean_r_cnt_unique = r_cnt_unique_sum / n_unique_winning if n_unique_winning > 0 else 0.0
    mean_gap_novel = gap_novel_sum / n_novel if n_novel > 0 else 0.0
    debug_counts = {
        "n_valid": n_valid, "n_unique": n_unique, "n_counter": n_counter, "n_novel": n_novel,
        "n_unique_winning": n_unique_winning, "mean_r_cnt_unique": mean_r_cnt_unique,
        "n_balanced": n_balanced, "mean_gap_novel": mean_gap_novel,
    }
    return torch.tensor(scores, dtype=torch.float32), qualifying, r_cnt_scores, debug_counts
