import torch
import chess
import chess.engine


def extract_board_position(fen: str) -> str:
    """Extract the first 4 FEN fields as a canonical board position string."""
    parts = fen.split(" ")
    if len(parts) < 4:
        return fen
    return " ".join(parts[:4])


def realism_reward(fens: list[str], engine: chess.engine.SimpleEngine) -> torch.Tensor:
    """Return per-FEN realism scores in [0, 1] based on Stockfish evaluation."""
    scores: list[float] = []
    for fen in fens:
        try:
            board = chess.Board(fen)
        except Exception:
            scores.append(0.0)
            continue

        if board.is_game_over():
            scores.append(0.0)
            continue

        try:
            info = engine.analyse(board, chess.engine.Limit(depth=5))
        except Exception:
            scores.append(0.0)
            continue

        score = info["score"].relative
        if score.is_mate():
            mate_n = score.mate()
            if mate_n is None or mate_n <= 0:
                scores.append(0.0)
            else:
                scores.append(0.8)
        else:
            cp = score.score()
            if cp is None:
                scores.append(0.0)
            elif abs(cp) < 900:
                scores.append(1.0 - abs(cp) / 900.0)
            else:
                scores.append(0.0)

    return torch.tensor(scores, dtype=torch.float32)


def uniqueness_reward(fens: list[str], board_position_set: set[str]) -> torch.Tensor:
    """Return 1.0 for positions not in the training set, 0.0 for duplicates."""
    scores = [
        0.0 if extract_board_position(fen) in board_position_set else 1.0
        for fen in fens
    ]
    return torch.tensor(scores, dtype=torch.float32)


def diversity_reward(fens: list[str]) -> torch.Tensor:
    """Return per-sample diversity scores based on pairwise uniqueness within the batch."""
    n = len(fens)
    if n <= 1:
        return torch.ones(n, dtype=torch.float32)

    board_positions = [extract_board_position(fen) for fen in fens]
    scores = [
        sum(1 for j in range(n) if j != i and board_positions[j] != board_positions[i]) / (n - 1)
        for i in range(n)
    ]
    return torch.tensor(scores, dtype=torch.float32)


def compute_rewards(
    fens: list[str],
    engine: chess.engine.SimpleEngine,
    board_position_set: set[str],
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> torch.Tensor:
    """Combine realism, uniqueness, and diversity into a single weighted reward tensor."""
    w_realism, w_uniqueness, w_diversity = weights
    r = realism_reward(fens, engine)
    u = uniqueness_reward(fens, board_position_set)
    d = diversity_reward(fens)
    return w_realism * r + w_uniqueness * u + w_diversity * d
