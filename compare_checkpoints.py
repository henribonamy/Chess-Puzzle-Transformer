"""Compare pretrained vs fine-tuned checkpoint on puzzle generation metrics."""
import sys
import os

import torch
import chess
import chess.engine

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "pretraining"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "rl_training"))

from model import AutoRegressiveTransformer
from tokenizer import FENTokenizer
from rewards import is_valid_fen, is_realistic_piece_count, _analyse_position

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
N_SAMPLES = 512
BATCH_SIZE = 64
TEMPERATURE = 1.0
TACTICAL_DEPTH = 8
TAU_UNI = 0.2
MAX_NEW_TOKENS = 83

CHECKPOINTS = {
    "pretrained": "outputs/model_checkpoint.pt",
    "finetuned": "outputs/model_checkpoint_finetuned.pt",
}


def generate_fens(model: AutoRegressiveTransformer, tokenizer: FENTokenizer, device: torch.device, n: int) -> list[str]:
    """Generate n FEN strings from model using batched sampling."""
    fens = []
    while len(fens) < n:
        batch = min(BATCH_SIZE, n - len(fens))
        start = torch.zeros((batch, 1), dtype=torch.long, device=device)
        with torch.no_grad():
            sequences = model.generate(start, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
        for seq in sequences:
            try:
                fen = tokenizer.decode(seq.tolist())
                fens.append(fen)
            except KeyError:
                fens.append("")
    return fens


def evaluate_fens(fens: list[str], engine: chess.engine.SimpleEngine) -> dict:
    """Run Stockfish analysis on all FENs and return aggregated metrics."""
    n_total = len(fens)
    n_valid = n_realistic = n_playable = 0
    n_unique = n_reversal = n_non_obvious = n_multi = 0
    gaps = []

    for i, fen in enumerate(fens):
        if i % 50 == 0:
            print(f"  [{i}/{len(fens)}] valid={n_playable} unique={n_unique} multi={n_multi}", flush=True)
        if not is_valid_fen(fen):
            continue
        if not is_realistic_piece_count(fen):
            continue
        board = chess.Board(fen)
        if not board.is_valid() or board.is_game_over():
            continue
        n_playable += 1

        unique, eval_reversal, non_obvious, multi_move, pv, gap, w_deep, w2_deep = (
            _analyse_position(board, engine, TACTICAL_DEPTH, TAU_UNI)
        )
        if unique:
            n_unique += 1
            gaps.append(gap)
        if eval_reversal:
            n_reversal += 1
        if non_obvious:
            n_non_obvious += 1
        if multi_move:
            n_multi += 1

    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    return {
        "n_total": n_total,
        "valid_pct": 100.0 * n_playable / n_total,
        "unique_pct": 100.0 * n_unique / max(n_playable, 1),
        "reversal_pct": 100.0 * n_reversal / max(n_playable, 1),
        "non_obvious_pct": 100.0 * n_non_obvious / max(n_playable, 1),
        "multi_move_pct": 100.0 * n_multi / max(n_playable, 1),
        "mean_gap": mean_gap,
        "n_playable": n_playable,
        "n_unique": n_unique,
        "n_reversal": n_reversal,
        "n_non_obvious": n_non_obvious,
        "n_multi": n_multi,
    }


def print_results(name: str, metrics: dict) -> None:
    """Print formatted metrics for one checkpoint."""
    print(f"\n{'='*50}")
    print(f"  {name.upper()}")
    print(f"{'='*50}")
    print(f"  Samples:        {metrics['n_total']}")
    print(f"  Valid+playable: {metrics['valid_pct']:.1f}%  (n={metrics['n_playable']})")
    print(f"  Unique (gap>={TAU_UNI}): {metrics['unique_pct']:.1f}%  (n={metrics['n_unique']})")
    print(f"  Eval reversal:  {metrics['reversal_pct']:.1f}%  (n={metrics['n_reversal']})")
    print(f"  Non-obvious:    {metrics['non_obvious_pct']:.1f}%  (n={metrics['n_non_obvious']})")
    print(f"  Multi-move:     {metrics['multi_move_pct']:.1f}%  (n={metrics['n_multi']})")
    print(f"  Mean gap:       {metrics['mean_gap']:.4f}")


def main() -> None:
    """Run comparison between pretrained and fine-tuned checkpoints."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    tokenizer = FENTokenizer()

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 4, "Hash": 256})

    results = {}
    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\nLoading {name} from {ckpt_path}...")
        model = AutoRegressiveTransformer().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
        model.eval()

        print(f"Generating {N_SAMPLES} FENs...", flush=True)
        fens = generate_fens(model, tokenizer, device, N_SAMPLES)

        print(f"Evaluating with Stockfish (depth={TACTICAL_DEPTH})...", flush=True)
        metrics = evaluate_fens(fens, engine)
        results[name] = metrics
        print_results(name, metrics)

        del model

    engine.quit()

    print(f"\n{'='*50}")
    print("  DELTA (finetuned - pretrained)")
    print(f"{'='*50}")
    pre = results["pretrained"]
    fin = results["finetuned"]
    for key in ["valid_pct", "unique_pct", "reversal_pct", "non_obvious_pct", "multi_move_pct", "mean_gap"]:
        delta = fin[key] - pre[key]
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<22}: {sign}{delta:.2f}")


if __name__ == "__main__":
    main()
