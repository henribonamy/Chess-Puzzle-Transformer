"""Compare pretrained vs fine-tuned checkpoint on puzzle generation metrics (HF Space)."""
import sys
import os
import threading
import time
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
import chess
import chess.engine
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pretraining"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rl"))

from model import AutoRegressiveTransformer
from tokenizer import FENTokenizer
from rewards import is_valid_fen, is_realistic_piece_count, _analyse_position

HF_PRETRAINED_REPO = "henribonamy/chess-puzzles-pretrained"
N_SAMPLES = 1024
BATCH_SIZE = 128
TEMPERATURE = 1.0
TACTICAL_DEPTH = 8
TAU_UNI = 0.2
MAX_NEW_TOKENS = 83

CHECKPOINTS = {
    "pretrained": ("outputs/model_checkpoint.pt", "model_checkpoint.pt"),
    "finetuned": ("outputs/model_checkpoint_finetuned.pt", "model_checkpoint_finetuned.pt"),
}

_log_lines: list[str] = []
_log_lock = threading.Lock()


def _log(msg: str) -> None:
    """Print and buffer a log line for the /logs endpoint."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with _log_lock:
        _log_lines.append(line)


def _start_log_server() -> None:
    """Start HTTP log server on port 7860."""
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            with _log_lock:
                body = "\n".join(_log_lines).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args: object) -> None:
            pass
    threading.Thread(target=HTTPServer(("0.0.0.0", 7860), _Handler).serve_forever, daemon=True).start()


def ensure_checkpoint(local_path: str, filename: str) -> None:
    """Download checkpoint from HF Hub if not present locally."""
    if os.path.exists(local_path):
        return
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    _log(f"Downloading {filename}...")
    hf_hub_download(repo_id=HF_PRETRAINED_REPO, filename=filename, repo_type="model", local_dir="outputs")
    _log(f"{filename} ready.")


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
                fens.append(tokenizer.decode(seq.tolist()))
            except KeyError:
                fens.append("")
        _log(f"  Generated {len(fens)}/{n} FENs")
    return fens


def evaluate_fens(fens: list[str], engine: chess.engine.SimpleEngine) -> dict:
    """Run Stockfish analysis on all FENs and return aggregated metrics."""
    n_total = len(fens)
    n_playable = n_unique = n_reversal = n_non_obvious = n_multi = 0
    gaps = []

    for i, fen in enumerate(fens):
        if i % 100 == 0:
            _log(f"  Stockfish [{i}/{n_total}] playable={n_playable} unique={n_unique} multi={n_multi}")
        if not is_valid_fen(fen) or not is_realistic_piece_count(fen):
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


def log_results(name: str, metrics: dict) -> None:
    """Log formatted metrics for one checkpoint."""
    _log(f"{'='*50}")
    _log(f"  {name.upper()}")
    _log(f"{'='*50}")
    _log(f"  Samples:        {metrics['n_total']}")
    _log(f"  Valid+playable: {metrics['valid_pct']:.1f}%  (n={metrics['n_playable']})")
    _log(f"  Unique (gap>={TAU_UNI}): {metrics['unique_pct']:.1f}%  (n={metrics['n_unique']})")
    _log(f"  Eval reversal:  {metrics['reversal_pct']:.1f}%  (n={metrics['n_reversal']})")
    _log(f"  Non-obvious:    {metrics['non_obvious_pct']:.1f}%  (n={metrics['n_non_obvious']})")
    _log(f"  Multi-move:     {metrics['multi_move_pct']:.1f}%  (n={metrics['n_multi']})")
    _log(f"  Mean gap:       {metrics['mean_gap']:.4f}")


def push_results(results: dict) -> None:
    """Push comparison results JSON to HF Hub."""
    os.makedirs("outputs", exist_ok=True)
    path = "outputs/comparison_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    try:
        api = HfApi()
        api.upload_file(path_or_fileobj=path, path_in_repo="comparison_results.json",
                        repo_id=HF_PRETRAINED_REPO, repo_type="model")
        _log("Pushed comparison_results.json to HF Hub")
    except Exception as e:
        _log(f"WARNING: Could not push results ({e})")


def main() -> None:
    """Run comparison between pretrained and fine-tuned checkpoints."""
    _start_log_server()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    for local_path, filename in CHECKPOINTS.values():
        ensure_checkpoint(local_path, filename)

    stockfish_path = (
        __import__("shutil").which("stockfish")
        or "/usr/games/stockfish"
        or "/usr/bin/stockfish"
    )
    _log(f"Stockfish: {stockfish_path}")
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 4, "Hash": 512})

    tokenizer = FENTokenizer()
    results = {}

    for name, (local_path, _) in CHECKPOINTS.items():
        _log(f"\nLoading {name}...")
        model = AutoRegressiveTransformer().to(device)
        ckpt = torch.load(local_path, map_location=device)
        model.load_state_dict(ckpt)
        model.eval()

        _log(f"Generating {N_SAMPLES} FENs...")
        fens = generate_fens(model, tokenizer, device, N_SAMPLES)

        _log(f"Evaluating with Stockfish (depth={TACTICAL_DEPTH})...")
        metrics = evaluate_fens(fens, engine)
        results[name] = metrics
        log_results(name, metrics)
        del model

    engine.quit()

    _log(f"\n{'='*50}")
    _log("  DELTA (finetuned - pretrained)")
    _log(f"{'='*50}")
    pre = results["pretrained"]
    fin = results["finetuned"]
    for key in ["valid_pct", "unique_pct", "reversal_pct", "non_obvious_pct", "multi_move_pct", "mean_gap"]:
        delta = fin[key] - pre[key]
        sign = "+" if delta >= 0 else ""
        _log(f"  {key:<22}: {sign}{delta:.2f}")

    push_results(results)
    _log("Done.")

    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
