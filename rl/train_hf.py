import sys
import os

# Set writable cache/home dirs before any HF imports
os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache")
os.environ.setdefault("HOME", "/tmp")

import json
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
import chess
import chess.engine
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pretraining"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import ChessDataset
from model import AutoRegressiveTransformer
from tokenizer import FENTokenizer
from rewards import compute_binary_rewards, extract_board_position, score_single_fen
from replay_buffer import ReplayBuffer

HF_PRETRAINED_REPO = "henribonamy/chess-puzzles-pretrained"
HF_RL_REPO = "henribonamy/chess-puzzles-rl"
HF_DATA_REPO = "henribonamy/chess-puzzles-data"

BATCH_SIZE = 256
ACCUM_STEPS = 1
LR = 1e-6
PPO_EPOCHS = 1
PPO_EPS = 0.1
KL_COEFF = 0.3
SL_COEFF = 0.1
ENTROPY_COEFF = 0.02
NUM_STEPS = 1000
LOG_INTERVAL = 10
CHECKPOINT_INTERVAL = 100
DATA_MIX_SIZE = 100_000
REPLAY_BUFFER_SIZE = 10_000
BUFFER_SEED_SIZE = 10_000
TAU_BOARD = 5
TAU_PV = 3
TAU_ENT = 0.0
TAU_UNI = 0.2
TACTICAL_DEPTH = 6
NUM_STOCKFISH_WORKERS = 4
_BASE_DIR = "/tmp/chess_rl"
PRETRAINED_CHECKPOINT_PATH = f"{_BASE_DIR}/outputs/model_checkpoint_1000_iterations_128_bs.pt"
RL_CHECKPOINT_DIR = f"{_BASE_DIR}/outputs/rl_checkpoints"
DATA_PATH = f"{_BASE_DIR}/data/encoded_fens.npy"
HIGH_RATED_INDICES_PATH = f"{_BASE_DIR}/data/high_rated_indices.npy"


def ensure_high_rated_indices() -> None:
    """Download high-rated puzzle indices from HF Hub if not present locally."""
    if os.path.exists(HIGH_RATED_INDICES_PATH):
        return
    data_dir = os.path.dirname(HIGH_RATED_INDICES_PATH)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading high_rated_indices.npy from {HF_DATA_REPO}...")
    hf_hub_download(
        repo_id=HF_DATA_REPO,
        filename="high_rated_indices.npy",
        repo_type="dataset",
        local_dir=data_dir,
    )
    print("High-rated indices downloaded.")


def ensure_data() -> None:
    """Download encoded FENs from HF Hub, or run preprocessing as fallback."""
    if os.path.exists(DATA_PATH):
        return
    data_dir = os.path.dirname(DATA_PATH)
    os.makedirs(data_dir, exist_ok=True)
    try:
        print(f"Downloading encoded FENs from {HF_DATA_REPO}...")
        hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename="encoded_fens.npy",
            repo_type="dataset",
            local_dir=data_dir,
        )
        print("Data downloaded.")
    except Exception as e:
        raise RuntimeError(f"Failed to download encoded FENs from {HF_DATA_REPO}: {e}")


def ensure_pretrained_checkpoint() -> None:
    """Download the pretrained checkpoint from HF Hub if not present locally."""
    if os.path.exists(PRETRAINED_CHECKPOINT_PATH):
        return
    print(f"Checkpoint not found locally — downloading from {HF_PRETRAINED_REPO}...")
    out_dir = os.path.dirname(PRETRAINED_CHECKPOINT_PATH)
    os.makedirs(out_dir, exist_ok=True)
    hf_hub_download(
        repo_id=HF_PRETRAINED_REPO,
        filename="model_checkpoint_finetuned.pt",
        local_dir=out_dir,
    )
    dl_path = os.path.join(out_dir, "model_checkpoint_finetuned.pt")
    if dl_path != PRETRAINED_CHECKPOINT_PATH:
        os.rename(dl_path, PRETRAINED_CHECKPOINT_PATH)


def find_latest_rl_checkpoint() -> tuple[str | None, int]:
    """Return (local_path, step) of the latest RL checkpoint from HF Hub, or (None, 0)."""
    api = HfApi()
    try:
        files = api.list_repo_files(HF_RL_REPO, repo_type="model")
    except Exception:
        return None, 0
    rl_files = [f for f in files if f.startswith("rl_step_") and f.endswith(".pt")]
    if not rl_files:
        return None, 0
    # parse step numbers and find max
    def _step(f: str) -> int:
        try:
            return int(f.replace("rl_step_", "").replace(".pt", ""))
        except ValueError:
            return 0
    latest = max(rl_files, key=_step)
    step = _step(latest)
    local_path = os.path.join(RL_CHECKPOINT_DIR, latest)
    if not os.path.exists(local_path):
        os.makedirs(RL_CHECKPOINT_DIR, exist_ok=True)
        print(f"Resuming from RL checkpoint {latest} (step {step})...")
        hf_hub_download(repo_id=HF_RL_REPO, filename=latest, local_dir=RL_CHECKPOINT_DIR)
    return local_path, step


def push_checkpoint_to_hub(local_path: str, filename: str) -> None:
    """Upload a checkpoint file to the RL model repo on HF Hub."""
    api = HfApi()
    api.create_repo(HF_RL_REPO, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=filename,
        repo_id=HF_RL_REPO,
        repo_type="model",
    )


def seed_replay_buffer(replay_buffer: ReplayBuffer, dataset: ChessDataset, tokenizer: FENTokenizer, n: int) -> None:
    """Seed the replay buffer with n training positions for novelty filtering."""
    indices = torch.randperm(len(dataset))[:n].tolist()
    seeded = 0
    for idx in tqdm(indices, desc="Seeding replay buffer"):
        try:
            fen_str = tokenizer.decode(dataset.data[idx].tolist())
            board_str = extract_board_position(fen_str)
            replay_buffer.seed(board_str)
            seeded += 1
        except Exception:
            pass
    _log(f"Seeded replay buffer with {seeded} training positions")


def decode_sequences(sequences: torch.Tensor, tokenizer: FENTokenizer) -> list[str]:
    """Decode a batch of token id tensors to FEN strings, returning empty string on failure."""
    fens: list[str] = []
    for i in range(sequences.size(0)):
        try:
            fens.append(tokenizer.decode(sequences[i].cpu().tolist()))
        except Exception:
            fens.append("")
    return fens


def sample_sl_batch(dataset: ChessDataset, sl_indices: list[int], batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample a random batch from the pre-selected SL indices."""
    chosen = torch.randint(len(sl_indices), (batch_size,))
    return torch.stack([dataset[sl_indices[i.item()]] for i in chosen]).to(device)


_log_lines: list[str] = []
_log_lock = threading.Lock()

def _log(msg: str) -> None:
    """Print and buffer a log line for the /logs endpoint."""
    print(msg, flush=True)
    with _log_lock:
        _log_lines.append(msg)
        if len(_log_lines) > 200:
            _log_lines.pop(0)


def _start_health_server() -> None:
    """Start HTTP server on port 7860: / returns health, /logs returns recent training output."""
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/logs":
                with _log_lock:
                    body = "\n".join(_log_lines).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Training in progress")
        def log_message(self, *args: object) -> None:
            pass
    threading.Thread(target=HTTPServer(("0.0.0.0", 7860), _Handler).serve_forever, daemon=True).start()


def _parallel_compute_rewards(
    fens: list[str],
    sequences: torch.Tensor,
    engines: list[chess.engine.SimpleEngine],
    engine_locks: list[threading.Lock],
    replay_buffer: ReplayBuffer,
    entropies: list[float],
    tau_board: int = 5,
    tau_pv: int = 3,
    tau_ent: float = 0.3,
    tau_uni: float = 0.2,
    tactical_depth: int = 6,
) -> tuple[torch.Tensor, list[tuple[str, str]], list[float], dict[str, int]]:
    """Parallel version of compute_binary_rewards using multiple Stockfish engines."""
    n_workers = len(engines)
    results: list[tuple[float, bool, str, str, float, dict[str, int]]] = [None] * len(fens)  # type: ignore

    def _process(idx: int) -> None:
        worker_id = idx % n_workers
        with engine_locks[worker_id]:
            results[idx] = score_single_fen(
                fens[idx], entropies[idx], engines[worker_id], replay_buffer,
                tau_ent, tau_uni, tactical_depth, tau_board, tau_pv,
            )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(_process, range(len(fens))))

    scores: list[float] = []
    qualifying: list[tuple[str, str]] = []
    r_cnt_scores: list[float] = []
    agg_debug: dict[str, float] = {}
    for reward, is_novel, board_str, pv, gap, debug in results:
        scores.append(reward)
        if is_novel and board_str:
            qualifying.append((board_str, pv))
            r_cnt_scores.append(gap)
        for k, v in debug.items():
            agg_debug[k] = agg_debug.get(k, 0) + v

    mean_gap = agg_debug.pop("gap_novel", 0.0)
    n_novel = int(agg_debug.get("n_novel", 0))
    agg_debug["mean_gap_novel"] = mean_gap / n_novel if n_novel > 0 else 0.0
    debug_counts = {k: int(v) if k != "mean_gap_novel" else v for k, v in agg_debug.items()}
    return torch.tensor(scores, dtype=torch.float32), qualifying, r_cnt_scores, debug_counts


def main() -> None:
    """Run PPO RL training with HF Hub checkpoint pushing."""
    _start_health_server()
    ensure_data()
    ensure_high_rated_indices()
    ensure_pretrained_checkpoint()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    _log(f"Using device: {device}")

    tokenizer = FENTokenizer()
    dataset = ChessDataset(DATA_PATH)

    sl_indices = np.load(HIGH_RATED_INDICES_PATH).tolist()
    _log(f"Loaded {len(sl_indices)} high-rated SL indices from {HIGH_RATED_INDICES_PATH}")

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    seed_replay_buffer(replay_buffer, dataset, tokenizer, BUFFER_SEED_SIZE)

    model = AutoRegressiveTransformer().to(device)
    state_dict = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    resume_step = 0
    _log(f"Loaded pretrained checkpoint from {PRETRAINED_CHECKPOINT_PATH}")

    ref_model = AutoRegressiveTransformer().to(device)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    use_amp = device.type == "cuda"
    autocast_ctx = lambda: torch.autocast("cuda", dtype=torch.float16) if use_amp else torch.autocast("cpu", enabled=False)
    if use_amp:
        _log("fp16 autocast enabled")

    os.makedirs(RL_CHECKPOINT_DIR, exist_ok=True)
    backup_path = os.path.join(RL_CHECKPOINT_DIR, "pretrained_backup.pt")
    torch.save(model.state_dict(), backup_path)

    stockfish_path = (
        shutil.which("stockfish")
        or "/usr/games/stockfish"
        or "/usr/bin/stockfish"
        or "/opt/homebrew/bin/stockfish"
    )

    def _open_engine() -> chess.engine.SimpleEngine:
        return chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def _close_engine(eng: chess.engine.SimpleEngine) -> None:
        try:
            eng.quit()
        except Exception:
            pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    step_start_time = time.time()

    try:
        for step in range(resume_step, NUM_STEPS):
            all_sequences: list[torch.Tensor] = []
            all_old_log_probs: list[torch.Tensor] = []
            all_rewards: list[torch.Tensor] = []
            all_fens: list[str] = []
            all_r_cnt: list[float] = []
            all_qualifying: list[tuple[str, str]] = []
            agg_debug: dict[str, float] = {}

            # Generate all batches first (GPU), then analyse in parallel (CPU)
            gen_batches: list[tuple[torch.Tensor, torch.Tensor, list[str]]] = []
            for accum_idx in range(ACCUM_STEPS):
                start_idx = torch.zeros((BATCH_SIZE, 1), dtype=torch.long, device=device)
                with torch.no_grad(), autocast_ctx():
                    seqs, old_lp = model.generate_with_log_probs(start_idx, max_new_tokens=83)
                fens = decode_sequences(seqs, tokenizer)
                gen_batches.append((seqs, old_lp, fens))

            # Compute entropies for all sequences at once (GPU)
            all_seqs_cat = torch.cat([b[0] for b in gen_batches], dim=0)
            with torch.no_grad():
                all_entropies: list[float] = model.compute_entropy(all_seqs_cat).tolist()

            # Parallel Stockfish analysis using thread pool with per-thread engines
            engines: list[chess.engine.SimpleEngine] = [_open_engine() for _ in range(NUM_STOCKFISH_WORKERS)]
            engine_locks = [threading.Lock() for _ in range(NUM_STOCKFISH_WORKERS)]

            try:
                for accum_idx, (seqs, old_lp, fens) in enumerate(gen_batches):
                    ent_slice = all_entropies[accum_idx * BATCH_SIZE:(accum_idx + 1) * BATCH_SIZE]
                    rews, qualifying, r_cnt, dbg = _parallel_compute_rewards(
                        fens, seqs, engines, engine_locks, replay_buffer, ent_slice,
                        tau_board=TAU_BOARD, tau_pv=TAU_PV, tau_ent=TAU_ENT,
                        tau_uni=TAU_UNI, tactical_depth=TACTICAL_DEPTH,
                    )
                    for board_str, pv in qualifying:
                        replay_buffer.add(board_str, pv)
                    all_sequences.append(seqs)
                    all_old_log_probs.append(old_lp)
                    all_rewards.append(rews.to(device))
                    all_fens.extend(fens)
                    all_r_cnt.extend(r_cnt)
                    all_qualifying.extend(qualifying)
                    for k, v in dbg.items():
                        agg_debug[k] = agg_debug.get(k, 0) + v
            finally:
                for eng in engines:
                    _close_engine(eng)

            effective_bs = BATCH_SIZE * ACCUM_STEPS
            cat_sequences = torch.cat(all_sequences, dim=0)
            cat_old_log_probs = torch.cat(all_old_log_probs, dim=0)
            cat_rewards = torch.cat(all_rewards, dim=0)

            model.eval()
            kl_chunks = []
            with torch.no_grad(), autocast_ctx():
                for c in range(0, effective_bs, BATCH_SIZE):
                    chunk = cat_sequences[c:c + BATCH_SIZE]
                    ref_c = ref_model.compute_log_probs(chunk)
                    pol_c = model.compute_log_probs(chunk)
                    kl_chunks.append((pol_c - ref_c).mean(dim=-1))
            token_kl = torch.cat(kl_chunks, dim=0)
            shaped_rewards = cat_rewards - KL_COEFF * token_kl
            advantages = (shaped_rewards - shaped_rewards.mean()) / (shaped_rewards.std() + 1e-8)

            sl_batch = sample_sl_batch(dataset, sl_indices, BATCH_SIZE, device)

            last_ppo_loss = torch.tensor(0.0)
            last_sl_loss = torch.tensor(0.0)
            last_ent_bonus = torch.tensor(0.0)
            last_loss = torch.tensor(0.0)

            for _ in range(PPO_EPOCHS):
                optimizer.zero_grad()
                ppo_loss_accum = 0.0

                model.train()
                for c in range(0, effective_bs, BATCH_SIZE):
                    chunk_seq = cat_sequences[c:c + BATCH_SIZE]
                    chunk_old_lp = cat_old_log_probs[c:c + BATCH_SIZE]
                    chunk_adv = advantages[c:c + BATCH_SIZE].detach()

                    with autocast_ctx():
                        curr_lp = model.compute_log_probs(chunk_seq)
                        token_ratio = torch.exp(curr_lp - chunk_old_lp)
                        adv_expanded = chunk_adv.unsqueeze(1)
                        surr1 = token_ratio * adv_expanded
                        surr2 = torch.clamp(token_ratio, 1 - PPO_EPS, 1 + PPO_EPS) * adv_expanded
                        chunk_ppo = -torch.min(surr1, surr2).mean() / ACCUM_STEPS

                    if scaler is not None:
                        scaler.scale(chunk_ppo).backward()
                    else:
                        chunk_ppo.backward()
                    ppo_loss_accum += chunk_ppo.detach().item()

                with autocast_ctx():
                    ent_logits = model.get_logits(cat_sequences[:BATCH_SIZE, :-1])
                    ent = -(torch.softmax(ent_logits, dim=-1) * torch.log_softmax(ent_logits, dim=-1)).sum(dim=-1)
                    ent_bonus = ent.mean()
                    ent_loss = -ENTROPY_COEFF * ent_bonus

                with autocast_ctx():
                    x_sl, y_sl = sl_batch[:, :-1], sl_batch[:, 1:]
                    _, sl_loss = model(x_sl, y_sl)
                    aux_loss = ent_loss + SL_COEFF * sl_loss

                if scaler is not None:
                    scaler.scale(aux_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    aux_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                last_ppo_loss = torch.tensor(ppo_loss_accum)
                last_sl_loss = sl_loss.detach()
                last_ent_bonus = ent_bonus.detach()
                last_loss = torch.tensor(ppo_loss_accum + SL_COEFF * sl_loss.item() - ENTROPY_COEFF * ent_bonus.item())

            if step % LOG_INTERVAL == 0:
                now = time.time()
                elapsed = now - step_start_time
                step_start_time = now

                mean_reward = cat_rewards.mean().item()
                reward_std = cat_rewards.std().item()
                n_illegal = (cat_rewards == -2).sum().item()
                n_zero = (cat_rewards == 0).sum().item()
                n_pos = (cat_rewards > 0).sum().item()
                puzzle_rate = n_pos / effective_bs
                validity = (cat_rewards > -2).sum().item() / effective_bs
                mean_kl = token_kl.mean().item()

                true_puzzle_fens = [all_fens[i] for i in range(effective_bs) if cat_rewards[i].item() >= 0.9]
                mean_r_cnt = sum(all_r_cnt) / len(all_r_cnt) if all_r_cnt else 0.0
                examples_str = ""
                if true_puzzle_fens:
                    examples_str += "\n           MULTI-MOVE PUZZLES:\n" + "\n".join(f"             {f}" for f in true_puzzle_fens[:3])

                debug_counts = {k: (v / ACCUM_STEPS if k.startswith("mean") else v) for k, v in agg_debug.items()}

                _log(
                    f"Step {step:4d} | Loss: {last_loss.item():.4f} | Reward: {mean_reward:.4f} ± {reward_std:.4f} | "
                    f"{elapsed:.1f}s/10steps\n"
                    f"           PPO: {last_ppo_loss.item():.4f} | KL: {mean_kl:.4f} | SL: {last_sl_loss.item():.4f} (coeff={SL_COEFF:.2f}) | Ent: {last_ent_bonus.item():.4f}\n"
                    f"           Rewards [-2/0/+1]: {n_illegal}/{n_zero}/{n_pos} | "
                    f"Puzzles: {puzzle_rate:.1%} | Valid: {validity:.1%} | "
                    f"Filters [valid/uniq/rev/nonobv/multi/novel]: {debug_counts.get('n_valid', 0)}/{debug_counts.get('n_unique', 0)}/{debug_counts.get('n_reversal', 0)}/{debug_counts.get('n_non_obvious', 0)}/{debug_counts.get('n_multi', 0)}/{debug_counts.get('n_novel', 0)} | "
                    f"TruePuzzles: {debug_counts.get('n_true_puzzles', 0)} | "
                    f"gap: {debug_counts.get('mean_gap_novel', 0):.4f} | balanced: {debug_counts.get('n_balanced', 0)} | "
                    f"ReplayBuf: {len(replay_buffer)}"
                    f"{examples_str}"
                )

            if (step + 1) % CHECKPOINT_INTERVAL == 0:
                ckpt_path = os.path.join(RL_CHECKPOINT_DIR, f"rl_step_{step + 1}.pt")
                torch.save(model.state_dict(), ckpt_path)
                push_checkpoint_to_hub(ckpt_path, f"rl_step_{step + 1}.pt")
                _log(f"Saved and pushed RL checkpoint at step {step + 1}")

    finally:
        os.makedirs("outputs", exist_ok=True)
        metrics_path = os.path.join("outputs", "rl_training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({}, f)
        _log("Training finished.")


if __name__ == "__main__":
    main()
