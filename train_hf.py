import sys
import os
import json
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
import chess
import chess.engine
import trackio
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "pretraining"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "rl_training"))

from data import ChessDataset
from model import AutoRegressiveTransformer
from tokenizer import FENTokenizer
from rewards import compute_binary_rewards, extract_board_position
from replay_buffer import ReplayBuffer

HF_PRETRAINED_REPO = "henribonamy/chess-puzzles-pretrained"
HF_RL_REPO = "henribonamy/chess-puzzles-rl"
HF_DATA_REPO = "henribonamy/chess-puzzles-data"
HF_TRACKIO_SPACE = "henribonamy/chess-puzzles-trackio"
TRACKIO_PROJECT = "chess-puzzles-rl"

BATCH_SIZE = 64
LR = 1e-6
PPO_EPOCHS = 4
PPO_EPS = 0.2
KL_COEFF = 0.3
SL_COEFF = 0.1
SL_COEFF_WARMUP = 1.0
SL_WARMUP_STEPS = 200
NUM_STEPS = 1000
LOG_INTERVAL = 10
CHECKPOINT_INTERVAL = 100
DATA_MIX_SIZE = 100_000
REPLAY_BUFFER_SIZE = 10_000
BUFFER_SEED_SIZE = 10_000
TAU_BOARD = 5
TAU_PV = 3
TAU_ENT = 0.6
TAU_UNI = 0.1
TAU_CNT = -0.1
TACTICAL_DEPTH = 6
PRETRAINED_CHECKPOINT_PATH = "outputs/model_checkpoint_1000_iterations_128_bs.pt"
RL_CHECKPOINT_DIR = "outputs/rl_checkpoints"
DATA_PATH = "data/encoded_fens.npy"
HIGH_RATED_INDICES_PATH = "data/high_rated_indices.npy"


def ensure_high_rated_indices() -> None:
    """Download high-rated puzzle indices from HF Hub if not present locally."""
    if os.path.exists(HIGH_RATED_INDICES_PATH):
        return
    os.makedirs("data", exist_ok=True)
    print(f"Downloading high_rated_indices.npy from {HF_DATA_REPO}...")
    hf_hub_download(
        repo_id=HF_DATA_REPO,
        filename="high_rated_indices.npy",
        repo_type="dataset",
        local_dir="data",
    )
    print("High-rated indices downloaded.")


def ensure_data() -> None:
    """Download encoded FENs from HF Hub, or run preprocessing as fallback."""
    if os.path.exists(DATA_PATH):
        return
    os.makedirs("data", exist_ok=True)
    try:
        print(f"Downloading encoded FENs from {HF_DATA_REPO}...")
        hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename="encoded_fens.npy",
            repo_type="dataset",
            local_dir="data",
        )
        os.rename("data/encoded_fens.npy", DATA_PATH) if not os.path.exists(DATA_PATH) else None
        print("Data downloaded.")
    except Exception as e:
        print(f"Download failed ({e}), running preprocessing instead...")
        env = {**os.environ, "PYTHONPATH": os.path.join("src", "pretraining")}
        subprocess.run(
            [sys.executable, os.path.join("src", "pretraining", "preprocessing.py")],
            env=env,
            check=True,
        )


def ensure_pretrained_checkpoint() -> None:
    """Download the pretrained checkpoint from HF Hub if not present locally."""
    if os.path.exists(PRETRAINED_CHECKPOINT_PATH):
        return
    print(f"Checkpoint not found locally — downloading from {HF_PRETRAINED_REPO}...")
    os.makedirs("outputs", exist_ok=True)
    hf_hub_download(
        repo_id=HF_PRETRAINED_REPO,
        filename="model_checkpoint_finetuned.pt",
        local_dir="outputs",
    )
    os.rename("outputs/model_checkpoint_finetuned.pt", PRETRAINED_CHECKPOINT_PATH)


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


def main() -> None:
    """Run PPO RL training with trackio logging and HF Hub checkpoint pushing."""
    _start_health_server()
    ensure_data()
    ensure_high_rated_indices()
    ensure_pretrained_checkpoint()

    trackio.init(TRACKIO_PROJECT, space_id=HF_TRACKIO_SPACE)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    step_start_time = time.time()

    try:
        for step in range(NUM_STEPS):
            start_idx = torch.zeros((BATCH_SIZE, 1), dtype=torch.long, device=device)

            # generate_with_log_probs runs in eval mode and returns log probs at sampling time
            with torch.no_grad(), autocast_ctx():
                sequences, old_log_probs = model.generate_with_log_probs(start_idx, max_new_tokens=83)

            fens = decode_sequences(sequences, tokenizer)
            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            try:
                tau_cnt_current = min(TAU_CNT + step * 0.0015, 0.0)
                rewards, qualifying, r_cnt_scores, debug_counts = compute_binary_rewards(
                    fens, sequences, engine, replay_buffer, model,
                    tau_board=TAU_BOARD, tau_pv=TAU_PV, tau_ent=TAU_ENT,
                    tau_uni=TAU_UNI, tau_cnt=tau_cnt_current,
                    tactical_depth=TACTICAL_DEPTH,
                )
            finally:
                try:
                    engine.quit()
                except Exception as e:
                    _log(f"WARNING: engine.quit() failed at step {step}: {e}")
            for board_str, pv in qualifying:
                replay_buffer.add(board_str, pv)

            rewards = rewards.to(device)

            # KL reward shaping (detached — shapes reward signal, not gradient)
            model.eval()
            with torch.no_grad(), autocast_ctx():
                ref_log_probs = ref_model.compute_log_probs(sequences)
                policy_log_probs = model.compute_log_probs(sequences)
            token_kl = (policy_log_probs - ref_log_probs).mean(dim=-1)
            shaped_rewards = rewards - KL_COEFF * token_kl

            # Within-batch advantage normalization
            advantages = (shaped_rewards - shaped_rewards.mean()) / (shaped_rewards.std() + 1e-8)

            sl_batch = sample_sl_batch(dataset, sl_indices, BATCH_SIZE, device)

            last_ppo_loss = torch.tensor(0.0)
            last_sl_loss = torch.tensor(0.0)
            last_loss = torch.tensor(0.0)

            for _ in range(PPO_EPOCHS):
                model.eval()  # match eval mode used by generate_with_log_probs

                with autocast_ctx():
                    curr_log_probs = model.compute_log_probs(sequences)
                    log_ratio = (curr_log_probs - old_log_probs).mean(dim=-1)
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * advantages.detach()
                    surr2 = torch.clamp(ratio, 1 - PPO_EPS, 1 + PPO_EPS) * advantages.detach()
                    ppo_loss = -torch.min(surr1, surr2).mean()

                    model.train()
                    x_sl, y_sl = sl_batch[:, :-1], sl_batch[:, 1:]
                    _, sl_loss = model(x_sl, y_sl)

                    sl_coeff = SL_COEFF_WARMUP + (SL_COEFF - SL_COEFF_WARMUP) * min(step / SL_WARMUP_STEPS, 1.0)
                    loss = ppo_loss + sl_coeff * sl_loss
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

                last_ppo_loss = ppo_loss.detach()
                last_sl_loss = sl_loss.detach()
                last_loss = loss.detach()

            if step % LOG_INTERVAL == 0:
                now = time.time()
                elapsed = now - step_start_time
                step_start_time = now

                mean_reward = rewards.mean().item()
                reward_std = rewards.std().item()
                n_illegal = (rewards == -2).sum().item()
                n_zero = (rewards == 0).sum().item()
                n_pos = (rewards > 0).sum().item()
                puzzle_rate = n_pos / BATCH_SIZE
                validity = (rewards > -2).sum().item() / BATCH_SIZE
                mean_kl = token_kl.mean().item()

                puzzle_fens = [fens[i] for i in range(BATCH_SIZE) if rewards[i].item() > 0.0]
                mean_r_cnt = sum(r_cnt_scores) / len(r_cnt_scores) if r_cnt_scores else 0.0
                examples_str = ""
                if puzzle_fens:
                    examples_str = "\n           Puzzles:\n" + "\n".join(f"             {f}" for f in puzzle_fens[:3])

                trackio.log({
                    "loss": last_loss.item(),
                    "ppo_loss": last_ppo_loss.item(),
                    "kl": mean_kl,
                    "sl_loss": last_sl_loss.item(),
                    "reward_mean": mean_reward,
                    "reward_std": reward_std,
                    "puzzle_rate": puzzle_rate * 100,
                    "validity": validity * 100,
                    "n_qualifying": n_pos,
                    "n_illegal": n_illegal,
                    "mean_r_cnt": mean_r_cnt,
                    "r_cnt_uniq": debug_counts["mean_r_cnt_unique"],
                    "n_unique_winning": debug_counts["n_unique_winning"],
                    "mean_gap": debug_counts["mean_gap_novel"],
                    "n_balanced": debug_counts["n_balanced"],
                    "replay_buffer_size": len(replay_buffer),
                    "step": step,
                })

                _log(
                    f"Step {step:4d} | Loss: {last_loss.item():.4f} | Reward: {mean_reward:.4f} ± {reward_std:.4f} | "
                    f"{elapsed:.1f}s/10steps\n"
                    f"           PPO: {last_ppo_loss.item():.4f} | KL: {mean_kl:.4f} | SL: {last_sl_loss.item():.4f} (coeff={sl_coeff:.2f})\n"
                    f"           Rewards [-2/0/+1]: {n_illegal}/{n_zero}/{n_pos} | "
                    f"Puzzles: {puzzle_rate:.1%} | Valid: {validity:.1%} | "
                    f"Filters [valid/unique/counter/novel]: {debug_counts['n_valid']}/{debug_counts['n_unique']}/{debug_counts['n_counter']}/{debug_counts['n_novel']} | "
                    f"UniqueWin: {debug_counts['n_unique_winning']} | r_cnt_uniq: {debug_counts['mean_r_cnt_unique']:.4f} | "
                    f"r_cnt: {mean_r_cnt:.4f} | gap: {debug_counts['mean_gap_novel']:.4f} | "
                    f"balanced: {debug_counts['n_balanced']} | tau_cnt: {tau_cnt_current:.3f} | ReplayBuf: {len(replay_buffer)}"
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
