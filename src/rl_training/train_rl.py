import sys
import os
import json
import shutil

import torch
import chess
import chess.engine
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pretraining"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import ChessDataset
from model import AutoRegressiveTransformer
from tokenizer import FENTokenizer
from rewards import compute_binary_rewards, extract_board_position
from replay_buffer import ReplayBuffer

BATCH_SIZE = 32
LR = 1e-6
PPO_EPOCHS = 4
PPO_EPS = 0.2
KL_COEFF = 0.1
SL_COEFF = 0.1
NUM_STEPS = 1000
LOG_INTERVAL = 10
CHECKPOINT_INTERVAL = 100
DATA_MIX_SIZE = 100_000
REPLAY_BUFFER_SIZE = 10_000
TAU_BOARD = 5
TAU_PV = 3
TAU_ENT = 1.0
TACTICAL_DEPTH = 10
PRETRAINED_CHECKPOINT_PATH = "outputs/model_checkpoint_1000_iterations_128_bs.pt"
RL_CHECKPOINT_DIR = "outputs/rl_checkpoints"
DATA_PATH = "data/encoded_fens.npy"


def build_board_position_hash_set(dataset: ChessDataset, tokenizer: FENTokenizer) -> set[str]:
    """Build a set of canonical board position strings from the training dataset."""
    board_positions: set[str] = set()
    for i in tqdm(range(len(dataset)), desc="Building board position hash set"):
        fen_str = tokenizer.decode(dataset.data[i].tolist())
        board_positions.add(extract_board_position(fen_str))
    print(f"Built hash set with {len(board_positions)} unique positions")
    return board_positions


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


def main() -> None:
    """Run PPO RL training loop to improve chess puzzle generation quality."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer = FENTokenizer()
    dataset = ChessDataset(DATA_PATH)
    board_position_set = build_board_position_hash_set(dataset, tokenizer)

    sl_indices = torch.randperm(len(dataset))[:DATA_MIX_SIZE].tolist()
    print(f"Pre-sampled {len(sl_indices)} SL indices for data mixture")

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    model = AutoRegressiveTransformer().to(device)
    if os.path.exists(PRETRAINED_CHECKPOINT_PATH):
        state_dict = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained checkpoint from {PRETRAINED_CHECKPOINT_PATH}")
    else:
        print(f"Warning: no checkpoint found at {PRETRAINED_CHECKPOINT_PATH}, starting from scratch")

    ref_model = AutoRegressiveTransformer().to(device)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    os.makedirs(RL_CHECKPOINT_DIR, exist_ok=True)
    backup_path = os.path.join(RL_CHECKPOINT_DIR, "pretrained_backup.pt")
    torch.save(model.state_dict(), backup_path)
    print(f"Saved pretrained model backup to {backup_path}")

    stockfish_path = shutil.which("stockfish") or "/usr/games/stockfish"
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    metrics: dict[str, list] = {
        "rewards": [],
        "loss": [],
        "ppo_loss": [],
        "kl_penalty": [],
        "sl_loss": [],
        "validity": [],
        "tactical_rate": [],
        "replay_buffer_size": [],
    }

    try:
        for step in range(NUM_STEPS):
            start_idx = torch.zeros((BATCH_SIZE, 1), dtype=torch.long, device=device)

            model.eval()
            with torch.no_grad():
                sequences = model.generate(start_idx, max_new_tokens=83)
                model.eval()  # generate() calls self.train() internally; re-enter eval
                old_seq_log_probs = model.compute_sequence_log_prob(sequences)

            fens = decode_sequences(sequences, tokenizer)
            rewards, qualifying = compute_binary_rewards(
                fens, sequences, engine, board_position_set, replay_buffer, model,
                tau_board=TAU_BOARD, tau_pv=TAU_PV, tau_ent=TAU_ENT,
                tactical_depth=TACTICAL_DEPTH,
            )
            for board_str, pv in qualifying:
                replay_buffer.add(board_str, pv)

            rewards = rewards.to(device)
            advantages = rewards - rewards.mean()

            sl_batch = sample_sl_batch(dataset, sl_indices, BATCH_SIZE, device)

            last_ppo_loss = torch.tensor(0.0)
            last_kl = torch.tensor(0.0)
            last_sl_loss = torch.tensor(0.0)
            last_loss = torch.tensor(0.0)

            for _ in range(PPO_EPOCHS):
                model.eval()

                new_log_probs = model.compute_log_probs(sequences)
                new_seq_log_probs = new_log_probs.sum(dim=1)
                ratio = torch.exp(new_seq_log_probs - old_seq_log_probs.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - PPO_EPS, 1 + PPO_EPS) * advantages
                ppo_loss = -torch.min(surr1, surr2).mean()

                with torch.no_grad():
                    ref_log_probs = ref_model.compute_log_probs(sequences)
                kl_penalty = (new_log_probs - ref_log_probs).mean()

                model.train()
                x_sl, y_sl = sl_batch[:, :-1], sl_batch[:, 1:]
                _, sl_loss = model(x_sl, y_sl)

                loss = ppo_loss + KL_COEFF * kl_penalty + SL_COEFF * sl_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                last_ppo_loss = ppo_loss.detach()
                last_kl = kl_penalty.detach()
                last_sl_loss = sl_loss.detach()
                last_loss = loss.detach()

            if step % LOG_INTERVAL == 0:
                mean_reward = rewards.mean().item()
                reward_std = rewards.std().item()
                n_illegal = (rewards == -2).sum().item()
                n_zero = (rewards == 0).sum().item()
                n_pos = (rewards == 1).sum().item()
                tactical_rate = n_pos / BATCH_SIZE
                validity = (rewards > -2).sum().item() / BATCH_SIZE

                metrics["rewards"].append((step, mean_reward))
                metrics["loss"].append((step, last_loss.item()))
                metrics["ppo_loss"].append((step, last_ppo_loss.item()))
                metrics["kl_penalty"].append((step, last_kl.item()))
                metrics["sl_loss"].append((step, last_sl_loss.item()))
                metrics["validity"].append((step, validity))
                metrics["tactical_rate"].append((step, tactical_rate))
                metrics["replay_buffer_size"].append((step, len(replay_buffer)))

                qualifying_fens = [fens[i] for i in range(BATCH_SIZE) if rewards[i].item() == 1.0]
                sample_fens = qualifying_fens[:3] or [f for f in fens if f][:3]
                samples_str = "\n".join(f"             {f}" for f in sample_fens)

                print(
                    f"Step {step:4d} | Loss: {last_loss.item():.4f} | Reward: {mean_reward:.4f} ± {reward_std:.4f}\n"
                    f"           PPO: {last_ppo_loss.item():.4f} | KL: {last_kl.item():.4f} | SL: {last_sl_loss.item():.4f}\n"
                    f"           Rewards [-2/0/+1]: {n_illegal}/{n_zero}/{n_pos} | "
                    f"TacticalRate: {tactical_rate:.2%} | ReplayBuf: {len(replay_buffer)}\n"
                    f"           Samples ({'qualifying' if qualifying_fens else 'any valid'}):\n{samples_str}"
                )

            if (step + 1) % CHECKPOINT_INTERVAL == 0:
                ckpt_path = os.path.join(RL_CHECKPOINT_DIR, f"rl_step_{step + 1}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved RL checkpoint to {ckpt_path}")

    finally:
        engine.quit()
        os.makedirs("outputs", exist_ok=True)
        metrics_path = os.path.join("outputs", "rl_training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
