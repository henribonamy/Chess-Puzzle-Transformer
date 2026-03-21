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
from rewards import compute_rewards, realism_reward, uniqueness_reward, diversity_reward

BATCH_SIZE = 32
LR = 1e-6
NUM_STEPS = 1000
LOG_INTERVAL = 10
CHECKPOINT_INTERVAL = 100
PRETRAINED_CHECKPOINT_PATH = "outputs/model_checkpoint_1000_iterations_128_bs.pt"
RL_CHECKPOINT_DIR = "outputs/rl_checkpoints"
DATA_PATH = "data/encoded_fens.npy"


def extract_board_position(fen: str) -> str:
    """Extract the first 4 FEN fields as a canonical board position string."""
    parts = fen.split(" ")
    return " ".join(parts[:4])


def build_board_position_hash_set(dataset: ChessDataset, tokenizer: FENTokenizer) -> set[str]:
    """Build a set of canonical board position strings from the training dataset."""
    board_positions: set[str] = set()
    for i in tqdm(range(len(dataset)), desc="Building board position hash set"):
        fen_str = tokenizer.decode(dataset.data[i].tolist())
        board_positions.add(extract_board_position(fen_str))
    print(f"Built hash set with {len(board_positions)} unique positions")
    return board_positions


def is_valid_fen(fen: str) -> bool:
    """Return True if the FEN string represents a legal chess position."""
    try:
        chess.Board(fen)
        return True
    except Exception:
        return False


def decode_sequences(sequences: torch.Tensor, tokenizer: FENTokenizer) -> list[str]:
    """Decode a batch of token id tensors to FEN strings, returning empty string on failure."""
    fens: list[str] = []
    for i in range(sequences.size(0)):
        try:
            fens.append(tokenizer.decode(sequences[i].cpu().tolist()))
        except Exception:
            fens.append("")
    return fens


def main() -> None:
    """Run REINFORCE RL training loop to improve chess puzzle generation quality."""
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

    model = AutoRegressiveTransformer().to(device)
    if os.path.exists(PRETRAINED_CHECKPOINT_PATH):
        state_dict = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained checkpoint from {PRETRAINED_CHECKPOINT_PATH}")
    else:
        print(f"Warning: no checkpoint found at {PRETRAINED_CHECKPOINT_PATH}, starting from scratch")

    os.makedirs(RL_CHECKPOINT_DIR, exist_ok=True)
    backup_path = os.path.join(RL_CHECKPOINT_DIR, "pretrained_backup.pt")
    torch.save(model.state_dict(), backup_path)
    print(f"Saved pretrained model backup to {backup_path}")

    stockfish_path = shutil.which("stockfish") or "/usr/games/stockfish"
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    metrics: dict[str, list] = {"rewards": [], "loss": [], "validity": [], "realism": [], "uniqueness": [], "diversity": []}

    try:
        for step in range(NUM_STEPS):
            start_idx = torch.zeros((BATCH_SIZE, 1), dtype=torch.long, device=device)

            with torch.no_grad():
                sequences = model.generate(start_idx, max_new_tokens=83)

            fens = decode_sequences(sequences, tokenizer)
            r_realism = realism_reward(fens, engine)
            r_uniqueness = uniqueness_reward(fens, board_position_set)
            r_diversity = diversity_reward(fens)
            rewards = (0.4 * r_realism + 0.3 * r_uniqueness + 0.3 * r_diversity).to(device)

            model.train()
            log_probs = model.compute_log_probs(sequences)
            seq_log_probs = log_probs.sum(dim=1)

            advantages = rewards - rewards.mean()
            loss = -(seq_log_probs * advantages).mean()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()
            optimizer.zero_grad()

            if step % LOG_INTERVAL == 0:
                mean_reward = rewards.mean().item()
                reward_std = rewards.std().item()
                loss_val = loss.item()
                validity = sum(1 for f in fens if is_valid_fen(f)) / BATCH_SIZE
                mean_realism = r_realism.mean().item()
                mean_uniqueness = r_uniqueness.mean().item()
                mean_diversity = r_diversity.mean().item()
                sample_fen = next((f for f in fens if is_valid_fen(f)), fens[0] if fens else "")

                metrics["rewards"].append((step, mean_reward))
                metrics["loss"].append((step, loss_val))
                metrics["validity"].append((step, validity))
                metrics["realism"].append((step, mean_realism))
                metrics["uniqueness"].append((step, mean_uniqueness))
                metrics["diversity"].append((step, mean_diversity))

                print(
                    f"Step {step:4d} | Loss: {loss_val:.4f} | Reward: {mean_reward:.4f} ± {reward_std:.4f} | "
                    f"Validity: {validity:.2%} | GradNorm: {grad_norm:.4f}\n"
                    f"           Realism: {mean_realism:.4f} | Uniqueness: {mean_uniqueness:.4f} | Diversity: {mean_diversity:.4f}\n"
                    f"           Sample: {sample_fen}"
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
