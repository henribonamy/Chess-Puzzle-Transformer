import sys
import os
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "pretraining"))

from model import AutoRegressiveTransformer
from tokenizer import FENTokenizer

HF_PRETRAINED_REPO = "henribonamy/chess-puzzles-pretrained"
HF_DATA_REPO = "henribonamy/chess-puzzles-data"
PRETRAINED_CHECKPOINT_PATH = "outputs/model_checkpoint.pt"
FINETUNED_CHECKPOINT_PATH = "outputs/model_checkpoint_finetuned.pt"
DATA_PATH = "data/encoded_fens.npy"
HIGH_RATED_INDICES_PATH = "data/high_rated_indices.npy"

BATCH_SIZE = 128
LR = 1e-6
EPOCHS = 1
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 5000

log_lines: list[str] = []


class SubsetDataset(Dataset):
    """Wraps encoded_fens.npy restricted to a set of indices."""

    def __init__(self, data: np.ndarray, indices: np.ndarray) -> None:
        """Load subset of encoded FENs at the given indices."""
        self.data = data[indices]

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return encoded FEN as a tensor."""
        return torch.tensor(self.data[idx], dtype=torch.long)


def log(msg: str) -> None:
    """Append message to log buffer and print."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    log_lines.append(line)
    print(line, flush=True)


class LogHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        """Serve accumulated log lines."""
        body = "\n".join(log_lines).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        """Suppress default HTTP access logs."""
        pass


def start_log_server() -> None:
    """Start HTTP log server on port 7860 in a background thread."""
    server = HTTPServer(("0.0.0.0", 7860), LogHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log("Log server started on port 7860")


def ensure_file(local_path: str, repo_id: str, filename: str, repo_type: str) -> None:
    """Download file from HF Hub if not already present."""
    if os.path.exists(local_path):
        return
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    log(f"Downloading {filename} from {repo_id}...")
    hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=os.path.dirname(local_path))
    log(f"{filename} ready.")


def push_checkpoint(local_path: str) -> None:
    """Upload fine-tuned checkpoint to HF pretrained repo."""
    api = HfApi()
    api.create_repo(HF_PRETRAINED_REPO, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo="model_checkpoint_finetuned.pt",
        repo_id=HF_PRETRAINED_REPO,
        repo_type="model",
    )
    log(f"Pushed fine-tuned checkpoint to {HF_PRETRAINED_REPO}")


def main() -> None:
    """Fine-tune pretrained model on high-rated puzzle positions."""
    start_log_server()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    ensure_file(DATA_PATH, HF_DATA_REPO, "encoded_fens.npy", "dataset")
    ensure_file(HIGH_RATED_INDICES_PATH, HF_DATA_REPO, "high_rated_indices.npy", "dataset")
    ensure_file(PRETRAINED_CHECKPOINT_PATH, HF_PRETRAINED_REPO, "model_checkpoint.pt", "model")

    log("Loading data...")
    all_fens = np.load(DATA_PATH)
    high_rated_indices = np.load(HIGH_RATED_INDICES_PATH)
    log(f"Total encoded FENs: {len(all_fens):,} | High-rated indices: {len(high_rated_indices):,}")

    dataset = SubsetDataset(all_fens, high_rated_indices)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    log(f"Dataset size: {len(dataset):,} | Batches per epoch: {len(dataloader):,}")

    model = AutoRegressiveTransformer().to(device)
    ckpt = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt)
    log("Loaded pretrained checkpoint.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    total_steps = 0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                _, loss = model(inputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item()
            total_steps += 1

            if total_steps % LOG_INTERVAL == 0:
                elapsed = time.time() - t0
                avg_loss = running_loss / LOG_INTERVAL
                pct = 100.0 * batch_idx / len(dataloader)
                log(f"Epoch {epoch+1} | Step {total_steps} | {pct:.1f}% | Loss: {avg_loss:.4f} | {elapsed:.1f}s/{LOG_INTERVAL}steps")
                running_loss = 0.0
                t0 = time.time()

            if total_steps % CHECKPOINT_INTERVAL == 0:
                os.makedirs("outputs", exist_ok=True)
                torch.save(model.state_dict(), FINETUNED_CHECKPOINT_PATH)
                push_checkpoint(FINETUNED_CHECKPOINT_PATH)
                log(f"Checkpoint saved and pushed at step {total_steps}")

        log(f"Epoch {epoch+1} complete.")

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), FINETUNED_CHECKPOINT_PATH)
    push_checkpoint(FINETUNED_CHECKPOINT_PATH)
    log("Fine-tuning complete.")


if __name__ == "__main__":
    main()
