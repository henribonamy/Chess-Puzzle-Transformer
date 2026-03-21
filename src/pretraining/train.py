import torch
from torch.utils.data import DataLoader
import json
import os
import time
from tqdm import tqdm
import chess

from data import ChessDataset
from model import AutoRegressiveTransformer
from tokenizer import FENTokenizer


def extract_board_position(fen_str):
    parts = fen_str.split(" ")
    return " ".join(parts[:4])


def build_board_position_hash_set(dataset, tokenizer):
    board_positions = set()
    total = len(dataset)

    for i in tqdm(range(total), desc="Building board position hash set"):
        encoded_fen = dataset.data[i]
        fen_str = tokenizer.decode(encoded_fen.tolist())
        board_position = extract_board_position(fen_str)
        board_positions.add(board_position)

    print(f"Built hash set with {len(board_positions)} unique positions")
    return board_positions


def is_valid_fen(fen_str):
    """Check if a FEN string represents a valid chess position."""
    try:
        chess.Board(fen_str)
        return True
    except:
        return False


def calculate_novelty(generated_fens, board_position_set):
    novel_count = sum(1 for fen in generated_fens
                     if extract_board_position(fen) not in board_position_set)
    return (novel_count / len(generated_fens)) * 100 if generated_fens else 0


EPOCH = 1
BATCH_SIZE = 64

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device {device}")

dataset = ChessDataset("data/encoded_fens.npy")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

tokenizer = FENTokenizer()
board_position_set = build_board_position_hash_set(dataset, tokenizer)
os.makedirs("outputs", exist_ok=True)
loss_history = []
validity_history = []
uniqueness_history = []
novelty_history = []

model = AutoRegressiveTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5625e-6, weight_decay=1.5625e-6)


iteration = 0
start_time = time.time()

for epoch in range(EPOCH):
    for batch in dataloader:
        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (iteration % 10) == 0:
            loss_val = loss.item()
            loss_history.append((iteration, loss_val))

            model.eval()
            with torch.no_grad():
                start_idx = torch.zeros((100, 1), dtype=torch.long, device=device)
                generated = model.generate(start_idx, max_new_tokens=83, temperature=1.0)

                generated_fens = []
                for i in range(generated.size(0)):
                    token_ids = generated[i].cpu().tolist()
                    try:
                        fen_str = tokenizer.decode(token_ids)
                        generated_fens.append(fen_str)
                    except:
                        continue

                # Filter for valid FENs only
                valid_fens = [fen for fen in generated_fens if is_valid_fen(fen)]

                validity = (len(valid_fens) / 100) * 100
                uniqueness = (len(set(valid_fens)) / len(valid_fens)) * 100 if valid_fens else 0
                novelty = calculate_novelty(valid_fens, board_position_set)
                print(list(set(generated_fens))[:3])

                validity_history.append((iteration, validity))
                uniqueness_history.append((iteration, uniqueness))
                novelty_history.append((iteration, novelty))

            model.train()

            time_taken = time.time() - start_time

            print(f"Iteration {iteration}, Loss: {loss_val:.4f}, "
                  f"Validity: {validity:.1f}%, Uniqueness: {uniqueness:.1f}%, "
                  f"Novelty: {novelty:.1f}%, Time: {time_taken:.2f}s")

            start_time = time.time()

        iteration += 1

print("Saving training metrics...")
metrics_data = {
    "loss": loss_history,
    "validity": validity_history,
    "uniqueness": uniqueness_history,
    "novelty": novelty_history,
}

output_file = os.path.join("outputs", "training_metrics.json")
with open(output_file, "w") as f:
    json.dump(metrics_data, f, indent=4)

print(f"Saved training metrics to {output_file}")

