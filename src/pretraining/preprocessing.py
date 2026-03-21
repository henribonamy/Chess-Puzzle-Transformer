import chess
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from tokenizer import FENTokenizer

tokenizer = FENTokenizer()

print("Downloading dataset from Hugging Face...")
dataset = load_dataset("Lichess/chess-puzzles")["train"]
dataset = dataset.select_columns(["FEN", "Moves"])

encoded_fens = []

print("Tokenizing all FENs...")
for i in tqdm(range(dataset.num_rows)):
    fen = dataset[i]["FEN"]
    move = dataset[i]["Moves"].split(" ")[0]
    board = chess.Board(fen)
    board.push(board.parse_uci(move))
    encoded = tokenizer.encode(board.fen())
    encoded_fens.append(encoded)

encoded_array = np.array(encoded_fens, dtype=np.int32)
np.save("data/encoded_fens.npy", encoded_array)
print("Done ! Saved encoded FENs to data/encoded_fens.npy")
