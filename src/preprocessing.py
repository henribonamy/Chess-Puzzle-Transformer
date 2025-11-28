import chess
from datasets import load_dataset
from tqdm import tqdm
from tokenizer import tokenize


dataset = load_dataset("Lichess/chess-puzzles")["train"]
dataset = dataset.select_columns(["FEN", "Moves"])

fens = []

# takes about 10min
for i in tqdm(range(dataset.num_rows)):
    fen = dataset[i]["FEN"]
    move = dataset[i]["Moves"].split(" ")[0]
    board = chess.Board(fen)
    board.push(board.parse_uci(move))
    processed_fen = tokenize(board.fen())
    fens.append(processed_fen)

with open("data/fen_strings.txt", "w") as file:
    file.writelines(fens)
