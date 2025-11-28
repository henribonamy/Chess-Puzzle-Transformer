import pathlib

from torch.utils.data import Dataset as TorchDataset


class ChessDataset(TorchDataset):
    def __init__(self, path:pathlib.Path):
        try:
            with open(path, "r") as file:
                self.fen_list = file.readlines()
        except FileNotFoundError:
            raise FileNotFoundError("This dataset expects a specific file format. The preprocessing.py file creates it for you.")

    def __len__(self):
        return len(self.fen_list)

    def __getitem__(self, index):
        return self.fen_list[index]
