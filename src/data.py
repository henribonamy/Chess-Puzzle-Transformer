import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class ChessDataset(TorchDataset):
    def __init__(self, path: pathlib.Path):
        self.data = np.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.long)
