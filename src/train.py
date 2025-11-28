import torch
from torch.utils.data import DataLoader

from data import ChessDataset
from model import AutoRegressiveTransformer

EPOCH = 3
BATCH_SIZE = 32

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dataset = ChessDataset("data/encoded_fens.npy")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = AutoRegressiveTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

iteration = 0

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
            print(f"Iteration {iteration}, Loss: {loss.item():.4f}")

        iteration += 1
