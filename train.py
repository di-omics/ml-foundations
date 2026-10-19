"""Train a small image classifier on FashionMNIST.

This follows the official PyTorch Quickstart
(https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
almost line for line, so you can read the two side by side. The additions are
command-line flags for the things you will change in experiments, a fixed random
seed, a validation split, and a JSON record of every run in runs/.

Usage:
    uv run train.py --name baseline
    uv run train.py --name lr-0.1 --lr 0.1
    uv run train.py --name final --lr <chosen> --test   # only once, at the very end
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import NeuralNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="run name; results go to runs/<name>.json")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (Quickstart default 1e-3)")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
# CPU keeps runs bit-for-bit reproducible, so the only difference between two
# runs is the thing you changed. This model is small enough that CPU is fast.
parser.add_argument("--device", default="cpu", help="cpu or mps (Apple GPU)")
parser.add_argument("--test", action="store_true", help="also score the held-out test set")
args = parser.parse_args()

torch.manual_seed(args.seed)
device = args.device

# --- Working with data -------------------------------------------------------
# 60,000 training and 10,000 test images, 28x28 grayscale, 10 clothing classes.
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

# Split the 60,000 training images into 50,000 to learn from and 10,000 to
# compare runs on (validation). The split seed is fixed, so every run is scored
# on the same 10,000 images. The test set stays untouched until the end: if you
# choose a learning rate by looking at test scores, the test score is no longer
# an honest estimate of how the model does on new data.
train_subset, val_subset = random_split(
    training_data, [50_000, 10_000], generator=torch.Generator().manual_seed(0)
)

train_dataloader = DataLoader(train_subset, batch_size=args.batch_size)
val_dataloader = DataLoader(val_subset, batch_size=args.batch_size)
test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

# --- Creating the model (see model.py) ---------------------------------------
model = NeuralNetwork().to(device)
print(model)

# --- Optimizing the model parameters -----------------------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass: how wrong is the model on this batch?
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward pass: compute gradients, nudge every weight by lr * gradient,
        # then clear the gradients so they do not pile up into the next batch.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        if batch % 100 == 0:
            current = (batch + 1) * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
    return total_loss / len(dataloader)


def evaluate(dataloader, model, loss_fn, label):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    avg_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    avg_loss /= num_batches
    correct /= size
    print(f"{label}: accuracy {(100 * correct):>0.1f}%, avg loss {avg_loss:>8f}\n")
    return avg_loss, correct


history = []
start = time.time()
for t in range(args.epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    val_loss, val_acc = evaluate(val_dataloader, model, loss_fn, "Validation")
    history.append(
        {"epoch": t + 1, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc}
    )
seconds = round(time.time() - start, 1)

test_result = None
if args.test:
    test_loss, test_acc = evaluate(test_dataloader, model, loss_fn, "TEST")
    test_result = {"test_loss": test_loss, "test_accuracy": test_acc}
print("Done!")

# --- Record the run ----------------------------------------------------------
record = {
    "name": args.name,
    "config": {
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": device,
        "optimizer": "SGD",
        "torch": torch.__version__,
    },
    "seconds": seconds,
    "history": history,
    "test": test_result,
}
Path("runs").mkdir(exist_ok=True)
out = Path("runs") / f"{args.name}.json"
out.write_text(json.dumps(record, indent=2) + "\n")
print(f"Saved {out}")
