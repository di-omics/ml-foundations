"""Exercise 1: do one weight update by hand.

In train.py, optimizer.step() updates all 669,706 weights at once, 3,910 times
per 5-epoch run. Here you follow a single weight through the very first of
those steps and predict its new value yourself.

Run it:
    uv run exercise1_weight_update.py
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import NeuralNetwork

lr = 1e-3

# Same starting point as train.py: same seed, same split, same first batch.
torch.manual_seed(0)
model = NeuralNetwork()
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
train_subset, _ = random_split(
    training_data, [50_000, 10_000], generator=torch.Generator().manual_seed(0)
)
X, y = next(iter(DataLoader(train_subset, batch_size=64)))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# The weight we follow: in the first layer, the connection from one pixel in
# the center of the image (row 14, column 14) into hidden unit 1.
weights = model.linear_relu_stack[0].weight  # shape [512 hidden units, 784 pixels]
unit, pixel = 1, 14 * 28 + 14

# 1. Prediction: 64 rows of 10 scores (logits), one row per image.
pred = model(X)
# 2. Loss: one number for how wrong those scores are, averaged over the batch.
loss = loss_fn(pred, y)
# 3. Gradients: for every weight, how fast the loss rises if that weight rises.
loss.backward()

before = weights[unit, pixel].item()
grad = weights.grad[unit, pixel].item()
print(f"loss on this batch:  {loss.item():.4f}")
print(f"weight before:       {before:+.10f}")
print(f"its gradient:        {grad:+.10f}")
print(f"learning rate:       {lr}")

# ---------------------------------------------------------------------------
# YOUR TURN. Using only `before`, `grad` and `lr`, write the weight's value
# after one SGD step. It is one line. Hint: SGD moves every weight downhill,
# meaning in the direction that makes the loss smaller.
my_after = None  # TODO
# ---------------------------------------------------------------------------

# 4. Update: PyTorch takes the real step for every weight in the model.
optimizer.step()
after = weights[unit, pixel].item()

if my_after is None:
    print("\nFill in my_after, then run this again.")
else:
    print(f"\nyour prediction:     {my_after:+.10f}")
    print(f"PyTorch's result:    {after:+.10f}")
    if abs(my_after - after) < 1e-7:
        print("\nPASS: that is exactly what optimizer.step() did.")
        print("Now answer the three questions under Exercise 1 in NOTEBOOK.md.")
    else:
        print("\nNot quite. Check the sign: which direction is downhill?")
