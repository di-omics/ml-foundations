"""Plot every run in runs/ on the same axes and save runs/compare.png.

Usage:
    uv run plot.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

runs = [json.loads(p.read_text()) for p in sorted(Path("runs").glob("*.json"))]
if not runs:
    raise SystemExit("No runs yet. Train one first: uv run train.py --name baseline")

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
for run in runs:
    epochs = [h["epoch"] for h in run["history"]]
    label = f"{run['name']} (lr={run['config']['lr']:g})"
    (line,) = ax_loss.plot(epochs, [h["train_loss"] for h in run["history"]], marker="o", label=f"{label} train")
    ax_loss.plot(
        epochs, [h["val_loss"] for h in run["history"]],
        marker="o", linestyle="--", color=line.get_color(), label=f"{label} validation",
    )
    ax_acc.plot(epochs, [100 * h["val_accuracy"] for h in run["history"]], marker="o", label=label)

ax_loss.set(title="Loss (lower is better)", xlabel="epoch", ylabel="cross-entropy loss")
ax_acc.set(title="Validation accuracy (higher is better)", xlabel="epoch", ylabel="accuracy (%)")
for ax in (ax_loss, ax_acc):
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
fig.tight_layout()
fig.savefig("runs/compare.png", dpi=150)
print("Saved runs/compare.png")

print(f"\n{'run':<20} {'lr':>8} {'train loss':>11} {'val loss':>9} {'val acc':>8}")
for run in runs:
    last = run["history"][-1]
    print(
        f"{run['name']:<20} {run['config']['lr']:>8g} {last['train_loss']:>11.4f} "
        f"{last['val_loss']:>9.4f} {100 * last['val_accuracy']:>7.1f}%"
    )
