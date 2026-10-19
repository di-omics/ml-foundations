# Lab notebook

Entries are in order. Every prediction is written down before the run it predicts.

## 2026-10-18: Quickstart baseline

**Setup.** PyTorch Quickstart network (784 -> 512 -> 512 -> 10, ReLU), plain SGD,
learning rate 1e-3, batch size 64, 5 epochs, seed 0, CPU. FashionMNIST split into
50,000 train / 10,000 validation (both from the official training set) / 10,000 test
(the official test set, not used yet).

**Command.** `uv run train.py --name baseline`

**Result** (`runs/baseline.json`, 11 s):

| epoch | train loss | val loss | val accuracy |
|------:|-----------:|---------:|-------------:|
| 1 | 2.251 | 2.191 | 46.0% |
| 2 | 2.106 | 2.006 | 58.1% |
| 3 | 1.862 | 1.708 | 60.3% |
| 4 | 1.551 | 1.413 | 62.0% |
| 5 | 1.300 | 1.207 | 63.8% |

Running the same command a second time gave identical numbers to every decimal place.

**My observations:**

### Exercise 1: one weight update (`exercise1_weight_update.py`)

- [ ] The script prints PASS.

**Q1.** Before any training, the loss on the first batch is 2.31. Note that ln(10) = 2.30.
What does that say about how the untrained model is guessing?

A:

**Q2.** How much did the weight change in that one step, compared with the size of the
weight itself? What does that suggest about why the baseline was still improving when
training stopped?

A:

**Q3.** If the learning rate were 100 times larger, how big would that one step be?

A:

**Something to notice.** In the table above, train loss is higher than validation loss in
every epoch. That is unusual, because a model normally does better on the data it learns
from. Why might it happen here? (Hint: when during the epoch is each number measured?)

A:

## Run 2: change only the learning rate

Fill in the prediction before running anything.

- Learning rate tried:
- Everything else unchanged (seed, split, epochs, batch size): yes
- Prediction (final validation accuracy, shape of the curves, and why):
- Command: `uv run train.py --name lr-<value> --lr <value>`, then `uv run plot.py`
- Result:
- Interpretation:
- What remains uncertain:
