import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import learning_rate


def plot(paths, output):
    output = Path(output)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    table, losses = [], []
    for path in paths:
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text())
        rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines() if line.strip()]
        split = meta.get("evaluation_split", "test")
        steps = [row["step"] for row in rows]
        seconds = [row["train_seconds"] for row in rows]
        accuracy = [100 * row[f"{split}_accuracy"] for row in rows]
        loss = [row[f"{split}_loss"] for row in rows]
        losses.extend(loss)
        name = meta["config"]["name"]
        label = f"{name}, {split}, seed {meta['seed']}, {meta['status']}"
        axes[0, 0].plot(steps, accuracy, marker=".", label=label)
        axes[0, 1].plot(seconds, accuracy, marker=".", label=label)
        axes[1, 0].plot(steps, loss, marker=".", label=label)
        rate_steps = range(max(1, steps[-1]))
        rates = [learning_rate(meta["config"]["schedule"], step) for step in rate_steps]
        axes[1, 1].plot(rate_steps, rates, label=label)
        table.append({
            "name": name, "seed": meta["seed"], "status": meta["status"], "split": split,
            "step": steps[-1], "accuracy_pct": accuracy[-1],
            "best_accuracy_pct": max(accuracy), "train_seconds": seconds[-1],
        })
    labels = [
        ("Updates", "Accuracy (%)"),
        ("Training seconds", "Accuracy (%)"),
        ("Updates", "Loss"),
        ("Update (from zero)", "Learning rate"),
    ]
    for axis, (x, y) in zip(axes.flat, labels):
        axis.set(xlabel=x, ylabel=y)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    if min(losses) > 0 and max(losses) / min(losses) > 100:
        axes[1, 0].set_yscale("log")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "curves.png", dpi=160)
    fig.savefig(output / "curves.pdf")
    plt.close(fig)
    with (output / "runs.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=table[0])
        writer.writeheader()
        writer.writerows(table)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", default="results/comparison")
    args = parser.parse_args()
    plot(args.runs, args.output)
