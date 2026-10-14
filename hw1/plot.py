import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibrate import read_measurements
from equations import bytes_moved, energy, flops, latency, layer_costs, memory

METRICS = {
    "flops": ("Logical FLOPs", "GFLOPs", 1e9),
    "memory": ("Peak allocated memory", "MiB", 2**20),
    "latency": ("Forward latency", "ms", 1e-3),
    "energy": ("Whole-GPU energy / forward", "mJ", 1e-3),
}


def predict(key, s, b, theta):
    if key == "latency":
        return latency(s, b, theta["latency"])
    if key == "energy":
        return energy(s, b, theta["energy"])
    return {"flops": flops, "memory": memory}[key](s, b)


def actual(row, key):
    return float(row["flops_reference"] if key == "flops" else row[key])


def make_plots(rows, theta, output, label="Real GPU measurements", vram_bytes=None):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    sizes = sorted({r["S"] for r in rows})
    curve_batches = np.unique(np.rint(np.geomspace(1, 256, 160)).astype(int))
    ok = [r for r in rows if r["status"] == "OK"]
    has_oom = any(r["status"] == "OOM" for r in rows)
    for key, (title, unit, scale) in METRICS.items():
        fig, axes = plt.subplots(3, 4, figsize=(14, 9), squeeze=False, layout="constrained")
        for axis, s in zip(axes.flat, sizes):
            subset = [r for r in ok if r["S"] == s]
            axis.plot(curve_batches, np.asarray(predict(key, s, curve_batches, theta))/scale,
                      color="#183b56", label="Prediction")
            for validation, color, marker in [(False, "#007f72", "o"), (True, "#c96816", "^")]:
                points = [r for r in subset if r["is_validation"] == validation]
                axis.scatter([r["B"] for r in points], [actual(r, key)/scale for r in points],
                             s=24, color=color, marker=marker,
                             label="Held-out observations" if validation else "Calibration observations", zorder=3)
            if key == "memory":
                oom = [r["B"] for r in rows if r["S"] == s and r["status"] == "OOM"]
                if has_oom:
                    axis.scatter(oom, [0.025]*len(oom), transform=axis.get_xaxis_transform(),
                                 color="#b52525", marker="x", label="OOM (no measured peak)", zorder=4)
                if vram_bytes:
                    axis.axhline(vram_bytes/scale, ls=":", color="#888888", lw=0.8, label="Total VRAM")
            axis.set(xscale="log", yscale="log", title=f"S = {s}", xlabel="Batch B", ylabel=unit)
            axis.grid(alpha=0.15, which="both")
        for axis in list(axes.flat)[len(sizes):]:
            axis.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
        detail = label+"; FLOPs reference: runtime layer shapes." if key == "flops" else label
        fig.suptitle(f"{title}: predicted curves and observations\n{detail}")
        fig.savefig(output/f"{key}_grid.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")
    for axis, (key, (title, unit, scale)) in zip(axes.flat, METRICS.items()):
        a = np.array([actual(r, key) for r in ok])/scale
        p = np.asarray(predict(key, np.array([r["S"] for r in ok]), np.array([r["B"] for r in ok]), theta))/scale
        for validation, color in [(False, "#007f72"), (True, "#c96816")]:
            mask = np.array([r["is_validation"] == validation for r in ok])
            axis.scatter(a[mask], p[mask], s=20, c=color, label="Held out" if validation else "Calibration")
        low, high = min(a.min(), p.min())*0.8, max(a.max(), p.max())*1.2
        axis.plot([low, high], [low, high], "--", color="#555555", lw=1, label="Exact agreement")
        axis.set(xscale="log", yscale="log", xlabel=f"Observed ({unit})", ylabel=f"Predicted ({unit})", title=title)
        axis.grid(alpha=0.15)
        axis.legend(frameon=False)
    fig.suptitle(label+" — parity (FLOPs: logical shape reference)")
    fig.savefig(output/"parity.png", dpi=160)
    plt.close(fig)

    ss, bb = np.array([r["S"] for r in ok]), np.array([r["B"] for r in ok])
    costs = layer_costs(ss, bb)
    cp = np.stack([c["flops"]/theta["latency"]["compute_flops_per_s"] for c in costs])
    mp = np.stack([c["bytes"]/theta["latency"]["bandwidth_bytes_per_s"] for c in costs])
    components = np.stack([np.full(len(ok), theta["latency"]["overhead_s"]),
                           np.where(mp > cp, mp, 0).sum(axis=0),
                           np.where(cp >= mp, cp, 0).sum(axis=0)])
    regimes = np.argmax(components, axis=0)
    intensity = flops(ss, bb)/bytes_moved(ss, bb)
    observed = flops(ss, bb)/np.array([r["latency"] for r in ok])/1e12
    predicted = flops(ss, bb)/latency(ss, bb, theta["latency"])/1e12
    fig, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    for index, (name, color) in enumerate(zip(("launch", "memory", "compute"), ("#7953a1", "#007f72", "#c96816"))):
        mask = regimes == index
        axis.scatter(intensity[mask], observed[mask], s=28, color=color,
                     label=f"Observed, model: {name} (n={int(mask.sum())})")
        axis.scatter(intensity[mask], predicted[mask], s=24, edgecolor=color, facecolor="none", marker="s",
                     label=f"Predicted, model: {name}")
    axis.set(xscale="log", yscale="log", xlabel="Nominal arithmetic intensity (FLOPs / byte)",
             ylabel="Effective throughput (TFLOPs/s)", title=label+" — bottlenecks assigned by the fitted model")
    axis.grid(alpha=0.15)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(output/"regimes.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results-reproduction"))
    parser.add_argument("--vram-gib", type=float)
    args = parser.parse_args()
    rows = read_measurements(args.results/"measurements.csv")
    theta = json.loads((args.results/"theta.json").read_text())
    vram = args.vram_gib * 2**30 if args.vram_gib else None
    if (args.results/"metadata.json").exists():
        meta = json.loads((args.results/"metadata.json").read_text())
        vram = meta["gpu"]["total_memory_bytes"]
    make_plots(rows, theta, args.results/"figures", vram_bytes=vram)
    predictions = [{"S": row["S"], "B": row["B"], "status": row["status"],
                    "is_validation": row["is_validation"],
                    **{key+"_predicted": predict(key, row["S"], row["B"], theta) for key in METRICS}}
                   for row in rows]
    with (args.results/"predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print("Saved 6 figures and predictions.csv")
