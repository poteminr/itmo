import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, nnls

from equations import bytes_moved, energy, flops, latency, layer_costs, memory


def read_measurements(path):
    with Path(path).open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["S"], row["B"] = int(row["S"]), int(row["B"])
        row["is_validation"] = row["is_validation"].lower() == "true"
        if row["status"] == "OK":
            for key in ("latency", "memory", "energy", "flops_reference"):
                row[key] = float(row[key])
    return sorted(rows, key=lambda row: (row["S"], row["B"]))


def error_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if not len(actual):
        return dict(n=0, mape_percent=None, median_ape_percent=None, p95_ape_percent=None, rmse=None)
    relative = np.abs(predicted-actual)/actual*100
    return dict(n=len(actual), mape_percent=float(np.mean(relative)),
                median_ape_percent=float(np.median(relative)),
                p95_ape_percent=float(np.percentile(relative, 95)),
                rmse=float(np.sqrt(np.mean((predicted-actual)**2))))


def fit_models(records):
    train = [r for r in records if r["status"] == "OK" and not r["is_validation"]]
    s, b = np.array([r["S"] for r in train]), np.array([r["B"] for r in train])
    times = np.array([r["latency"] for r in train], float)
    energies = np.array([r["energy"] for r in train], float)
    costs = layer_costs(s, b)
    fs, ds = np.stack([r["flops"] for r in costs]), np.stack([r["bytes"] for r in costs])
    bounds = (np.log([1e-9, 1e6, 1e5]), np.log([10.0, 1e16, 1e14]))

    def residual(log_parameters):
        overhead, compute, bandwidth = np.exp(log_parameters)
        predicted = overhead+np.maximum(fs/compute, ds/bandwidth).sum(axis=0)
        return np.log(predicted)-np.log(times)

    initial_compute = np.median(flops(s, b)/times)
    initial_bandwidth = np.median(bytes_moved(s, b)/times)
    fits = []
    for compute_factor in (1, 4, 16):
        for bandwidth_factor in (1, 4, 16):
            initial = np.log([max(1e-8, times.min()*0.25),
                              initial_compute*compute_factor, initial_bandwidth*bandwidth_factor])
            initial = np.clip(initial, bounds[0]+1e-6, bounds[1]-1e-6)
            fits.append(least_squares(residual, initial, bounds=bounds, max_nfev=1500))
    best = min((fit for fit in fits if fit.success), key=lambda fit: fit.cost)
    theta_latency = dict(zip(("overhead_s", "compute_flops_per_s", "bandwidth_bytes_per_s"), np.exp(best.x)))

    features = np.column_stack([np.ones(len(train)), latency(s, b, theta_latency),
                                flops(s, b), bytes_moved(s, b)])
    scales = np.linalg.norm(features, axis=0)
    normalized = features/scales
    coefficients, _ = nnls(normalized/energies[:, None], np.ones(len(train)), maxiter=1000)
    coefficients /= scales
    theta_energy = dict(zip(("offset_j", "base_power_w", "joules_per_flop", "joules_per_byte"), coefficients))
    theta_energy["latency"] = theta_latency
    output = dict(latency=theta_latency, energy=theta_energy,
                  fit=dict(calibration_points=[[r["S"], r["B"]] for r in train],
                           energy_feature_rank=int(np.linalg.matrix_rank(normalized)),
                           energy_condition_number=float(np.linalg.cond(normalized))), metrics={})
    for split, validation in (("calibration", False), ("validation", True)):
        subset = [r for r in records if r["status"] == "OK" and r["is_validation"] == validation]
        ss, bb = np.array([r["S"] for r in subset]), np.array([r["B"] for r in subset])
        predictions = dict(latency=latency(ss, bb, theta_latency), memory=memory(ss, bb),
                           energy=energy(ss, bb, theta_energy), flops=flops(ss, bb))
        output["metrics"][split] = {
            key: error_metrics([r["flops_reference"] if key == "flops" else r[key] for r in subset], values)
            for key, values in predictions.items()
        }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results-reproduction"))
    args = parser.parse_args()
    rows = read_measurements(args.results/"measurements.csv")
    theta = fit_models(rows)
    theta["provenance"] = dict(
        measurements_sha256=hashlib.sha256((args.results/"measurements.csv").read_bytes()).hexdigest(),
        calibrate_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.results/"theta.json").write_text(json.dumps(theta, indent=2) + "\n")
    print(json.dumps(theta["metrics"], indent=2))
