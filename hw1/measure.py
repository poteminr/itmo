import argparse
import csv
import hashlib
import json
import datetime as dt
import multiprocessing as mp
from pathlib import Path
import random
import time

import numpy as np
import pynvml
import torch

from equations import flops, memory
from models import SmallCNN

FIELDS = ["S", "B", "status", "latency", "memory", "energy", "is_validation",
          "flops", "flops_reference", "memory_predicted", "energy_method"]


BASE_SIZES = (32, 64, 128, 224, 256, 384, 512)
BASE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def make_grid(seed: int = 2026) -> dict:
    rng = random.Random(seed)
    extra_sizes = sorted(rng.sample([s for s in range(32, 513, 16) if s not in BASE_SIZES], 4))
    extra_batches = sorted(rng.sample([b for b in range(1, 257) if b not in BASE_BATCHES], 3))
    sizes = sorted([*BASE_SIZES, *extra_sizes])
    batches = sorted([*BASE_BATCHES, *extra_batches])
    points = [dict(S=s, B=b, is_validation=s not in BASE_SIZES or b not in BASE_BATCHES)
              for s in sizes for b in batches]
    rng.shuffle(points)
    return dict(seed=seed, sizes=sizes, batches=batches, extra_sizes=extra_sizes,
                extra_batches=extra_batches, points=points,
                split="base Cartesian grid: calibration; any added axis value: validation")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def source_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ("models.py", "equations.py", "measure.py")}


def configure(device=0, seed=2026):
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def count_flops(model, x):
    counts = []

    def count(module, inputs, output):
        if isinstance(module, torch.nn.Conv2d):
            counts.append(2*output.numel()*module.weight[0].numel())
        elif isinstance(module, torch.nn.Linear):
            counts.append(2*output.numel()*module.in_features + output.numel())
        elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
            counts.append(inputs[0].numel())

    handles = [m.register_forward_hook(count) for m in model.modules()]
    try:
        model(x)
    finally:
        for handle in handles:
            handle.remove()
    return sum(counts)


def measure_energy(forward, gpu, seconds=3):
    before = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
    started = time.perf_counter()
    repeats = 0
    while time.perf_counter()-started < seconds or repeats < 5:
        forward()
        repeats += 1
    total = (pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)-before)/1000
    if total <= 0:
        raise RuntimeError("NVML energy counter did not advance")
    return dict(total_j=total, per_forward_j=total/repeats, repeats=repeats,
                duration_s=time.perf_counter()-started)


def measure_point(job):
    point, device = job
    configure(device)
    pynvml.nvmlInit()
    row = dict(**point, status="OK", latency="", memory="", energy="",
               flops=flops(point["S"], point["B"]),
               memory_predicted=memory(point["S"], point["B"]),
               energy_method="nvml_total_energy_counter")
    raw = dict(row=row)
    try:
        uuid = str(torch.cuda.get_device_properties(device).uuid)
        gpu = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + uuid.removeprefix("GPU-"))
        with torch.inference_mode():
            model = SmallCNN().to(device="cuda", dtype=torch.float32).eval()
            x = torch.randn(point["B"], 3, point["S"], point["S"], device="cuda", dtype=torch.float32)
            row["flops_reference"] = count_flops(model, x)
            torch.cuda.synchronize()

            def forward():
                started = time.perf_counter()
                y = model(x)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                del y
                return elapsed

            for _ in range(10):
                forward()
            torch.cuda.reset_peak_memory_stats()
            forward()
            row["memory"] = torch.cuda.max_memory_allocated()
            raw["latency_samples_s"] = [forward() for _ in range(30)]
            row["latency"] = float(np.median(raw["latency_samples_s"]))
            deadline = time.perf_counter()+3
            while time.perf_counter() < deadline:
                forward()
            raw["energy_windows"] = [measure_energy(forward, gpu) for _ in range(3)]
            row["energy"] = float(np.median([w["per_forward_j"] for w in raw["energy_windows"]]))
    except torch.cuda.OutOfMemoryError:
        row.update(status="OOM", latency="", memory="OOM", energy="")
    finally:
        pynvml.nvmlShutdown()
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results-reproduction"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    grid = make_grid()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plan_only:
        write_json(args.output/"grid.json", grid)
        print("132 points: 63 calibration, 69 validation")
        return

    properties = torch.cuda.get_device_properties(args.device)
    if properties.total_memory > 16*2**30:
        raise RuntimeError("Use a GPU with at most 16 GiB, as required by the assignment")
    meta = dict(kind="gpu_measurement", complete=False,
                created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                gpu=dict(name=properties.name, total_memory_bytes=properties.total_memory),
                torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                source_hashes=source_hashes(),
                settings=dict(device=args.device, seed=2026, warmup=10, repeats=30,
                              energy_seconds=3, energy_repeats=3),
                flags=dict(cudnn_benchmark=False, cudnn_allow_tf32=False, matmul_allow_tf32=False))
    points = grid["points"][:args.max_configs]
    with (args.output/"measurements.csv").open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        write_json(args.output/"grid.json", grid)
        write_json(args.output/"metadata.json", meta)
        (args.output/"raw").mkdir(exist_ok=True)

        with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
            jobs = [(point, args.device) for point in points]
            for i, raw in enumerate(pool.imap(measure_point, jobs), 1):
                row = raw["row"]
                write_json(args.output/"raw"/f"S{row['S']}_B{row['B']}.json", raw)
                writer.writerow(row)
                file.flush()
                print(f"{i}/{len(points)}: S={row['S']}, B={row['B']}, {row['status']}", flush=True)
    meta.update(complete=len(points) == len(grid["points"]),
                finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write_json(args.output/"metadata.json", meta)


if __name__ == "__main__":
    main()
