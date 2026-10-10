import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torchvision.datasets import CIFAR10


SETTINGS = {
    "depth": 56,
    "steps": 1500,
    "batch_size": 1000,
    "microbatch_size": 125,
    "eval_batch_size": 250,
    "eval_every": 100,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "validation_size": 5000,
    "split_seed": 1729,
}

METHODS = {
    "step": {
        "name": "matched-step-4000",
        "steps": 4000,
        "validation_size": 0,
        "eval_batch_size": 125,
        "bn_fraction": 0.80,
        "schedule": {
            "kind": "multistep",
            "base_lr": 0.1,
            "milestones": [2000, 3000],
            "gamma": 0.1,
        },
    },
    "paper": {
        "name": "table6-super-2000",
        "steps": 2000,
        "validation_size": 0,
        "eval_batch_size": 125,
        "bn_fraction": 0.80,
        "schedule": {
            "kind": "triangular",
            "base_lr": 0.1,
            "max_lr": 3.0,
            "half_cycle": 1000,
        },
    },
    "baseline": {
        "name": "validation-baseline-warmup20",
        "bn_fraction": 0.999,
        "schedule": {
            "kind": "multistep",
            "base_lr": 0.35,
            "milestones": [938, 1313],
            "gamma": 0.1,
            "warmup_steps": 20,
            "warmup_start_lr": 0.1,
        },
    },
    "super": {
        "name": "validation-super",
        "bn_fraction": 0.95,
        "schedule": {
            "kind": "triangular",
            "base_lr": 0.1,
            "max_lr": 3.0,
            "half_cycle": 750,
        },
    },
}


class CifarImages(Dataset):
    def __init__(self, data, targets, mean_image):
        self.data = data
        self.targets = targets
        self.mean = mean_image

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, key):
        index, flip = key if isinstance(key, tuple) else (key, False)
        image = torch.from_numpy(self.data[index].copy()).permute(2, 0, 1).float()
        image = image - self.mean
        if flip:
            image = image.flip(-1)
        return image, int(self.targets[index])

def stratified_split(targets, validation_size, seed):
    targets = np.asarray(targets)
    classes = np.unique(targets)
    per_class = validation_size // len(classes)
    generator = np.random.default_rng(seed)
    train_indices, validation_indices = [], []
    for label in classes:
        indices = generator.permutation(np.flatnonzero(targets == label))
        validation_indices.extend(indices[:per_class])
        train_indices.extend(indices[per_class:])
    return np.sort(train_indices), np.sort(validation_indices)

def load_cifar(root, validation_size=5000, split_seed=1729, download=True):
    train = CIFAR10(root, train=True, download=download)
    targets = np.asarray(train.targets)
    if validation_size:
        train_indices, val_indices = stratified_split(targets, validation_size, split_seed)
        train_data, train_targets = train.data[train_indices], targets[train_indices]
        eval_data, eval_targets = train.data[val_indices], targets[val_indices]
    else:
        test = CIFAR10(root, train=False, download=download)
        train_data, train_targets = train.data, targets
        eval_data, eval_targets = test.data, np.asarray(test.targets)
    mean = train_data.mean(axis=0, dtype=np.float64).astype(np.float32)
    mean = torch.from_numpy(mean).permute(2, 0, 1).contiguous()
    return CifarImages(train_data, train_targets, mean), CifarImages(eval_data, eval_targets, mean), mean


def make_batches(size, batch_size, microbatch_size, steps, seed):
    previous_epoch = -1
    for step in range(steps):
        epoch, batch = divmod(step, size // batch_size)
        if epoch != previous_epoch:
            generator = torch.Generator().manual_seed(seed + epoch)
            permutation = torch.randperm(size, generator=generator).tolist()
            previous_epoch = epoch
        indices = permutation[batch * batch_size:(batch + 1) * batch_size]
        generator = torch.Generator().manual_seed(seed + 1_000_003 + step)
        flips = torch.rand(batch_size, generator=generator) < 0.5
        samples = list(zip(indices, flips.tolist()))
        yield from BatchSampler(samples, microbatch_size, drop_last=False)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, downsample: bool = False):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(channels, channels, 3, 2 if downsample else 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.bn2(self.conv2(residual))
        shortcut = F.avg_pool2d(x, 3, 2, ceil_mode=True) if self.downsample else x
        out = F.relu(residual + shortcut)
        if self.downsample:
            out = torch.cat((out, torch.zeros_like(out)), dim=1)
        return out

class CifarResNet(nn.Module):
    def __init__(self, depth: int = 56):
        super().__init__()
        blocks = (depth - 2) // 6
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU())
        layers = [ResidualBlock(16) for _ in range(blocks)]
        for channels in (16, 32):
            layers.append(ResidualBlock(channels, downsample=True))
            layers.extend(ResidualBlock(channels * 2) for _ in range(blocks - 1))
        self.blocks = nn.Sequential(*layers)
        self.classifier = nn.Linear(64, 10)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(layer.bias)

        nn.init.uniform_(self.classifier.weight, -math.sqrt(3 / 64), math.sqrt(3 / 64))
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.classifier(F.avg_pool2d(x, 8).flatten(1))

    def prepare_batchnorm(self, fraction: float, step: int, update: bool):
        momentum = 0.0
        if update:
            if fraction == 0:
                momentum = 1.0
            else:
                mass = -math.expm1((step + 1) * math.log(fraction))
                momentum = (1 - fraction) / mass
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.momentum = momentum

    def parameter_groups(self, weight_decay):
        groups = []
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                bias_decay = 0.0 if isinstance(layer, nn.Conv2d) else weight_decay
                groups.append({"params": [layer.weight], "lr_mult": 1.0, "weight_decay": weight_decay})
                groups.append({"params": [layer.bias], "lr_mult": 2.0, "weight_decay": bias_decay})
            elif isinstance(layer, nn.BatchNorm2d):
                groups.append({"params": [layer.weight, layer.bias], "lr_mult": 1.0, "weight_decay": weight_decay})
        return groups


class CaffeSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.9):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=0.0, lr_mult=1.0))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            rate = group["lr"] * group["lr_mult"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if group["weight_decay"]:
                    grad = grad.add(param, alpha=group["weight_decay"])
                state = self.state[param]
                if "velocity" not in state:
                    state["velocity"] = torch.zeros_like(param)
                state["velocity"].mul_(group["momentum"]).add_(grad, alpha=rate)
                param.sub_(state["velocity"])

def learning_rate(schedule: dict, step: int) -> float:
    base_lr = schedule["base_lr"]
    warmup_steps = schedule.get("warmup_steps", 0)
    if step < warmup_steps:
        start_lr = schedule["warmup_start_lr"]
        return start_lr + (base_lr - start_lr) * step / warmup_steps
    if schedule["kind"] == "multistep":
        drops = sum(step >= milestone for milestone in schedule["milestones"])
        return base_lr * schedule["gamma"] ** drops
    phase = step / schedule["half_cycle"]
    triangle = max(0.0, 1.0 - abs(phase - 1.0))
    return base_lr + (schedule["max_lr"] - base_lr) * triangle


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


@torch.inference_mode()
def evaluate(model, loader, device, split="test"):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        count += len(y)
    return {f"{split}_loss": loss_sum / count, f"{split}_accuracy": correct / count, f"{split}_examples": count}


def train_step(model, optimizer, batches, config, step, device):
    lr = learning_rate(config["schedule"], step)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)

    accumulation = config["batch_size"] // config["microbatch_size"]
    loss_sum, correct, examples = 0.0, 0, 0
    for microbatch in range(accumulation):
        images, labels = next(batches)
        images, labels = images.to(device), labels.to(device)
        model.prepare_batchnorm(config["bn_fraction"], step, microbatch == 0)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        value = loss.item()
        (loss / accumulation).backward()
        loss_sum += value * len(labels)
        correct += (logits.detach().argmax(1) == labels).sum().item()
        examples += len(labels)
    optimizer.step()
    return lr, loss_sum, correct, examples


def log(output, row):
    with (output / "metrics.jsonl").open("w" if row["step"] == 0 else "a") as file:
        file.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def train(args, config=None):
    config = {**SETTINGS, **METHODS[args.method]} if config is None else config
    stop = config["steps"] if args.stop_after is None else args.stop_after
    device = torch.device(args.device if args.device != "auto" else torch.accelerator.current_accelerator(check_available=True) or "cpu")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    output = Path(args.output)
    run_info = {"config": config, "seed": args.seed, "deterministic": args.deterministic}
    validation_size = config.get("validation_size", 0)
    split = "validation" if validation_size else "test"
    train_set, eval_set, mean = load_cifar(
        args.data, validation_size, config.get("split_seed", 1729), download=not args.no_download,
    )
    model = CifarResNet(config["depth"]).to(device)
    optimizer = CaffeSGD(model.parameter_groups(config["weight_decay"]), momentum=config["momentum"])
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "mean_image.npy", mean.numpy())
    metadata = {
        **run_info, "device": str(device), "torch": str(torch.__version__),
        "train_examples": len(train_set), f"{split}_examples": len(eval_set),
        "evaluation_split": split, "status": "running",
    }
    write_json(output / "metadata.json", metadata)

    sampler = make_batches(len(train_set), config["batch_size"], config["microbatch_size"], stop, args.seed)
    loader_options = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    batches = iter(DataLoader(train_set, batch_sampler=sampler, **loader_options))
    eval_loader = DataLoader(eval_set, batch_size=config["eval_batch_size"], **loader_options)
    loss_sum, correct, count = 0.0, 0, 0
    seconds = 0.0
    metrics = evaluate(model, eval_loader, device, split)
    log(output, {"step": 0, "train_seconds": 0.0, "lr": learning_rate(config["schedule"], 0), **metrics})

    for step in range(stop):
        model.train()
        if device.type != "cpu":
            torch.accelerator.synchronize(device)
        tick = time.perf_counter()
        lr, batch_loss, batch_correct, batch_count = train_step(model, optimizer, batches, config, step, device)
        loss_sum += batch_loss
        correct += batch_correct
        count += batch_count
        if device.type != "cpu":
            torch.accelerator.synchronize(device)
        seconds += time.perf_counter() - tick
        completed = step + 1
        if completed % config["eval_every"] == 0 or completed == stop:
            metrics = evaluate(model, eval_loader, device, split)
            log(output, {
                "step": completed, "examples_seen": completed * config["batch_size"],
                "lr": lr, "train_loss": loss_sum / count, "train_accuracy": correct / count,
                "train_interval_examples": count, "train_seconds": seconds, **metrics,
            })
            loss_sum, correct, count = 0.0, 0, 0

    torch.save(model.state_dict(), output / "model.pt")
    metadata["status"] = "completed" if stop == config["steps"] else "partial"
    write_json(output / "metadata.json", metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, default="paper")
    parser.add_argument("--output")
    parser.add_argument("--data", default="data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    args.output = args.output or str(Path("runs") / args.method)
    train(args)


if __name__ == "__main__":
    main()
