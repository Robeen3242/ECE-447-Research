# Similar paramets matching using different group equivariant architectures (Z2, P4, P4M) on a custom Rotated-MNIST dataset with fixed random rotations per image. Each model is trained for multiple seeds and results are saved and summarized in JSON files. The code uses the escnn library for group-equivariant CNNs.

import argparse
import copy
import json
import os
import random
from dataclasses import asdict, dataclass

from pathlib import Path
import csv
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import escnn.nn as enn
import escnn.gspaces as gspaces

# python mnist.py --model all 
@dataclass
class Config:
    seeds: tuple = (42, 999)
    split_seed: int = 42
    rotation_seed: int = 123

    data_root: str = "./data"
    out_dir: str = "./results/rotated_mnist_method_compare"

    epochs: int = 50
    batch_size: int = 128
    num_workers: int = 2

    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_lr_scheduler: bool = True
    scheduler_type: str = "step"   # options: "none", "step", "cosine"
    step_size: int = 25
    gamma: float = 0.1
    eta_min: float = 1e-5          # only used for cosine

    train_size: int = 10000
    val_size: int = 2000
    test_size: int = 50000
    num_classes: int = 10

    deterministic: bool = False
    save_checkpoints: bool = True

    train_rotation_range: tuple = (-180.0, 180.0)
    val_rotation_range: tuple = (-180.0, 180.0)
    test_rotation_range: tuple = (-180.0, 180.0)

    z2_channels = (64, 64, 64, 64)
    p4_widths   = (32, 32, 32, 32)
    p4m_widths  = (24, 24, 24, 24)


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



class FixedRotatedMNIST(Dataset):
    def __init__(self, base_dataset, indices, angles):
        self.base = base_dataset
        self.indices = indices
        self.angles = angles

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = int(self.indices[idx])
        image, label = self.base[base_idx]

        image = TF.rotate(
            image,
            angle=float(self.angles[idx]),
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        x = TF.to_tensor(image)
        x = TF.normalize(x, mean=[0.1307], std=[0.3081])
        return x, int(label)

def build_rotated_mnist_splits(cfg: Config):
    train_base = torchvision.datasets.MNIST(
        root=cfg.data_root,
        train=True,
        download=True,
        transform=None,
    )
    test_base = torchvision.datasets.MNIST(
        root=cfg.data_root,
        train=False,
        download=True,
        transform=None,
    )

    g = torch.Generator().manual_seed(cfg.split_seed)
    perm = torch.randperm(len(train_base), generator=g).tolist()

    train_idx = np.array(perm[: cfg.train_size], dtype=np.int64)
    val_idx = np.array(perm[cfg.train_size : cfg.train_size + cfg.val_size], dtype=np.int64)
    test_idx = np.arange(len(test_base), dtype=np.int64)

    rng_train = np.random.default_rng(cfg.rotation_seed)
    rng_val = np.random.default_rng(cfg.rotation_seed + 1000)
    rng_test = np.random.default_rng(cfg.rotation_seed + 2000)

    train_angles = rng_train.uniform(cfg.train_rotation_range[0], cfg.train_rotation_range[1], size=len(train_idx))
    val_angles = rng_val.uniform(cfg.val_rotation_range[0], cfg.val_rotation_range[1], size=len(val_idx))
    test_angles = rng_test.uniform(cfg.test_rotation_range[0], cfg.test_rotation_range[1], size=len(test_idx))

    trainset = FixedRotatedMNIST(train_base, train_idx, train_angles)
    valset = FixedRotatedMNIST(train_base, val_idx, val_angles)
    testset = FixedRotatedMNIST(test_base, test_idx, test_angles)

    return trainset, valset, testset

def make_loader(dataset, batch_size, shuffle, num_workers, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=generator if shuffle else None,
    )

# 4 layers
class Z2MNISTCNN(nn.Module):
    def __init__(self, channels=(64, 64, 64, 64), num_classes=10):
        super().__init__()
        c1, c2, c3, c4 = channels

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),

            nn.Conv2d(c1, c2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),

            nn.Conv2d(c3, c4, kernel_size=4, padding=0, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Linear(c4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))   # global average pooling
        return self.head(x)

# Generic group-equivariant CNN for both P4 and P4M, with configurable widths and group space
class GroupMNISTCNN(nn.Module):
    def __init__(self, gspace, widths=(32, 32, 32, 32), num_classes=10):
        super().__init__()
        self.r2_act = gspace
        self.in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        def feat_type(c):
            return enn.FieldType(self.r2_act, c * [self.r2_act.regular_repr])

        f1 = feat_type(widths[0])
        f2 = feat_type(widths[1])
        f3 = feat_type(widths[2])
        f4 = feat_type(widths[3])

        self.block1 = enn.SequentialModule(
            enn.R2Conv(self.in_type, f1, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(f1),
            enn.ReLU(f1, inplace=True),
            enn.PointwiseMaxPool(f1, kernel_size=2, stride=2),
        )
        self.block2 = enn.SequentialModule(
            enn.R2Conv(f1, f2, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(f2),
            enn.ReLU(f2, inplace=True),
            enn.PointwiseMaxPool(f2, kernel_size=2, stride=2),
        )
        self.block3 = enn.SequentialModule(
            enn.R2Conv(f2, f3, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(f3),
            enn.ReLU(f3, inplace=True),
        )
        self.block4 = enn.SequentialModule(
            enn.R2Conv(f3, f4, kernel_size=4, padding=0, bias=False),
            enn.InnerBatchNorm(f4),
            enn.ReLU(f4, inplace=True),
        )

        # Global pooling over the group dimension, leaving only the spatial dimensions
        self.gpool = enn.GroupPooling(f4)
        self.head = nn.Linear(widths[3], num_classes)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gpool(x)
        x = x.tensor
        x = x.mean(dim=(2, 3))
        return self.head(x)


class P4MNISTCNN(GroupMNISTCNN):
    def __init__(self, widths=(32, 32, 32, 32), num_classes=10):
        super().__init__(
            gspace=gspaces.rot2dOnR2(N=4), #N=4 for 90-degree rotations, 8 for 45-degree rotations
            widths=widths,
            num_classes=num_classes,
        )


class P4MMNISTCNN(GroupMNISTCNN):
    def __init__(self, widths=(24, 24, 24, 24), num_classes=10):
        super().__init__(
            gspace=gspaces.flipRot2dOnR2(N=4), #N=4 for 90-degree rotations + reflections, 8 for 45-degree rotations + reflections  
            widths=widths,
            num_classes=num_classes,
        )

# Helper functions for training and evaluation
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Evaluate the model on a given DataLoader and return average loss and accuracy.
def evaluate(model, loader, criterion, return_predictions=False):
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if return_predictions:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)

    if return_predictions:
        return avg_loss, acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)
    return avg_loss, acc

# Train the model for one seed and return the results and metrics.
def train_one_seed(model_name, model_builder, cfg: Config, seed: int, trainset, valset, testset):
    set_seed(seed, deterministic=cfg.deterministic)

    train_loader = make_loader(trainset, cfg.batch_size, True, cfg.num_workers, seed)
    val_loader = make_loader(valset, cfg.batch_size, False, cfg.num_workers, seed)
    test_loader = make_loader(testset, cfg.batch_size, False, cfg.num_workers, seed)

    model = model_builder().to(DEVICE)

    criterion_train = nn.CrossEntropyLoss()
    criterion_eval = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    scheduler = None
    if cfg.use_lr_scheduler:
        if cfg.scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=cfg.step_size,
                gamma=cfg.gamma
            )
        elif cfg.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.epochs,
                eta_min=cfg.eta_min
            )
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = -1.0
    best_state = None

    run_dir = os.path.join(cfg.out_dir, model_name, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n=== {model_name.upper()} | Seed {seed} ===", flush=True)
    print(f"Params: {count_parameters(model):,}", flush=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        total_train = 0
        correct_train = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion_train(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / max(total_train, 1)
        train_acc = 100.0 * correct_train / max(total_train, 1)

        val_loss, val_acc = evaluate(model, val_loader, criterion_eval)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(current_lr))
        if scheduler is not None:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            if cfg.save_checkpoints:
                torch.save(best_state, os.path.join(run_dir, "best_model.pth"))

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            print(
                f"Epoch [{epoch:03d}/{cfg.epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:6.2f}% "
                f"val_loss={val_loss:.4f} val_acc={val_acc:6.2f}%",
                flush=True,
            )

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(
        model, val_loader, criterion_eval, return_predictions=True
    )
    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion_eval, return_predictions=True
    )

    result = {
        "model": model_name,
        "seed": seed,
        "params": int(count_parameters(model)),
        "best_val_accuracy_during_training": float(best_val_acc),
        "final_best_checkpoint_val_loss": float(val_loss),
        "final_best_checkpoint_val_accuracy": float(val_acc),
        "final_best_checkpoint_test_loss": float(test_loss),
        "final_best_checkpoint_test_accuracy": float(test_acc),
        "history_file": os.path.join(run_dir, "train_history.json"),
        "checkpoint_file": os.path.join(run_dir, "best_model.pth") if cfg.save_checkpoints else None,
        "predictions_file": os.path.join(run_dir, "predictions.npz"),
    }

    np.savez(
        os.path.join(run_dir, "predictions.npz"),
        val_preds=val_preds,
        val_labels=val_labels,
        val_probs=val_probs,
        test_preds=test_preds,
        test_labels=test_labels,
        test_probs=test_probs,
    )

    with open(os.path.join(run_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(run_dir, "metrics_summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"Best-checkpoint eval | val_acc={val_acc:.2f}% test_acc={test_acc:.2f}%",
        flush=True,
    )

    return result


def summarize_model(cfg: Config, model_name: str, results):
    test_accs = [r["final_best_checkpoint_test_accuracy"] for r in results]
    val_accs = [r["final_best_checkpoint_val_accuracy"] for r in results]

    summary = {
        "config": asdict(cfg),
        "device": str(DEVICE),
        "model": model_name,
        "num_runs": len(results),
        "results_per_seed": results,
        "aggregate": {
            "val_accuracy_mean": float(np.mean(val_accs)),
            "val_accuracy_std": float(np.std(val_accs)),
            "test_accuracy_mean": float(np.mean(test_accs)),
            "test_accuracy_std": float(np.std(test_accs)),
            "test_error_mean": float(100.0 - np.mean(test_accs)),
            "test_error_std": float(np.std([100.0 - x for x in test_accs])),
        },
    }

    model_dir = os.path.join(cfg.out_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "summary_all_seeds.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(model_dir, "summary_all_seeds.txt"), "w") as f:
        f.write(f"=== {model_name.upper()} Rotated-MNIST Summary ===\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Runs: {len(results)}\n")
        f.write(f"Params: {results[0]['params']:,}\n")
        f.write(
            f"Val acc mean/std: {summary['aggregate']['val_accuracy_mean']:.2f}% / "
            f"{summary['aggregate']['val_accuracy_std']:.2f}%\n"
        )
        f.write(
            f"Test acc mean/std: {summary['aggregate']['test_accuracy_mean']:.2f}% / "
            f"{summary['aggregate']['test_accuracy_std']:.2f}%\n"
        )

    return summary

def save_comparison(cfg: Config, summaries: dict):
    comparison = {name: summary["aggregate"] for name, summary in summaries.items()}

    if "z2" in summaries and "p4" in summaries:
        comparison["delta_test_accuracy_mean_p4_minus_z2"] = (
            summaries["p4"]["aggregate"]["test_accuracy_mean"]
            - summaries["z2"]["aggregate"]["test_accuracy_mean"]
        )

    if "z2" in summaries and "p4m" in summaries:
        comparison["delta_test_accuracy_mean_p4m_minus_z2"] = (
            summaries["p4m"]["aggregate"]["test_accuracy_mean"]
            - summaries["z2"]["aggregate"]["test_accuracy_mean"]
        )

    if "p4" in summaries and "p4m" in summaries:
        comparison["delta_test_accuracy_mean_p4m_minus_p4"] = (
            summaries["p4m"]["aggregate"]["test_accuracy_mean"]
            - summaries["p4"]["aggregate"]["test_accuracy_mean"]
        )

    with open(os.path.join(cfg.out_dir, "comparison_summary.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    with open(os.path.join(cfg.out_dir, "comparison_summary.md"), "w") as f:
        f.write("# Rotated-MNIST Comparison\n\n")
        for name, summary in summaries.items():
            agg = summary["aggregate"]
            f.write(
                f"- {name.upper()} test accuracy mean/std: "
                f"{agg['test_accuracy_mean']:.2f}% +- {agg['test_accuracy_std']:.2f}%\n"
            )

def _seed_dir(cfg: Config, model_name: str, seed: int) -> Path:
    return Path(cfg.out_dir) / model_name / f"seed_{seed}"


def _load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _load_history(cfg: Config, model_name: str, seed: int):
    return _load_json(_seed_dir(cfg, model_name, seed) / "train_history.json")


def _load_summary(cfg: Config, model_name: str):
    return _load_json(Path(cfg.out_dir) / model_name / "summary_all_seeds.json")


def _load_predictions(cfg: Config, model_name: str, seed: int):
    return np.load(_seed_dir(cfg, model_name, seed) / "predictions.npz")


def _mean_history_across_seeds(cfg: Config, model_name: str):
    histories = []
    for seed in cfg.seeds:
        history_path = _seed_dir(cfg, model_name, seed) / "train_history.json"
        if history_path.exists():
            histories.append(_load_history(cfg, model_name, seed))

    if not histories:
        return None

    mean_history = {}
    keys = ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]
    for key in keys:
        arr = np.array([h[key] for h in histories], dtype=float)
        mean_history[key] = arr.mean(axis=0)
        mean_history[key + "_std"] = arr.std(axis=0)
    return mean_history


def save_report_table(cfg: Config, models, report_dir: Path):
    rows = []
    for model_name in models:
        summary = _load_summary(cfg, model_name)
        agg = summary["aggregate"]
        params = summary["results_per_seed"][0]["params"]

        rows.append({
            "model": model_name.upper(),
            "params": params,
            "val_acc_mean": agg["val_accuracy_mean"],
            "val_acc_std": agg["val_accuracy_std"],
            "test_acc_mean": agg["test_accuracy_mean"],
            "test_acc_std": agg["test_accuracy_std"],
            "test_error_mean": agg["test_error_mean"],
            "test_error_std": agg["test_error_std"],
        })

    csv_path = report_dir / "report_metrics_table.csv"
    md_path = report_dir / "report_metrics_table.md"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w") as f:
        f.write("| Model | Params | Val Acc Mean | Val Acc Std | Test Acc Mean | Test Acc Std | Test Error Mean | Test Error Std |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['model']} | {r['params']:,} | "
                f"{r['val_acc_mean']:.2f} | {r['val_acc_std']:.2f} | "
                f"{r['test_acc_mean']:.2f} | {r['test_acc_std']:.2f} | "
                f"{r['test_error_mean']:.2f} | {r['test_error_std']:.2f} |\n"
            )


def plot_accuracy_curves(cfg: Config, models, report_dir: Path):
    plt.figure(figsize=(10, 6))

    for model_name in models:
        hist = _mean_history_across_seeds(cfg, model_name)
        if hist is None:
            continue

        epochs = np.arange(1, len(hist["train_acc"]) + 1)
        plt.plot(epochs, hist["train_acc"], linestyle="--", label=f"{model_name.upper()} train")
        plt.plot(epochs, hist["val_acc"], label=f"{model_name.upper()} val")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir / "accuracy_vs_epoch.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_loss_curves(cfg: Config, models, report_dir: Path):
    plt.figure(figsize=(10, 6))

    for model_name in models:
        hist = _mean_history_across_seeds(cfg, model_name)
        if hist is None:
            continue

        epochs = np.arange(1, len(hist["train_loss"]) + 1)
        plt.plot(epochs, hist["train_loss"], linestyle="--", label=f"{model_name.upper()} train")
        plt.plot(epochs, hist["val_loss"], label=f"{model_name.upper()} val")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir / "loss_vs_epoch.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_test_accuracy_bar(cfg: Config, models, report_dir: Path):
    names = []
    means = []
    stds = []
    params = []

    for model_name in models:
        summary = _load_summary(cfg, model_name)
        agg = summary["aggregate"]
        names.append(model_name.upper())
        means.append(agg["test_accuracy_mean"])
        stds.append(agg["test_accuracy_std"])
        params.append(summary["results_per_seed"][0]["params"])

    x = np.arange(len(names))
    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, names)
    plt.ylabel("Test Accuracy (%)")
    plt.title("Final Test Accuracy by Model")

    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{means[i]:.2f}%\n{params[i]:,} params",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(report_dir / "test_accuracy_bar.png", dpi=200, bbox_inches="tight")
    plt.close()


def _confusion_matrix_from_preds(labels, preds, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for y_true, y_pred in zip(labels, preds):
        cm[int(y_true), int(y_pred)] += 1
    return cm


def plot_confusion_matrices(cfg: Config, models, report_dir: Path, seed=None):
    if seed is None:
        seed = cfg.seeds[0]

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        pred_file = _load_predictions(cfg, model_name, seed)
        labels = pred_file["test_labels"]
        preds = pred_file["test_preds"]

        cm = _confusion_matrix_from_preds(labels, preds, num_classes=cfg.num_classes)
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(model_name.upper())
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(cfg.num_classes))
        ax.set_yticks(range(cfg.num_classes))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_example_improvements(cfg: Config, report_dir: Path, seed=None, num_examples=12):
    required = {"z2", "p4", "p4m"}
    available = {d.name for d in Path(cfg.out_dir).iterdir() if d.is_dir()}
    if not required.issubset(available):
        return

    if seed is None:
        seed = cfg.seeds[0]

    z2_pred = _load_predictions(cfg, "z2", seed)
    p4_pred = _load_predictions(cfg, "p4", seed)
    p4m_pred = _load_predictions(cfg, "p4m", seed)

    test_labels = z2_pred["test_labels"]
    testset = build_rotated_mnist_splits(cfg)[2]

    candidates = []
    for i, y in enumerate(test_labels):
        z2_ok = int(z2_pred["test_preds"][i]) == int(y)
        p4_ok = int(p4_pred["test_preds"][i]) == int(y)
        p4m_ok = int(p4m_pred["test_preds"][i]) == int(y)

        if (not z2_ok) and (p4_ok or p4m_ok):
            score = max(
                float(p4_pred["test_probs"][i, int(y)]),
                float(p4m_pred["test_probs"][i, int(y)])
            ) - float(z2_pred["test_probs"][i, int(y)])
            candidates.append((score, i))

    if not candidates:
        return

    candidates.sort(reverse=True)
    chosen = [idx for _, idx in candidates[:num_examples]]

    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, chosen):
        x, y = testset[idx]
        img = (x.squeeze().numpy() * 0.3081 + 0.1307).clip(0, 1)
        angle = float(testset.angles[idx])

        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(
            f"gt={int(y)}, ang={angle:.1f}\n"
            f"z2={int(z2_pred['test_preds'][idx])}, "
            f"p4={int(p4_pred['test_preds'][idx])}, "
            f"p4m={int(p4m_pred['test_preds'][idx])}",
            fontsize=9
        )

    for ax in axes[len(chosen):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(report_dir / "example_improvements.png", dpi=200, bbox_inches="tight")
    plt.close()


def generate_report_artifacts(cfg: Config, models):
    report_dir = Path(cfg.out_dir) / "report_artifacts"
    report_dir.mkdir(parents=True, exist_ok=True)

    save_report_table(cfg, models, report_dir)
    plot_accuracy_curves(cfg, models, report_dir)
    plot_loss_curves(cfg, models, report_dir)
    plot_test_accuracy_bar(cfg, models, report_dir)
    plot_confusion_matrices(cfg, models, report_dir)

    if set(models) >= {"z2", "p4", "p4m"}:
        plot_example_improvements(cfg, report_dir)

    print(f"\nSaved report artifacts to: {report_dir}")

def run_for_model(cfg: Config, model_name: str):
    if model_name == "z2":
        builder = lambda: Z2MNISTCNN(
            channels=cfg.z2_channels,
            num_classes=cfg.num_classes
        )
    elif model_name == "p4":
        builder = lambda: P4MNISTCNN(
            widths=cfg.p4_widths,
            num_classes=cfg.num_classes
        )
    elif model_name == "p4m":
        builder = lambda: P4MMNISTCNN(
            widths=cfg.p4m_widths,
            num_classes=cfg.num_classes
        )
    else:
        raise ValueError("model_name must be one of: z2, p4, p4m")

    # trainset, valset, testset = build_prerotated_mnist_splits(cfg.prerotated_npz_path)
    trainset, valset, testset = build_rotated_mnist_splits(cfg)
    results = []
    for seed in cfg.seeds:
        results.append(train_one_seed(model_name, builder, cfg, seed, trainset, valset, testset))

    return summarize_model(cfg, model_name, results)


def main():
    parser = argparse.ArgumentParser(
        description="Z2 / P4 / P4M comparison on custom Rotated-MNIST"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="all",
        choices=["z2", "p4", "p4m", "all"],
        help="Run one model or all models"
    )
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--out-dir", type=str, default=CFG.out_dir)
    parser.add_argument("--data-root", type=str, default=CFG.data_root)

    args, _ = parser.parse_known_args()

    cfg = copy.deepcopy(CFG)
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.out_dir = args.out_dir
    cfg.data_root = args.data_root

    os.makedirs(cfg.out_dir, exist_ok=True)

    print("=" * 80)
    print("Custom Rotated-MNIST Comparison")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Output: {cfg.out_dir}")
    print(f"Seeds: {cfg.seeds}")

    requested = ["z2", "p4", "p4m"] if args.model == "all" else [args.model]

    summaries = {}
    for model_name in requested:
        summaries[model_name] = run_for_model(cfg, model_name)

    if len(summaries) >= 2:
        save_comparison(cfg, summaries)
        print("\nSaved cross-model comparison summary.")
    generate_report_artifacts(cfg, requested)

if __name__ == "__main__":
    main()
