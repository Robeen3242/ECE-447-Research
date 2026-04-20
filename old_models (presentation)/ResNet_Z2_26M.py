# =========================
# Colab-ready Z2 ResNet44 on CIFAR-10, 3 seeds
# Paper-style setup:
# - 40k train / 10k val / 10k test
# - ResNet44 => 6n+2 with n=7
# - widths = 32, 64, 128
# - SGD lr=0.05 momentum=0.9 weight_decay=5e-4
# - LR milestones at 50, 100, 150
# - 300 epochs
# - augmentation: random crop + horizontal flip
# =========================

import os
import json
import math
import copy
import random
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import escnn
import escnn.nn as enn
import escnn.gspaces as gspaces


# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    seeds: tuple = (42, 43, 44)
    split_seed: int = 42

    model_type: str = "p4m"   # "z2" or "p4m"
    
    data_root: str = "./data"
    out_dir: str = "./results_z2_resnet44_3seed"

    epochs: int = 50
    batch_size: int = 128
    num_workers: int = 2

    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4

    depth: int = 44           # 6n + 2
    widths: tuple = (32, 64, 128)
    num_classes: int = 10

    train_size: int = 40000
    val_size: int = 10000

    deterministic: bool = False   # faster on Colab if False
    save_checkpoints: bool = True


CFG = Config()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


# -------------------------
# Repro / speed
# -------------------------
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


# -------------------------
# Data
# -------------------------
def build_splits(cfg: Config):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train_aug = torchvision.datasets.CIFAR10(
        root=cfg.data_root, train=True, download=True, transform=train_transform
    )
    full_train_eval = torchvision.datasets.CIFAR10(
        root=cfg.data_root, train=True, download=True, transform=eval_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=cfg.data_root, train=False, download=True, transform=eval_transform
    )

    generator = torch.Generator().manual_seed(cfg.split_seed)
    perm = torch.randperm(len(full_train_aug), generator=generator).tolist()

    train_idx = perm[:cfg.train_size]
    val_idx = perm[cfg.train_size: cfg.train_size + cfg.val_size]

    trainset = Subset(full_train_aug, train_idx)
    valset = Subset(full_train_eval, val_idx)

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


# -------------------------
# Model: standard Z2 ResNet for CIFAR
# ResNet44 => n=(44-2)/6 = 7 blocks per stage
# -------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, depth=44, widths=(32, 64, 128), num_classes=10):
        super().__init__()

        if (depth - 2) % 6 != 0:
            raise ValueError("Depth must satisfy depth = 6n + 2, e.g. 20, 32, 44, 56.")
        n = (depth - 2) // 6

        self.in_channels = widths[0]

        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(widths[0], n, stride=1)
        self.stage2 = self._make_stage(widths[1], n, stride=2)
        self.stage3 = self._make_stage(widths[2], n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[2], num_classes)

        self._init_weights()

    def _make_stage(self, out_channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# # -------------------------
# # Model: P4m ResNet for CIFAR (via escnn)
# # P4m = rotations by 90 degrees + reflections
# # -------------------------

# class BasicBlockP4m(torch.nn.Module):
#     def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType, stride: int = 1):
#         super().__init__()

#         # Group convolution on P4m feature fields
#         self.conv1 = enn.R2Conv(in_type, out_type, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = enn.InnerBatchNorm(out_type)
#         self.relu1 = enn.ReLU(out_type, inplace=True)

#         self.conv2 = enn.R2Conv(out_type, out_type, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = enn.InnerBatchNorm(out_type)

#         # Shortcut: either identity or 1x1 conv to match type/stride
#         if stride != 1 or in_type != out_type:
#             self.shortcut = enn.SequentialModule(
#                 enn.R2Conv(in_type, out_type, kernel_size=1, stride=stride, padding=0, bias=False),
#                 enn.InnerBatchNorm(out_type),
#             )
#         else:
#             self.shortcut = enn.IdentityModule(in_type)

#         self.relu2 = enn.ReLU(out_type, inplace=True)

#     def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
#         identity = self.shortcut(x)

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu1(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         out = out + identity
#         out = self.relu2(out)
#         return out


# class ResNetCIFAR_P4m(torch.nn.Module):
#     def __init__(self, depth=44, widths=(32, 64, 128), num_classes=10):
#         super().__init__()

#         if (depth - 2) % 6 != 0:
#             raise ValueError("Depth must satisfy depth = 6n + 2, e.g. 20, 32, 44, 56.")
#         n = (depth - 2) // 6

#         # Define the P4m symmetry space
#         self.r2_act = gspaces.flipRot2dOnR2(N=4)  # rotations (4) + flips = P4m

#         # Input type: 3 trivial representations (RGB channels)
#         in_type = enn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])

#         # We’ll use the regular representation for feature fields
#         def feat_type(c):
#             return enn.FieldType(self.r2_act, c * [self.r2_act.regular_repr])

#         self.in_type = in_type
#         self.in_channels = widths[0]

#         # "Lifting" conv: from trivial (image) to regular (group) features
#         self.stem = enn.SequentialModule(
#             enn.R2Conv(in_type, feat_type(widths[0]), kernel_size=3, stride=1, padding=1, bias=False),
#             enn.InnerBatchNorm(feat_type(widths[0])),
#             enn.ReLU(feat_type(widths[0]), inplace=True),
#         )

#         self.stage1 = self._make_stage(feat_type(widths[0]), n, stride=1)
#         self.stage2 = self._make_stage(feat_type(widths[1]), n, stride=2)
#         self.stage3 = self._make_stage(feat_type(widths[2]), n, stride=2)

#         # Pool over the group dimension to get invariance to P4m
#         self.gpool = enn.GroupPooling(self.stage3.out_type)

#         # After group pooling, output is a plain tensor with channels = widths[2]
#         self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = torch.nn.Linear(widths[2], num_classes)

#     def _make_stage(self, out_type: enn.FieldType, blocks: int, stride: int):
#         layers = []

#         # Determine current type by looking at what came before
#         if not hasattr(self, "_current_type"):
#             # after stem:
#             self._current_type = self.stem.out_type

#         layers.append(BasicBlockP4m(self._current_type, out_type, stride=stride))
#         self._current_type = out_type

#         for _ in range(1, blocks):
#             layers.append(BasicBlockP4m(self._current_type, out_type, stride=1))
#             self._current_type = out_type

#         seq = enn.SequentialModule(*layers)
#         seq.out_type = out_type  # convenience
#         return seq

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Wrap tensor into a GeometricTensor with the correct input type
#         x = enn.GeometricTensor(x, self.in_type)

#         x = self.stem(x)
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)

#         x = self.gpool(x)      # GeometricTensor -> GeometricTensor with trivial reps
#         x = x.tensor           # -> plain torch Tensor [B, C, H, W]

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x




# -------------------------
# Train / eval
# -------------------------
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

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

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def train_one_seed(cfg: Config, seed: int, trainset, valset, testset):
    print(f"\n================ Seed {seed} ================\n", flush=True)
    set_seed(seed, deterministic=cfg.deterministic)

    train_loader = make_loader(trainset, cfg.batch_size, True, cfg.num_workers, seed)
    val_loader = make_loader(valset, cfg.batch_size, False, cfg.num_workers, seed)
    test_loader = make_loader(testset, cfg.batch_size, False, cfg.num_workers, seed)

    # if cfg.model_type == "z2":
    #     model = ResNetCIFAR(depth=cfg.depth, widths=cfg.widths, num_classes=cfg.num_classes).to(DEVICE)
    # elif cfg.model_type == "p4m":
    #     model = ResNetCIFAR_P4m(depth=cfg.depth, widths=cfg.widths, num_classes=cfg.num_classes).to(DEVICE)
    # else:
    #     raise ValueError(f"Unknown model_type: {cfg.model_type}")
    model = ResNetCIFAR(depth=cfg.depth, widths=cfg.widths, num_classes=cfg.num_classes).to(DEVICE)

    criterion_train = nn.CrossEntropyLoss()
    criterion_eval = nn.CrossEntropyLoss(reduction="sum")

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=False,
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[25, 40],
        gamma=0.1,
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

    run_dir = os.path.join(cfg.out_dir, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

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

        train_loss = running_loss / total_train
        train_acc = 100.0 * correct_train / total_train

        val_loss, val_acc = evaluate(model, val_loader, criterion_eval)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(current_lr))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

            if cfg.save_checkpoints:
                torch.save(best_state, os.path.join(run_dir, "best_model.pth"))

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            print(
                f"Seed {seed} | Epoch [{epoch:03d}/{cfg.epochs}] "
                f"lr={current_lr:.5f} "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.2f}% "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.2f}%",
                flush=True,
            )

        scheduler.step()

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    val_loss, val_acc = evaluate(model, val_loader, criterion_eval)
    test_loss, test_acc = evaluate(model, test_loader, criterion_eval)

    result = {
        "seed": seed,
        "params": int(count_parameters(model)),
        "best_val_accuracy_during_training": float(best_val_acc),
        "final_best_checkpoint_val_loss": float(val_loss),
        "final_best_checkpoint_val_accuracy": float(val_acc),
        "final_best_checkpoint_test_loss": float(test_loss),
        "final_best_checkpoint_test_accuracy": float(test_acc),
        "history_file": os.path.join(run_dir, "train_history.json"),
        "checkpoint_file": os.path.join(run_dir, "best_model.pth") if cfg.save_checkpoints else None,
    }

    with open(os.path.join(run_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(run_dir, "metrics_summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\nBest-checkpoint evaluation", flush=True)
    print(f"Seed {seed} | Validation loss: {val_loss:.4f}", flush=True)
    print(f"Seed {seed} | Validation accuracy: {val_acc:.2f}%", flush=True)
    print(f"Seed {seed} | Test loss: {test_loss:.4f}", flush=True)
    print(f"Seed {seed} | Test accuracy: {test_acc:.2f}%", flush=True)

    return result


def summarize_results(cfg: Config, results):
    test_accs = [r["final_best_checkpoint_test_accuracy"] for r in results]
    val_accs = [r["final_best_checkpoint_val_accuracy"] for r in results]

    summary = {
        "config": asdict(cfg),
        "device": str(DEVICE),
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

    os.makedirs(cfg.out_dir, exist_ok=True)

    with open(os.path.join(cfg.out_dir, "summary_all_seeds.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(cfg.out_dir, "summary_all_seeds.txt"), "w") as f:
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Params: {results[0]['params']:,}\n")
        f.write(f"Val acc mean/std: {summary['aggregate']['val_accuracy_mean']:.2f} / {summary['aggregate']['val_accuracy_std']:.2f}\n")
        f.write(f"Test acc mean/std: {summary['aggregate']['test_accuracy_mean']:.2f} / {summary['aggregate']['test_accuracy_std']:.2f}\n")
        f.write(f"Test error mean/std: {summary['aggregate']['test_error_mean']:.2f} / {summary['aggregate']['test_error_std']:.2f}\n")

    print("\n================ Final 3-seed summary ================\n", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Params: {results[0]['params']:,}", flush=True)
    print(
        f"Validation accuracy: {summary['aggregate']['val_accuracy_mean']:.2f}% "
        f"+/- {summary['aggregate']['val_accuracy_std']:.2f}",
        flush=True,
    )
    print(
        f"Test accuracy: {summary['aggregate']['test_accuracy_mean']:.2f}% "
        f"+/- {summary['aggregate']['test_accuracy_std']:.2f}",
        flush=True,
    )
    print(
        f"Test error: {summary['aggregate']['test_error_mean']:.2f}% "
        f"+/- {summary['aggregate']['test_error_std']:.2f}",
        flush=True,
    )

    return summary


def main():
    os.makedirs(CFG.out_dir, exist_ok=True)

    print("Torch version:", torch.__version__, flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Device:", DEVICE, flush=True)

    trainset, valset, testset = build_splits(CFG)
    print(f"Train size: {len(trainset)}", flush=True)
    print(f"Val size:   {len(valset)}", flush=True)
    print(f"Test size:  {len(testset)}", flush=True)

    temp_model = ResNetCIFAR(depth=CFG.depth, widths=CFG.widths, num_classes=CFG.num_classes)
    print(f"Model params: {count_parameters(temp_model):,}", flush=True)
    del temp_model

    all_results = []
    for seed in CFG.seeds:
        result = train_one_seed(CFG, seed, trainset, valset, testset)
        all_results.append(result)

    summarize_results(CFG, all_results)


if __name__ == "__main__":
    main()
