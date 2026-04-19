import json
import os
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader

try:
    from e2cnn import gspaces
    from e2cnn import nn as enn
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "e2cnn is not installed in the current Python environment. "
        "Install it in a compatible environment, then re-run this script."
    ) from e


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# CIFAR-10 statistics (paper-faithful preprocessing choice)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResidualBlockP4M(nn.Module):
    def __init__(self, gspace, in_channels, out_channels, stride=1):
        super().__init__()

        in_type = enn.FieldType(gspace, in_channels * [gspace.regular_repr])
        out_type = enn.FieldType(gspace, out_channels * [gspace.regular_repr])

        self.conv1 = enn.R2Conv(
            in_type,
            out_type,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = enn.InnerBatchNorm(out_type)
        self.relu1 = enn.ReLU(out_type, inplace=True)

        self.conv2 = enn.R2Conv(
            out_type,
            out_type,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = enn.InnerBatchNorm(out_type)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = enn.SequentialModule(
                enn.R2Conv(in_type, out_type, kernel_size=1, stride=stride, padding=0, bias=False),
                enn.InnerBatchNorm(out_type),
            )
        else:
            self.shortcut = enn.IdentityModule(in_type)

        self.out_relu = enn.ReLU(out_type, inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.out_relu(out)
        return out


class P4MResNet(nn.Module):
    """
    Small p4m-ResNet for CIFAR-10.

    depth=20 -> n=3 residual blocks per stage
    depth=26 -> n=4 residual blocks per stage
    """

    def __init__(self, depth=20, widths=(12, 24, 48), num_classes=10):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("Depth must satisfy depth = 6n + 2, e.g., 20, 26, 32, 44.")

        self.gspace = gspaces.FlipRot2dOnR2(N=4)
        n = (depth - 2) // 6

        self.in_type = enn.FieldType(self.gspace, 3 * [self.gspace.trivial_repr])
        stem_type = enn.FieldType(self.gspace, widths[0] * [self.gspace.regular_repr])

        self.stem = enn.SequentialModule(
            enn.R2Conv(self.in_type, stem_type, kernel_size=3, stride=1, padding=1, bias=False),
            enn.InnerBatchNorm(stem_type),
            enn.ReLU(stem_type, inplace=True),
        )

        self.stage1 = self._make_stage(widths[0], widths[0], n, stride=1)
        self.stage2 = self._make_stage(widths[0], widths[1], n, stride=2)
        self.stage3 = self._make_stage(widths[1], widths[2], n, stride=2)

        self.gpool = enn.GroupPooling(enn.FieldType(self.gspace, widths[2] * [self.gspace.regular_repr]))
        self.classifier = nn.Linear(widths[2], num_classes)

    def _make_stage(self, in_channels, out_channels, blocks, stride):
        layers = [ResidualBlockP4M(self.gspace, in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlockP4M(self.gspace, out_channels, out_channels, stride=1))
        return nn.ModuleList(layers)

    @staticmethod
    def _forward_stage(x, stage):
        for block in stage:
            x = block(x)
        return x

    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        x = self.stem(x)

        x = self._forward_stage(x, self.stage1)
        x = self._forward_stage(x, self.stage2)
        x = self._forward_stage(x, self.stage3)

        x = self.gpool(x)
        # After GroupPooling, channel dimension equals the number of fields.
        x = x.tensor.mean(dim=(2, 3))
        x = self.classifier(x)
        return x


def build_splits(data_root="./data", seed=42):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    # Build identical image pools with different transforms so split indices are shared.
    full_train_aug = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    full_train_eval = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=eval_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=eval_transform
    )

    # Paper-style split used in your previous scripts: 40k train, 10k val.
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full_train_aug), generator=generator).tolist()
    train_idx = perm[:40000]
    val_idx = perm[40000:]

    trainset = Subset(full_train_aug, train_idx)
    valset = Subset(full_train_eval, val_idx)

    return trainset, valset, testset


def evaluate(net, dataset, batch_size=128, num_workers=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total = 0
    correct = 0

    net.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = net(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def train_model(
    net,
    trainset,
    valset,
    epochs=300,
    batch_size=128,
    num_workers=2,
    lr=0.05,
    momentum=0.9,
    weight_decay=5e-4,
):
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=False,
    )

    # Paper-style milestone schedule: drop LR by 10 at epochs 50, 100, 150.
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50, 100, 150],
        gamma=0.1,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        net.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = net(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate(net, valset, batch_size=batch_size, num_workers=num_workers)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(current_lr))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(net.state_dict())

        print(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"lr={current_lr:.5f} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.2f}%"
        )

        scheduler.step()

    return best_state, best_val_acc, history


def run_experiment(
    depth=20,
    widths=(12, 24, 48),
    epochs=300,
    batch_size=128,
    out_dir="results/gcnn_p4m_resnet_report",
    seed=42,
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    trainset, valset, testset = build_splits(seed=seed)

    net = P4MResNet(depth=depth, widths=widths, num_classes=num_classes).to(device)
    best_state, best_val_acc, history = train_model(
        net,
        trainset,
        valset,
        epochs=epochs,
        batch_size=batch_size,
    )

    if best_state is None:
        raise RuntimeError("Best checkpoint was not captured. Training may have failed.")

    # Evaluate using best-validation checkpoint, not final epoch weights.
    net.load_state_dict(best_state)
    val_loss, val_acc = evaluate(net, valset, batch_size=batch_size)
    test_loss, test_acc = evaluate(net, testset, batch_size=batch_size)

    print("\nBest-checkpoint evaluation")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.2f}%")

    best_model_path = os.path.join(out_dir, "gcnn_p4m_resnet_best.pth")
    torch.save(best_state, best_model_path)

    summary = {
        "config": {
            "depth": depth,
            "widths": list(widths),
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "SGD",
            "lr": 0.05,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "lr_milestones": [50, 100, 150],
            "lr_gamma": 0.1,
            "augmentation": ["RandomCrop(32,padding=4)", "RandomHorizontalFlip()"],
            "normalization_mean": list(CIFAR10_MEAN),
            "normalization_std": list(CIFAR10_STD),
            "seed": seed,
        },
        "split": {
            "train": len(trainset),
            "val": len(valset),
            "test": len(testset),
        },
        "best_checkpoint_metrics": {
            "best_val_accuracy_during_training": float(best_val_acc),
            "validation_loss": float(val_loss),
            "validation_accuracy": float(val_acc),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
        },
    }

    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved best checkpoint to: {best_model_path}")
    print(f"Saved report artifacts to: {out_dir}")


if __name__ == "__main__":
    # Use depth=26 for a stronger first run if your GPU memory/time permits.
    run_experiment(depth=20, widths=(12, 24, 48), epochs=300, batch_size=128)
