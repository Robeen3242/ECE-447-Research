# ======================================================================================
# P4m EQUIVARIANT ResNet44 on CIFAR-10
# ======================================================================================
# Group Equivariant CNN using escnn library
# 
# Key Concepts:
# - P4m: Symmetry group with 4 rotations (0°, 90°, 180°, 270°) + reflections (8 total)
# - Equivariance: Rotating input → rotated output (automatic, by design)
# - Group Convolutions: Convolve over group representations for equivariance
# - Group Pooling: Pool over group dimension for final invariance
#
# Matched with Z2 model: batch_size=64, epochs=50, seeds=42,43
# Parameter count controlled to ~166K for fair accuracy comparison
# ======================================================================================

import os
import json
import copy
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize


import importlib

# escnn internals call escnn.group.* in some code paths.
# On some installs, the submodule is not attached to the top-level package
# until it is imported explicitly.
importlib.import_module("escnn.group")

import escnn
import escnn.nn as enn
import escnn.gspaces as gspaces

# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    seeds: tuple = (42, 43)
    split_seed: int = 42

    data_root: str = "./data"
    out_dir: str = "./results_p4m_resnet44_matched"

    # Matched with Z2 for fair comparison
    epochs: int = 50
    batch_size: int = 64
    num_workers: int = 2

    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4

    depth: int = 44              # must satisfy depth = 6n + 2
    # Reduced widths to match Z2 parameter count (~165K)
    # Z2 uses (8, 16, 32) directly as channels
    # P4m regular rep is 8D, so (3, 6, 12) ~ (24, 48, 96) channels
    widths: tuple = (4, 8, 16)   # Adjusted for param matching
    num_classes: int = 10

    train_size: int = 40000
    val_size: int = 10000

    deterministic: bool = False
    save_checkpoints: bool = True

    print_every_batches: int = 100


CFG = Config()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
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
# Model: P4m ResNet for CIFAR
# ========================= 
# P4m = 4 rotations + reflections (8-element symmetry group)
# Uses escnn for equivariant convolutions
# =========================

# ======================================================================================
# EQUIVARIANT BASIC BLOCK - P4m Version
# ======================================================================================
# Equivariant residual block:
# - Input/output are GeometricTensors with P4m field types
# - All operations (conv, BN, ReLU) preserve equivariance
# - Shortcut handles stride/channel changes while maintaining equivariance
# ======================================================================================

class BasicBlockP4m(enn.EquivariantModule):
    expansion = 1

    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType, stride: int = 1):
        """
        Create equivariant residual block.
        
        Args:
            in_type: Input FieldType (defines feature representation)
            out_type: Output FieldType
            stride: Spatial stride for first convolution
        """
        super().__init__()

        self.in_type = in_type
        self.out_type = out_type

        # First equivariant convolution block
        self.conv1 = enn.R2Conv(
            in_type, out_type,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = enn.InnerBatchNorm(out_type)  # Batch norm for feature fields
        self.relu1 = enn.ReLU(out_type, inplace=True)  # Equivariant ReLU

        # Second equivariant convolution block
        self.conv2 = enn.R2Conv(
            out_type, out_type,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = enn.InnerBatchNorm(out_type)

        # Skip connection with type matching
        if stride != 1 or in_type != out_type:
            # Need to transform: 1x1 conv + BN to match dimensions
            self.shortcut = enn.SequentialModule(
                enn.R2Conv(
                    in_type, out_type,
                    kernel_size=1, stride=stride, padding=0, bias=False
                ),
                enn.InnerBatchNorm(out_type),
            )
        else:
            # Identity shortcut when dimensions match
            self.shortcut = enn.IdentityModule(in_type)

        self.relu2 = enn.ReLU(out_type, inplace=True)

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        """Forward pass with residual connection."""
        assert x.type == self.in_type

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu2(out)

        assert out.type == self.out_type
        return out

    def evaluate_output_shape(self, input_shape):
        """Required by EquivariantModule for shape inference."""
        return input_shape


class ResNetCIFAR_P4m(nn.Module):
    """
    ResNet44 with P4m group equivariance on CIFAR-10.
    
    Architecture:
    1. Stem: Lifting convolution (RGB image → regular rep features)
    2. Stage 1-3: Equivariant residual blocks with same spatial resolution
    3. Group Pooling: Pool over group dimension for invariance
    4. Global Avg Pool: Pool over spatial dimension
    5. FC layer: Classification head
    
    Key point: All intermediate computations respect P4m symmetry.
    """
    
    def __init__(self, depth=44, widths=(4, 8, 16), num_classes=10):
        """
        Initialize P4m ResNet.
        
        Args:
            depth: Total depth (44 for ResNet44), must be 6n+2
            widths: (w1, w2, w3) widths for each stage in group representation
            num_classes: Number of output classes
        """
        super().__init__()

        # Validate depth
        if (depth - 2) % 6 != 0:
            raise ValueError("Depth must satisfy depth = 6n + 2, e.g. 20, 32, 44, 56.")
        n = (depth - 2) // 6

        # Create P4m symmetry group: 4-fold rotations + reflections = 8 elements
        self.r2_act = gspaces.rot2dOnR2(N=4)

        # Input type: 3 RGB channels as trivial (non-equivariant) representations
        # This is correct because RGB images don't rotate - we learn to rotate them
        self.in_type = enn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])

        # Feature field type constructor: creates regular representations
        # Each width unit becomes 8D (size of P4m group)
        def feat_type(c: int):
            return enn.FieldType(self.r2_act, c * [self.r2_act.regular_repr])

        stem_out_type = feat_type(widths[0])

        # Lifting convolution: RGB (trivial) → group features (regular rep)
        # This learns to activate the group dimension from spatial patterns
        self.stem = enn.SequentialModule(
            enn.R2Conv(
                self.in_type, stem_out_type,
                kernel_size=3, stride=1, padding=1, bias=False
            ),
            enn.InnerBatchNorm(stem_out_type),
            enn.ReLU(stem_out_type, inplace=True),
        )

        self._current_type = stem_out_type

        # Three stages at different resolutions
        # stride=2 downsamples spatial dimensions, equivariance preserved
        self.stage1 = self._make_stage(feat_type(widths[0]), n, stride=1)
        self.stage2 = self._make_stage(feat_type(widths[1]), n, stride=2)
        self.stage3 = self._make_stage(feat_type(widths[2]), n, stride=2)

        # Group pooling: max pool over group dimension (P4m → trivial)
        # Achieves invariance to P4m transformations
        # Input: features that transform under P4m
        # Output: features invariant to P4m (scalar features)
        self.gpool = enn.GroupPooling(self.stage3.out_type)

        # Standard global spatial average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification layer
        self.fc = nn.Linear(widths[2], num_classes)

        self._init_weights()

    def _make_stage(self, out_type: enn.FieldType, blocks: int, stride: int):
        """Build a stage with 'blocks' equivariant residual blocks."""
        layers = [BasicBlockP4m(self._current_type, out_type, stride=stride)]
        self._current_type = out_type

        # Additional blocks at constant resolution
        for _ in range(1, blocks):
            layers.append(BasicBlockP4m(self._current_type, out_type, stride=1))

        seq = enn.SequentialModule(*layers)
        seq.out_type = out_type
        return seq

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, 32, 32) - standard RGB image
        
        Returns:
            Logits (B, num_classes)
        """
        # Wrap input: RGB image → GeometricTensor with trivial representations
        x = enn.GeometricTensor(x, self.in_type)

        # Forward through stages (all equivariant)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        # Group pooling: remove group dimension, achieve invariance
        x = self.gpool(x)
        
        # Convert GeometricTensor back to regular tensor
        x = x.tensor

        # Standard global average pooling and FC
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------
# Train / eval
# -------------------------
def evaluate(model, loader, criterion, return_predictions=False):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Neural network model
        loader: Data loader
        criterion: Loss function
        return_predictions: If True, return predictions for analysis (ROC, confusion matrix, etc.)
    
    Returns:
        If return_predictions=False: (avg_loss, accuracy)
        If return_predictions=True: (avg_loss, accuracy, predictions, labels, probabilities)
    """
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

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    
    if return_predictions:
        return avg_loss, acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)
    return avg_loss, acc


def train_one_seed(cfg: Config, seed: int, trainset, valset, testset):
    print(f"\n{'='*90}", flush=True)
    print(f"SEED {seed}", flush=True)
    print(f"{'='*90}\n", flush=True)
    
    set_seed(seed, deterministic=cfg.deterministic)

    train_loader = make_loader(trainset, cfg.batch_size, True, cfg.num_workers, seed)
    val_loader = make_loader(valset, cfg.batch_size, False, cfg.num_workers, seed)
    test_loader = make_loader(testset, cfg.batch_size, False, cfg.num_workers, seed)

    model = ResNetCIFAR_P4m(
        depth=cfg.depth,
        widths=cfg.widths,
        num_classes=cfg.num_classes
    ).to(DEVICE)

    total_params = count_parameters(model)

    # Count parameters by type
    conv_params = 0
    bn_params = 0
    fc_params = 0
    for name, p in model.named_parameters():
        if "conv" in name.lower():
            conv_params += p.numel()
        elif "bn" in name.lower() or "batch" in name.lower():
            bn_params += p.numel()
        elif "fc" in name.lower():
            fc_params += p.numel()

    print("[Model Information - Seed {}]".format(seed), flush=True)
    print(f"  Total Trainable Params:  {total_params:,}", flush=True)
    if total_params > 0:
        conv_pct = (conv_params / total_params) * 100
        bn_pct = (bn_params / total_params) * 100
        fc_pct = (fc_params / total_params) * 100
        print(f"    - Group/Equiv Conv:    {conv_params:>10,} ({conv_pct:5.1f}%)", flush=True)
        print(f"    - BatchNorm params:    {bn_params:>10,} ({bn_pct:5.1f}%)", flush=True)
        print(f"    - FC layer:            {fc_params:>10,} ({fc_pct:5.1f}%)", flush=True)
    model_size_mb = total_params * 4 / 1024 / 1024
    print(f"  Estimated model size:    {model_size_mb:.2f} MB (float32)", flush=True)

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
        milestones=[30, 40],
        gamma=0.1,
    )

    # Data Loader Information
    print("\n[Data Loader Information]", flush=True)
    print(f"  Training batches:    {len(train_loader)}", flush=True)
    print(f"  Validation batches:  {len(val_loader)}", flush=True)
    print(f"  Test batches:        {len(test_loader)}", flush=True)

    # Optimizer Configuration
    print("\n[Optimizer Configuration]", flush=True)
    print(f"  Optimizer:           SGD", flush=True)
    print(f"  Learning rate:       {cfg.lr}", flush=True)
    print(f"  Momentum:            {cfg.momentum}", flush=True)
    print(f"  Weight decay:        {cfg.weight_decay}", flush=True)
    print(f"  Nesterov:            False", flush=True)

    # Learning Rate Schedule
    print("\n[Learning Rate Schedule]", flush=True)
    print(f"  Initial LR:          {cfg.lr}", flush=True)
    print(f"  Milestones:          [30, 40]", flush=True)
    print(f"  Gamma (decay):       0.1", flush=True)

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

    print(f"\n[Epoch Training]", flush=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        total_train = 0
        correct_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
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

            if batch_idx % cfg.print_every_batches == 0:
                print(
                    f"Seed {seed} | Epoch [{epoch:03d}/{cfg.epochs}] "
                    f"Batch [{batch_idx:04d}/{len(train_loader)}] "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

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
                f"  Epoch [{epoch:03d}/{cfg.epochs}]  "
                f"lr={current_lr:.5f}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:6.2f}%  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:6.2f}%",
                flush=True,
            )

        scheduler.step()

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion_eval, return_predictions=True)
    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(model, test_loader, criterion_eval, return_predictions=True)

    print("\n[Best Checkpoint Evaluation]", flush=True)
    print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:6.2f}%", flush=True)
    print(f"  Test       - Loss: {test_loss:.4f}, Accuracy: {test_acc:6.2f}%", flush=True)
    per_class_acc = {}
    for c in range(cfg.num_classes):
        mask = test_labels == c
        if mask.sum() > 0:
            per_class_acc[CLASSES[c]] = float(100.0 * (test_preds[mask] == test_labels[mask]).sum() / mask.sum())
    
    result = {
        "seed": seed,
        "params": int(count_parameters(model)),
        "best_val_accuracy_during_training": float(best_val_acc),
        "final_best_checkpoint_val_loss": float(val_loss),
        "final_best_checkpoint_val_accuracy": float(val_acc),
        "final_best_checkpoint_test_loss": float(test_loss),
        "final_best_checkpoint_test_accuracy": float(test_acc),
        "per_class_test_accuracy": per_class_acc,
        "history_file": os.path.join(run_dir, "train_history.json"),
        "checkpoint_file": os.path.join(run_dir, "best_model.pth") if cfg.save_checkpoints else None,
        "predictions_file": os.path.join(run_dir, "predictions.npz"),
    }
    
    # Save predictions and probabilities
    np.savez(
        os.path.join(run_dir, "predictions.npz"),
        val_preds=val_preds, val_labels=val_labels, val_probs=val_probs,
        test_preds=test_preds, test_labels=test_labels, test_probs=test_probs,
    )

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


def generate_roc_curve(cfg: Config, results):
    """Generate ROC curves for each seed."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for idx, result in enumerate(results):
        seed_dir = os.path.join(cfg.out_dir, f"seed_{result['seed']}")
        pred_file = os.path.join(seed_dir, "predictions.npz")
        
        if not os.path.exists(pred_file):
            continue
            
        data = np.load(pred_file)
        test_labels = data['test_labels']
        test_probs = data['test_probs']
        
        # Binarize labels
        labels_bin = label_binarize(test_labels, classes=range(cfg.num_classes))
        
        # Compute ROC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(cfg.num_classes):
            fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], test_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot
        ax = axes[idx]
        colors = plt.cm.tab10(np.linspace(0, 1, cfg.num_classes))
        for i in range(cfg.num_classes):
            ax.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                   label=f'{CLASSES[i]} (AUC = {roc_auc[i]:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - Seed {result["seed"]}')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    print(f"Saved ROC curves to {os.path.join(cfg.out_dir, 'roc_curves.png')}", flush=True)
    plt.close()


def generate_confusion_matrices(cfg: Config, results):
    """Generate confusion matrices for each seed."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6))
    if len(results) == 1:
        axes = [axes]
    
    for idx, result in enumerate(results):
        seed_dir = os.path.join(cfg.out_dir, f"seed_{result['seed']}")
        pred_file = os.path.join(seed_dir, "predictions.npz")
        
        if not os.path.exists(pred_file):
            continue
            
        data = np.load(pred_file)
        test_labels = data['test_labels']
        test_preds = data['test_preds']
        
        cm = confusion_matrix(test_labels, test_preds)
        
        # Normalize for better visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        ax = axes[idx]
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=CLASSES, yticklabels=CLASSES,
               ylabel='True label', xlabel='Predicted label')
        ax.set_title(f'Confusion Matrix (Normalized) - Seed {result["seed"]}')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                       ha="center", va="center",
                       color="white" if cm_normalized[i, j] > thresh else "black",
                       fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrices to {os.path.join(cfg.out_dir, 'confusion_matrices.png')}", flush=True)
    plt.close()


def generate_classification_reports(cfg: Config, results):
    """Generate detailed classification reports for each seed."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    
    all_reports = {}
    
    for result in results:
        seed_dir = os.path.join(cfg.out_dir, f"seed_{result['seed']}")
        pred_file = os.path.join(seed_dir, "predictions.npz")
        
        if not os.path.exists(pred_file):
            continue
            
        data = np.load(pred_file)
        test_labels = data['test_labels']
        test_preds = data['test_preds']
        
        report = classification_report(test_labels, test_preds, 
                                      target_names=CLASSES, output_dict=True)
        all_reports[f"seed_{result['seed']}"] = report
        
        # Print to console
        print(f"\n========== Classification Report - Seed {result['seed']} ==========", flush=True)
        print(classification_report(test_labels, test_preds, target_names=CLASSES), flush=True)
    
    # Save as JSON
    with open(os.path.join(cfg.out_dir, 'classification_reports.json'), 'w') as f:
        json.dump(all_reports, f, indent=2)
    
    print(f"Saved classification reports to {os.path.join(cfg.out_dir, 'classification_reports.json')}", flush=True)


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
        f.write(f"=== P4m ResNet44 Summary (Group Equivariant) ===\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Params: {results[0]['params']:,}\n")
        f.write(f"Config: batch_size={cfg.batch_size}, epochs={cfg.epochs}, lr={cfg.lr}\n")
        f.write(f"Symmetry: P4m (4 rotations + reflections)\n")
        f.write(f"Val acc mean/std: {summary['aggregate']['val_accuracy_mean']:.2f}% / {summary['aggregate']['val_accuracy_std']:.2f}%\n")
        f.write(f"Test acc mean/std: {summary['aggregate']['test_accuracy_mean']:.2f}% / {summary['aggregate']['test_accuracy_std']:.2f}%\n")
        f.write(f"Test error mean/std: {summary['aggregate']['test_error_mean']:.2f}% / {summary['aggregate']['test_error_std']:.2f}%\n")

    print("\n================ Final Summary (P4m ResNet44 - Group Equivariant) ================\n", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Params: {results[0]['params']:,}", flush=True)
    print(f"Batch size: {cfg.batch_size}, Epochs: {cfg.epochs}", flush=True)
    print(f"Symmetry Group: P4m (4 rotations + reflections)", flush=True)
    print(
        f"Validation accuracy: {summary['aggregate']['val_accuracy_mean']:.2f}% "
        f"+/- {summary['aggregate']['val_accuracy_std']:.2f}%",
        flush=True,
    )
    print(
        f"Test accuracy: {summary['aggregate']['test_accuracy_mean']:.2f}% "
        f"+/- {summary['aggregate']['test_accuracy_std']:.2f}%",
        flush=True,
    )
    print(
        f"Test error: {summary['aggregate']['test_error_mean']:.2f}% "
        f"+/- {summary['aggregate']['test_error_std']:.2f}%",
        flush=True,
    )

    # Generate analysis plots
    print("\nGenerating analysis plots...", flush=True)
    generate_roc_curve(cfg, results)
    generate_confusion_matrices(cfg, results)
    generate_classification_reports(cfg, results)

    return summary


def main():
    os.makedirs(CFG.out_dir, exist_ok=True)

    # System Information
    print("\n" + "="*90, flush=True)
    print("P4m EQUIVARIANT ResNet44 on CIFAR-10 (Matched Comparison with Z2)", flush=True)
    print("="*90 + "\n", flush=True)

    print("[System Information]", flush=True)
    print(f"  PyTorch version:     {torch.__version__}", flush=True)
    print(f"  escnn version:       {escnn.__version__}", flush=True)
    print(f"  CUDA available:      {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  GPU:                 {torch.cuda.get_device_name(0)}", flush=True)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU memory:          {gpu_mem:.2f} GB", flush=True)
    print(f"  Device:              {DEVICE}", flush=True)
    print(f"  Deterministic:       {CFG.deterministic}", flush=True)

    # Dataset Information
    trainset, valset, testset = build_splits(CFG)
    print("\n[Dataset Information]", flush=True)
    print(f"  Training samples:    {len(trainset)}", flush=True)
    print(f"  Validation samples:  {len(valset)}", flush=True)
    print(f"  Test samples:        {len(testset)}", flush=True)
    print(f"  Input shape:         (3, 32, 32)  # CIFAR-10", flush=True)
    print(f"  Num classes:         {CFG.num_classes}", flush=True)

    # Model Architecture
    temp_model = ResNetCIFAR_P4m(
        depth=CFG.depth,
        widths=CFG.widths,
        num_classes=CFG.num_classes
    )
    total_params = count_parameters(temp_model)

    # Count parameters by type
    conv_params = 0
    bn_params = 0
    fc_params = 0
    for name, p in temp_model.named_parameters():
        if "conv" in name.lower():
            conv_params += p.numel()
        elif "bn" in name.lower() or "batch" in name.lower():
            bn_params += p.numel()
        elif "fc" in name.lower():
            fc_params += p.numel()

    print("\n[Model Architecture]", flush=True)
    print(f"  Architecture:        ResNet44 with P4m Equivariance", flush=True)
    print(f"  Widths (per stage):  {CFG.widths}", flush=True)
    print(f"  Group:               P4m (4-fold rotation + reflection, 8 elements)", flush=True)
    print(f"  Representation:      Regular representation (8D per channel)", flush=True)
    print("\n  Total Trainable Params:  {total_params:,}".format(total_params=total_params), flush=True)
    if total_params > 0:
        conv_pct = (conv_params / total_params) * 100
        bn_pct = (bn_params / total_params) * 100
        fc_pct = (fc_params / total_params) * 100
        print(f"    - Group/Equiv Conv:    {conv_params:>10,} ({conv_pct:5.1f}%)", flush=True)
        print(f"    - BatchNorm params:    {bn_params:>10,} ({bn_pct:5.1f}%)", flush=True)
        print(f"    - FC layer:            {fc_params:>10,} ({fc_pct:5.1f}%)", flush=True)
    model_size_mb = total_params * 4 / 1024 / 1024
    print(f"  Estimated model size:    {model_size_mb:.2f} MB (float32)", flush=True)
    del temp_model

    # Training Configuration
    print("\n[Training Configuration]", flush=True)
    print(f"  Epochs:              {CFG.epochs}", flush=True)
    print(f"  Batch size:          {CFG.batch_size}", flush=True)
    print(f"  Learning rate:       {CFG.lr}", flush=True)
    print(f"  Momentum:            {CFG.momentum}", flush=True)
    print(f"  Weight decay:        {CFG.weight_decay}", flush=True)
    print(f"  LR milestones:       [30, 40]", flush=True)
    print(f"  Seeds:               {CFG.seeds}", flush=True)
    print(f"  Output directory:    {CFG.out_dir}", flush=True)

    print(f"\n{'='*90}", flush=True)
    print(f"STARTING TRAINING - {len(CFG.seeds)} SEED(S)", flush=True)
    print(f"{'='*90}", flush=True)

    all_results = []
    for seed in CFG.seeds:
        result = train_one_seed(CFG, seed, trainset, valset, testset)
        all_results.append(result)

    print(f"\n{'='*90}", flush=True)
    print(f"TRAINING COMPLETE - GENERATING SUMMARY", flush=True)
    print(f"{'='*90}\n", flush=True)

    summarize_results(CFG, all_results)


if __name__ == "__main__":
    main()

