import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

try:
    # from escnn import gspaces
    # from escnn import nn as enn
    from e2cnn import gspaces
    from e2cnn import nn as enn

except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "escnn is not installed in the current Python environment. "
        "Install it in a compatible environment, then re-run this script."
    ) from e


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainset, valset = torch.utils.data.random_split(
    full_trainset,
    [40000, 10000],
    generator=torch.Generator().manual_seed(42)
)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
num_classes = len(classes)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GCNNP4M(nn.Module):
    """
    A p4m-equivariant CNN (equivariant to 90-degree rotations, reflections, and translations).

    Compared with p4, p4m has 8 group elements (4 rotations x 2 mirror states).
    We reduce the number of regular fields so parameter count remains in the same range.
    """

    def __init__(self):
        super().__init__()

        # p4m = translations + 90-degree rotations + reflections (dihedral D4)
        self.gspace = gspaces.FlipRot2dOnR2(N=4)

        self.in_type = enn.FieldType(self.gspace, 3 * [self.gspace.trivial_repr])
        feat_type_1 = enn.FieldType(self.gspace, 12 * [self.gspace.regular_repr])
        feat_type_2 = enn.FieldType(self.gspace, 32 * [self.gspace.regular_repr])

        self.block1 = enn.SequentialModule(
            enn.R2Conv(self.in_type, feat_type_1, kernel_size=3, padding=1, bias=False),
            enn.ReLU(feat_type_1),
        )
        self.pool1 = enn.PointwiseMaxPool(feat_type_1, kernel_size=2, stride=2)

        self.block2 = enn.SequentialModule(
            enn.R2Conv(feat_type_1, feat_type_2, kernel_size=3, padding=1, bias=False),
            enn.ReLU(feat_type_2),
        )
        self.pool2 = enn.PointwiseMaxPool(feat_type_2, kernel_size=2, stride=2)

        # Group pooling removes orientation/reflection channels before FC layers.
        self.gpool = enn.GroupPooling(feat_type_2)

        self.fc1 = None  # Initialized after seeing the output size of gpool
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self._initialize_fc()

    def _initialize_fc(self):
        dummy = torch.zeros(1, 3, 32, 32)  # Dummy input to determine output size after gpool
        x = enn.GeometricTensor(dummy, self.in_type)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.gpool(x)
        flat_size = x.tensor.flatten(1).shape[1]
        self.fc1 = nn.Linear(flat_size, 120)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.gpool(x)

        x = x.tensor.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, epochs=100):
    g = torch.Generator()
    g.manual_seed(42)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, generator=g
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epoch_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        epoch_losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.3f}')

    return epoch_losses


def collect_predictions(net, dataset):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    criterion = nn.CrossEntropyLoss(reduction='sum')

    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    total_samples = 0

    net.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            probs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)

            all_labels.append(labels.cpu())
            all_preds.append(predictions.cpu())
            all_probs.append(probs.cpu())
            total_loss += loss.item()
            total_samples += labels.size(0)

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * (y_true == y_pred).mean()

    return y_true, y_pred, y_prob, avg_loss, accuracy


def evaluate(net):
    y_true, y_pred, _, avg_loss, overall = collect_predictions(net, testset)
    print(f'\nTest loss: {avg_loss:.4f}')
    print(f'Overall accuracy: {overall:.1f}%')

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    for label_idx, pred_idx in zip(y_true, y_pred):
        if label_idx == pred_idx:
            correct_pred[classes[label_idx]] += 1
        total_pred[classes[label_idx]] += 1

    for classname in classes:
        accuracy = 100 * float(correct_pred[classname]) / total_pred[classname]
        print(f'  {classname:5s}: {accuracy:.1f}%')

    return overall


def validate(net):
    _, _, _, avg_loss, overall = collect_predictions(net, valset)
    print(f'\nValidation loss: {avg_loss:.4f}')
    print(f'Validation accuracy: {overall:.1f}%')
    return overall


def save_report_artifacts(net, epoch_losses, out_dir='results/gcnn_p4m_report'):
    os.makedirs(out_dir, exist_ok=True)

    y_true, y_pred, y_prob, test_loss, test_acc = collect_predictions(net, testset)
    _, _, _, val_loss, val_acc = collect_predictions(net, valset)

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(epoch_losses) + 1), epoch_losses, marker='o')
    plt.title('Training Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'train_loss_curve.png'), dpi=200)
    plt.close()

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Test Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=200)
    plt.close()

    y_true_onehot = np.eye(num_classes)[y_true]
    plt.figure(figsize=(9, 7))
    aucs = {}
    for i, cls_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[cls_name] = float(roc_auc)
        plt.plot(fpr, tpr, label=f'{cls_name} (AUC={roc_auc:.3f})', linewidth=1.4)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curves (One-vs-Rest)')
    plt.legend(fontsize=8, loc='lower right')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'roc_ovr.png'), dpi=200)
    plt.close()

    confidences = y_prob.max(axis=1)
    is_correct = (y_true == y_pred)
    plt.figure(figsize=(8, 4.5))
    plt.hist(confidences[is_correct], bins=20, alpha=0.7, label='Correct', density=True)
    plt.hist(confidences[~is_correct], bins=20, alpha=0.7, label='Incorrect', density=True)
    plt.xlabel('Predicted probability of chosen class')
    plt.ylabel('Density')
    plt.title('Prediction Confidence Histogram (Test)')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confidence_histogram.png'), dpi=200)
    plt.close()

    class_acc = []
    for c in range(num_classes):
        mask = (y_true == c)
        acc = 100.0 * (y_pred[mask] == y_true[mask]).mean()
        class_acc.append(acc)
    plt.figure(figsize=(9, 4.5))
    plt.bar(classes, class_acc)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-class Test Accuracy')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'per_class_accuracy.png'), dpi=200)
    plt.close()

    summary = {
        'split': {
            'train': len(trainset),
            'val': len(valset),
            'test': len(testset)
        },
        'final_metrics': {
            'validation_loss': float(val_loss),
            'validation_accuracy': float(val_acc),
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc)
        },
        'roc_auc_ovr': aucs
    }
    with open(os.path.join(out_dir, 'metrics_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved report artifacts to: {out_dir}")


if __name__ == '__main__':
    set_seed(42)
    net = GCNNP4M().to(device)
    losses = train(net, epochs=100)
    validate(net)
    evaluate(net)
    save_report_artifacts(net, losses)
    torch.save(net.state_dict(), 'gcnn_p4m.pth')
