import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

from escnn import gspaces
from escnn import nn as enn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GCNN(nn.Module):
    """
    p4-equivariant CNN — ~70,989 parameters.

    Uses 8 regular repr in each conv layer. Free conv params are stored as
    base filters (rotated copies derived), so param count matches the Z2 CNN
    despite the larger internal feature map.
      conv1: trivial(3) -> regular(8),  kernel=5
      conv2: regular(8) -> regular(8),  kernel=5
      fc1:   200 -> 235, fc2: 235 -> 84, fc3: 84 -> 10
    """
    def __init__(self):
        super().__init__()

        self.gspace = gspaces.Rot2dOnR2(N=4)

        self.in_type  = enn.FieldType(self.gspace, 3 * [self.gspace.trivial_repr])
        feat_type_1   = enn.FieldType(self.gspace, 8 * [self.gspace.regular_repr])
        feat_type_2   = enn.FieldType(self.gspace, 8 * [self.gspace.regular_repr])

        self.block1 = enn.SequentialModule(
            enn.R2Conv(self.in_type, feat_type_1, kernel_size=5, padding=0, bias=False),
            enn.ReLU(feat_type_1),
        )
        self.pool1 = enn.PointwiseMaxPool(feat_type_1, kernel_size=2, stride=2)

        self.block2 = enn.SequentialModule(
            enn.R2Conv(feat_type_1, feat_type_2, kernel_size=5, padding=0, bias=False),
            enn.ReLU(feat_type_2),
        )
        self.pool2 = enn.PointwiseMaxPool(feat_type_2, kernel_size=2, stride=2)

        # GroupPooling collapses the 4 rotation channels -> 1 per field
        self.gpool = enn.GroupPooling(feat_type_2)

        self.fc1 = None  # sized dynamically below
        self.fc2 = nn.Linear(235, 84)
        self.fc3 = nn.Linear(84, 10)

        self._initialize_fc()

    def _initialize_fc(self):
        dummy = torch.zeros(1, 3, 32, 32)
        x = enn.GeometricTensor(dummy, self.in_type)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.gpool(x)
        flat_size = x.tensor.flatten(1).shape[1]  # should be 8*5*5 = 200
        self.fc1 = nn.Linear(flat_size, 235)

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


def train(net, epochs=30):
    g = torch.Generator()
    g.manual_seed(42)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, generator=g
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    history = {'train_loss': []}

    for epoch in range(epochs):
        running_loss = 0.0
        net.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        history['train_loss'].append(float(avg_loss))
        print(f'Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.3f}')

    return history


def evaluate(net):
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    criterion = nn.CrossEntropyLoss(reduction='sum')

    correct_pred = {classname: 0 for classname in classes}
    total_pred   = {classname: 0 for classname in classes}
    total_loss = 0.0
    total_samples = 0

    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            total_loss += criterion(outputs, labels).item()
            total_samples += labels.size(0)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    overall = 100 * sum(correct_pred.values()) / total_samples
    avg_loss = total_loss / total_samples

    print(f'\nTest loss: {avg_loss:.4f}')
    print(f'Overall accuracy: {overall:.1f}%')
    for classname in classes:
        accuracy = 100 * float(correct_pred[classname]) / total_pred[classname]
        print(f'  {classname:5s}: {accuracy:.1f}%')

    per_class = {
        classname: round(100 * float(correct_pred[classname]) / total_pred[classname], 2)
        for classname in classes
    }
    return float(avg_loss), float(overall), per_class


def save_results(history, test_loss, test_acc, per_class, out_dir='results/gcnn_report'):
    os.makedirs(out_dir, exist_ok=True)

    net_tmp = GCNN()
    total_params = sum(p.numel() for p in net_tmp.parameters())
    summary = {
        'config': {
            'model': 'GCNN (p4)',
            'total_params': total_params,
            'epochs': len(history['train_loss']),
            'batch_size': batch_size,
            'optimizer': 'SGD',
            'lr': 0.001,
            'momentum': 0.9,
        },
        'split': {
            'train': len(trainset),
            'test': len(testset),
        },
        'test_metrics': {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'per_class_accuracy': per_class,
        },
    }

    with open(os.path.join(out_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nSaved report artifacts to: {out_dir}')


if __name__ == '__main__':
    set_seed(42)
    net = GCNN().to(device)
    print(f'Total parameters: {sum(p.numel() for p in net.parameters()):,}')
    history = train(net, epochs=30)
    test_loss, test_acc, per_class = evaluate(net)
    save_results(history, test_loss, test_acc, per_class)
    torch.save(net.state_dict(), 'gcnn.pth')