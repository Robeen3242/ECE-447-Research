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
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GCNNP4M(nn.Module):
    """
    p4m-equivariant CNN — ~1.37M parameters.

    p4m has 8 group elements (4 rotations x 2 mirror states). Filter counts
    are chosen so that learnable parameters match the Z2 CNN and p4 GCNN.
      conv1: trivial(3) -> regular(48),  kernel=5
      conv2: regular(48) -> regular(88), kernel=5
      After GroupPooling: flat = 88 * 5 * 5 = 2200
      fc1: 2200 -> 512, fc2: 512 -> 256, fc3: 256 -> 10
    """

    def __init__(self,in_channels=3,num_classes=10,group_order=4,regular_channels=(36, 66),kernel_sizes=(5, 5),hidden_dims=(384, 192),
        input_size=(32, 32),pool_kernel=2,pool_stride=2,):
        super().__init__()
        if len(regular_channels) != 2 or len(kernel_sizes) != 2 or len(hidden_dims) != 2:
            raise ValueError("regular_channels, kernel_sizes, and hidden_dims must each contain exactly 2 values")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.group_order = group_order
        self.regular_channels = tuple(regular_channels)
        self.kernel_sizes = tuple(kernel_sizes)
        self.hidden_dims = tuple(hidden_dims)
        self.input_size = tuple(input_size)

        self.gspace = gspaces.flipRot2dOnR2(N=group_order)

        self.in_type = enn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr])
        feat_type_1 = enn.FieldType(self.gspace, regular_channels[0] * [self.gspace.regular_repr])
        feat_type_2 = enn.FieldType(self.gspace, regular_channels[1] * [self.gspace.regular_repr])

        self.block1 = enn.SequentialModule(
            enn.R2Conv(self.in_type, feat_type_1, kernel_size=kernel_sizes[0], padding=0, bias=False),
            enn.ReLU(feat_type_1),
        )
        self.pool1 = enn.PointwiseMaxPool(feat_type_1, kernel_size=pool_kernel, stride=pool_stride)

        self.block2 = enn.SequentialModule(
            enn.R2Conv(feat_type_1, feat_type_2, kernel_size=kernel_sizes[1], padding=0, bias=False),
            enn.ReLU(feat_type_2),
        )
        self.pool2 = enn.PointwiseMaxPool(feat_type_2, kernel_size=pool_kernel, stride=pool_stride)

        self.gpool = enn.GroupPooling(feat_type_2)

        flattened_dim = self._get_flattened_dim()
        self.fc1 = nn.Linear(flattened_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)

    def _get_flattened_dim(self):
        with torch.no_grad():
            sample = torch.zeros(1, self.in_channels, *self.input_size)
            sample = enn.GeometricTensor(sample, self.in_type)
            sample = self.block1(sample)
            sample = self.pool1(sample)
            sample = self.block2(sample)
            sample = self.pool2(sample)
            sample = self.gpool(sample)
            return sample.tensor.flatten(1).shape[1]

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
    valloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    criterion = nn.CrossEntropyLoss()
    val_criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(trainloader)
        history['train_loss'].append(float(avg_train_loss))

        net.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                val_loss += val_criterion(net(inputs), labels).item()
                val_samples += labels.size(0)

        avg_val_loss = val_loss / val_samples
        history['val_loss'].append(float(avg_val_loss))

        print(f'Epoch [{epoch + 1}/{epochs}]  Train Loss: {avg_train_loss:.3f}  Val Loss: {avg_val_loss:.4f}')

    return history


def evaluate(net):
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    criterion = nn.CrossEntropyLoss(reduction='sum')

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
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

    print(f'\nTest Loss: {avg_loss:.4f}')
    print(f'Overall Accuracy: {overall:.1f}%')
    for classname in classes:
        accuracy = 100 * float(correct_pred[classname]) / total_pred[classname]
        print(f'  {classname:5s}: {accuracy:.1f}%')

    per_class = {
        classname: round(100 * float(correct_pred[classname]) / total_pred[classname], 2)
        for classname in classes
    }
    return float(avg_loss), float(overall), per_class


def save_results(history, test_loss, test_acc, per_class, total_params, out_dir='results/gcnn_p4m_report'):
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        'config': {
            'model': 'GCNN (p4m)',
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
    net = GCNNP4M().to(device)
    total_params = sum(p.numel() for p in net.parameters())
    print(f'Total parameters: {total_params:,}')
    history = train(net, epochs=30)
    test_loss, test_acc, per_class = evaluate(net)
    save_results(history, test_loss, test_acc, per_class, total_params)
    torch.save(net.state_dict(), 'gcnn_p4m.pth')
