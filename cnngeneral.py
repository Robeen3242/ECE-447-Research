import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Net(nn.Module):
    """
    Z2 CNN baseline — ~70,983 parameters.
    conv1: 3->68, conv2: 68->160, fc1: 4000->256, fc2: 256->256, fc3: 256->10
    """
    def __init__(self,in_channels=3,num_classes=10,conv_channels=(68, 160),kernel_sizes=(5, 5),hidden_dims=(256, 256),
        input_size=(32, 32),pool_kernel=2,pool_stride=2,):
        super().__init__()
        if len(conv_channels) != 2 or len(kernel_sizes) != 2 or len(hidden_dims) != 2:
            raise ValueError("conv_channels, kernel_sizes, and hidden_dims must each contain exactly 2 values")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_channels = tuple(conv_channels)
        self.kernel_sizes = tuple(kernel_sizes)
        self.hidden_dims = tuple(hidden_dims)
        self.input_size = tuple(input_size)
        self.pool = nn.MaxPool2d(pool_kernel, pool_stride)

        self.conv1 = nn.Conv2d(in_channels, conv_channels[0], kernel_sizes[0])
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_sizes[1])

        flattened_dim = self._get_flattened_dim()
        self.fc1 = nn.Linear(flattened_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)

    def _get_flattened_dim(self):
        with torch.no_grad():
            sample = torch.zeros(1, self.in_channels, *self.input_size)
            sample = self.pool(F.relu(self.conv1(sample)))
            sample = self.pool(F.relu(self.conv2(sample)))
            return sample.flatten(1).shape[1]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, epochs=30):
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


def save_results(history, test_loss, test_acc, per_class, net=None, out_dir='results/cnn_report'):
    os.makedirs(out_dir, exist_ok=True)

    total_params = sum(p.numel() for p in (net if net is not None else Net()).parameters())
    summary = {
        'config': {
            'model': 'CNN (Z2)',
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
    net = Net().to(device)
    print(f'Total parameters: {sum(p.numel() for p in net.parameters()):,}')
    history = train(net, epochs=30)
    test_loss, test_acc, per_class = evaluate(net)
    save_results(history, test_loss, test_acc, per_class, net=net)
    torch.save(net.state_dict(), 'cnn_baseline.pth')
