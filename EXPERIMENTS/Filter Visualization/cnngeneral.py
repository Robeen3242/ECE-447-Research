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

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar10_datasets(root='./data', transform=DEFAULT_TRANSFORM, download=True):
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
    return trainset, testset


def create_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, generator=None):
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
    }
    if generator is not None:
        loader_kwargs['generator'] = generator
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


class Net(nn.Module):
    """
    Z2 CNN baseline aligned more closely with the CIFAR-10 setup in
    Cohen and Welling (2016): an All-CNN-style network with 9 convolution
    layers and global average pooling.
    """
    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        conv_channels=(96, 96, 96, 192, 192, 192, 192, 192, 10),
        kernel_sizes=(3, 3, 3, 3, 3, 3, 3, 1, 1),
        strides=(1, 1, 2, 1, 1, 2, 1, 1, 1),
    ):
        super().__init__()
        if not (len(conv_channels) == len(kernel_sizes) == len(strides) == 9):
            raise ValueError("conv_channels, kernel_sizes, and strides must each contain exactly 9 values")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_channels = tuple(conv_channels)
        self.kernel_sizes = tuple(kernel_sizes)
        self.strides = tuple(strides)

        layers = []
        current_channels = in_channels
        for out_channels, kernel_size, stride in zip(conv_channels, kernel_sizes, strides):
            padding = kernel_size // 2
            layers.extend([
                nn.Conv2d(current_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.ReLU(inplace=True),
            ])
            current_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.classifier_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier_pool(x)
        x = torch.flatten(x, 1)
        return x


def train(net, trainloader=None, epochs=30, lr=0.001, momentum=0.9):
    if trainloader is None:
        trainset, _ = get_cifar10_datasets()
        trainloader = create_dataloader(trainset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
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


def evaluate(net, testloader=None, class_names=classes):
    if testloader is None:
        _, testset = get_cifar10_datasets()
        testloader = create_dataloader(testset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    correct_pred = {classname: 0 for classname in class_names}
    total_pred = {classname: 0 for classname in class_names}
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
                    correct_pred[class_names[label]] += 1
                total_pred[class_names[label]] += 1

    overall = 100 * sum(correct_pred.values()) / total_samples
    avg_loss = total_loss / total_samples

    print(f'\nTest loss: {avg_loss:.4f}')
    print(f'Overall accuracy: {overall:.1f}%')
    for classname in class_names:
        accuracy = 100 * float(correct_pred[classname]) / total_pred[classname]
        print(f'  {classname:5s}: {accuracy:.1f}%')

    per_class = {
        classname: round(100 * float(correct_pred[classname]) / total_pred[classname], 2)
        for classname in class_names
    }
    return float(avg_loss), float(overall), per_class


def save_results(
    history,
    test_loss,
    test_acc,
    per_class,
    net=None,
    train_size=50000,
    test_size=10000,
    out_dir='results/cnngeneral_report',
):
    os.makedirs(out_dir, exist_ok=True)

    model = net if net is not None else Net()
    total_params = sum(p.numel() for p in model.parameters())
    conv_filters = list(model.conv_channels) if hasattr(model, 'conv_channels') else []
    summary = {
        'config': {
            'model': 'CNN (Z2)',
            'total_params': total_params,
            'epochs': len(history['train_loss']),
            'batch_size': batch_size,
            'optimizer': 'SGD',
            'lr': 0.001,
            'momentum': 0.9,
            'num_conv_layers': len(conv_filters),
            'filters_per_conv_layer': conv_filters,
        },
        'split': {
            'train': train_size,
            'test': test_size,
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
    trainset, testset = get_cifar10_datasets()
    trainloader = create_dataloader(trainset, batch_size=batch_size, shuffle=True)
    testloader = create_dataloader(testset, batch_size=batch_size, shuffle=False)
    net = Net().to(device)
    print(f'Total parameters: {sum(p.numel() for p in net.parameters()):,}')
    history = train(net, trainloader=trainloader, epochs=30)
    test_loss, test_acc, per_class = evaluate(net, testloader=testloader)
    save_results(history, test_loss, test_acc, per_class, net=net, train_size=len(trainset), test_size=len(testset))
    torch.save(net.state_dict(), 'cnngeneral.pth')
