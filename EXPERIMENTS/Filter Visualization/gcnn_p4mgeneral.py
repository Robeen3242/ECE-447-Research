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

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def set_seed(seed=42):
    torch.manual_seed(seed)
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


def create_train_val_loaders(trainset, batch_size=batch_size, val_fraction=0.1, seed=42):
    val_size = int(len(trainset) * val_fraction)
    if val_size <= 0:
        raise ValueError("val_fraction is too small for the dataset size")
    train_size = len(trainset) - val_size
    g = torch.Generator()
    g.manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(trainset, [train_size, val_size], generator=g)
    trainloader = create_dataloader(train_subset, batch_size=batch_size, shuffle=True, generator=g)
    valloader = create_dataloader(val_subset, batch_size=batch_size, shuffle=False)
    return trainloader, valloader, train_size, val_size


class GCNNP4M(nn.Module):
    """
    p4m-equivariant All-CNN-style network aligned with the CIFAR-10 setup in
    Cohen and Welling (2016). The p4m filter counts are scaled by about
    sqrt(8) relative to the planar baseline to keep the parameter count similar.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        group_order=4,
        regular_channels=(32, 32, 32, 64, 64, 64, 64, 64, 10),
        kernel_sizes=(3, 3, 3, 3, 3, 3, 3, 1, 1),
        strides=(1, 1, 2, 1, 1, 2, 1, 1, 1),
        input_size=(32, 32),
    ):
        super().__init__()
        if not (len(regular_channels) == len(kernel_sizes) == len(strides) == 9):
            raise ValueError("regular_channels, kernel_sizes, and strides must each contain exactly 9 values")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.group_order = group_order
        self.regular_channels = tuple(regular_channels)
        self.kernel_sizes = tuple(kernel_sizes)
        self.strides = tuple(strides)
        self.input_size = tuple(input_size)

        self.gspace = gspaces.flipRot2dOnR2(N=group_order)

        self.in_type = enn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr])
        blocks = []
        current_type = self.in_type
        for out_channels, kernel_size, stride in zip(regular_channels, kernel_sizes, strides):
            next_type = enn.FieldType(self.gspace, out_channels * [self.gspace.regular_repr])
            blocks.append(
                enn.SequentialModule(
                    enn.R2Conv(
                        current_type,
                        next_type,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    enn.ReLU(next_type),
                )
            )
            current_type = next_type

        self.blocks = nn.ModuleList(blocks)
        self.gpool = enn.GroupPooling(current_type)
        self.classifier_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        for block in self.blocks:
            x = block(x)
        x = self.gpool(x)
        x = self.classifier_pool(x.tensor)
        x = torch.flatten(x, 1)
        return x


def train(net, trainloader=None, valloader=None, epochs=30, lr=0.001, momentum=0.9, seed=42):
    if trainloader is None or valloader is None:
        trainset, _ = get_cifar10_datasets()
        trainloader, valloader, _, _ = create_train_val_loaders(trainset, batch_size=batch_size, seed=seed)

    criterion = nn.CrossEntropyLoss()
    val_criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

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
        correct = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                val_loss += val_criterion(outputs, labels).item()
                val_samples += labels.size(0)
                _, predictions = torch.max(outputs, 1)
                correct += (predictions == labels).sum().item()

        avg_val_loss = val_loss / val_samples
        history['val_loss'].append(float(avg_val_loss))
        val_accuracy = 100.0 * correct / val_samples
        history['val_accuracy'].append(float(val_accuracy))

        print(
            f'Epoch [{epoch + 1}/{epochs}]  Train Loss: {avg_train_loss:.3f}  '
            f'Val Loss: {avg_val_loss:.4f}  Val Acc: {val_accuracy:.2f}%'
        )

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

    print(f'\nTest Loss: {avg_loss:.4f}')
    print(f'Overall Accuracy: {overall:.1f}%')
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
    total_params,
    train_size=45000,
    val_size=5000,
    test_size=10000,
    out_dir='results/gcnn_p4mgeneral_report',
):
    os.makedirs(out_dir, exist_ok=True)

    conv_filters = list(GCNNP4M().regular_channels)
    summary = {
        'config': {
            'model': 'GCNN (p4m)',
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
            'val': val_size,
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
    trainloader, valloader, train_size, val_size = create_train_val_loaders(trainset, batch_size=batch_size, seed=42)
    testloader = create_dataloader(testset, batch_size=batch_size, shuffle=False)
    net = GCNNP4M().to(device)
    total_params = sum(p.numel() for p in net.parameters())
    print(f'Total parameters: {total_params:,}')
    history = train(net, trainloader=trainloader, valloader=valloader, epochs=30)
    test_loss, test_acc, per_class = evaluate(net, testloader=testloader)
    save_results(history, test_loss, test_acc, per_class, total_params, train_size=train_size, val_size=val_size, test_size=len(testset))
    torch.save(net.state_dict(), 'gcnn_p4mgeneral.pth')

