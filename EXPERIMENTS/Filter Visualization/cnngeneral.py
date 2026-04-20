import json
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
#learning rate needed to be moved from 0.001 to 0.01 to get better performance, and the number of epochs needed to be increased from 20 to 30 to allow for convergence. With these adjustments, the model achieved a test accuracy of around 80% on CIFAR-10, which is a reasonable result for a simple CNN architecture without data augmentation or advanced regularization techniques.
#batch normalization was also added


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64

NORMALIZE_TRANSFORM = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def build_cifar10_transform(train=False, rotation_degrees=0):
    transform_steps = []
    if rotation_degrees > 0:
        if train:
            transform_steps.append(transforms.RandomRotation(rotation_degrees))
        else:
            transform_steps.append(transforms.RandomRotation((rotation_degrees, rotation_degrees)))
    transform_steps.extend([
        transforms.ToTensor(),
        NORMALIZE_TRANSFORM,
    ])
    return transforms.Compose(transform_steps)


DEFAULT_TRANSFORM = build_cifar10_transform()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar10_datasets(
    root='./data',
    train_transform=DEFAULT_TRANSFORM,
    test_transform=DEFAULT_TRANSFORM,
    download=True,
):
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=test_transform)
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


class Net(nn.Module):
    """
    Z2 CNN baseline aligned more closely with the CIFAR-10 setup in
    Cohen and Welling (2016): an All-CNN-style network with 9 convolution
    layers, batch normalization, and global average pooling.
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
                nn.BatchNorm2d(out_channels),
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


def train(
    net,
    trainloader=None,
    valloader=None,
    epochs=30,
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    seed=42,
    step_size=10,
    gamma=0.1,
):
    if trainloader is None or valloader is None:
        trainset, testset = get_cifar10_datasets()
        g = torch.Generator()
        g.manual_seed(seed)
        if trainloader is None:
            trainloader, valloader, _, _ = create_train_val_loaders(trainset, batch_size=batch_size, seed=seed)
        elif valloader is None:
            _, valloader, _, _ = create_train_val_loaders(trainset, batch_size=batch_size, seed=seed)

    criterion = nn.CrossEntropyLoss()
    val_criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_val_accuracy = float('-inf')
    best_state = None

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

        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                val_loss += val_criterion(outputs, labels).item()
                _, predictions = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        avg_val_loss = val_loss / total
        val_accuracy = 100.0 * correct / total
        history['val_loss'].append(float(avg_val_loss))
        history['val_accuracy'].append(float(val_accuracy))
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(net.state_dict())
        print(
            f'Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.3f} '
            f'Val Loss: {avg_val_loss:.4f} Val Acc: {val_accuracy:.2f}%'
        )
        scheduler.step()

    if best_state is not None:
        net.load_state_dict(best_state)

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
    train_size=45000,
    val_size=5000,
    test_size=10000,
    train_rotation_degrees=0,
    test_rotation_degrees=0,
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
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'lr_step_size': 10,
            'lr_gamma': 0.1,
            'train_rotation_degrees': train_rotation_degrees,
            'test_rotation_degrees': test_rotation_degrees,
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
    train_rotation_degrees = 0
    test_rotation_degrees = 0

    set_seed(42)
    train_transform = build_cifar10_transform(train=True, rotation_degrees=train_rotation_degrees)
    test_transform = build_cifar10_transform(train=False, rotation_degrees=test_rotation_degrees)
    trainset, testset = get_cifar10_datasets(train_transform=train_transform, test_transform=test_transform)
    trainloader, valloader, train_size, val_size = create_train_val_loaders(trainset, batch_size=batch_size, seed=42)
    testloader = create_dataloader(testset, batch_size=batch_size, shuffle=False)
    net = Net().to(device)
    print(f'Total parameters: {sum(p.numel() for p in net.parameters()):,}')
    history = train(net, trainloader=trainloader, valloader=valloader, epochs=30)
    test_loss, test_acc, per_class = evaluate(net, testloader=testloader)
    save_results(
        history,
        test_loss,
        test_acc,
        per_class,
        net=net,
        train_size=train_size,
        val_size=val_size,
        test_size=len(testset),
        train_rotation_degrees=train_rotation_degrees,
        test_rotation_degrees=test_rotation_degrees,
    )
    torch.save(net.state_dict(), 'cnngeneral.pth')
