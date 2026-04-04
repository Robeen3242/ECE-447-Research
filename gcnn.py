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
    A p4-equivariant CNN (equivariant to 90-degree rotations + translations).

    Filter counts are halved relative to the CNN baseline to keep the
    parameter count approximately equal, as described in the paper (Section 8).
    The group p4 has 4 elements, so we divide filters by sqrt(4) = 2.
    """
    def __init__(self):
        super().__init__()

        # The symmetry group: p4 = translations + 90-degree rotations
        self.gspace = gspaces.rot2dOnR2(N=4)

        # Field types define how feature maps transform under the group
        self.in_type  = enn.FieldType(self.gspace, 3 * [self.gspace.trivial_repr])
        feat_type_1   = enn.FieldType(self.gspace, 3 * [self.gspace.regular_repr])   # 6 / 2 = 3
        feat_type_2   = enn.FieldType(self.gspace, 8 * [self.gspace.regular_repr])   # 16 / 2 = 8

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

        # Pool over all rotations to produce a rotation-invariant feature map
        # before passing to the fully connected layers
        self.gpool = enn.GroupPooling(feat_type_2)

        # After GroupPooling the output is a standard tensor — use regular nn layers
        # feat_type_2 has 8 regular_repr, each of size 4 (p4 has 4 elements)
        # so the flattened size = 8 * 4 * 5 * 5 = 800... but we compute it dynamically below
        self.fc1 = None  # set after first forward pass, or compute manually
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Compute the flattened feature size with a dummy forward pass
        self._initialize_fc()

    def _initialize_fc(self):
        dummy = torch.zeros(1, 3, 32, 32)
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

        # .tensor unwraps the GeometricTensor back to a plain torch.Tensor
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


def evaluate(net):
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    correct_pred = {classname: 0 for classname in classes}
    total_pred   = {classname: 0 for classname in classes}

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    total_correct = sum(correct_pred.values())
    total_total   = sum(total_pred.values())
    overall = 100 * total_correct / total_total
    print(f'\nOverall accuracy: {overall:.1f}%')

    for classname in classes:
        accuracy = 100 * float(correct_pred[classname]) / total_pred[classname]
        print(f'  {classname:5s}: {accuracy:.1f}%')

    return overall


if __name__ == '__main__':
    set_seed(42)
    net = GCNN().to(device)
    losses = train(net, epochs=30)
    evaluate(net)
    torch.save(net.state_dict(), 'gcnn.pth')