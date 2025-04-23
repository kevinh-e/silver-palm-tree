from resnet_cifar import ResidualBlock, ResNet

import os
import sys
import argparse
import time
import math
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

DATAPATH = "./data/cifar10"
MODELPATH = "./models/RESNET_CIFAR10"

# Hyper parameters
batch_size = 128
n_epochs = 128
num_classes = 10
# as specified in Sec 4.2
learning_rate = 0.1


def ResNet20():
    """
    Creates a ResNet-20 Model for CIFAR10
            (n=3)
    Returns:
        (ResNet)
    """
    # 6n + 2 = 6*3 + 2 = 20
    return ResNet(ResidualBlock, [3, 3, 3])


def ResNet32():
    """
    Creates a ResNet-32 Model for CIFAR10
            (n=4)
    Returns:
        (ResNet)
    """
    # 6n + 2 = 6*5 + 2 = 32
    return ResNet(ResidualBlock, [5, 5, 5])


def ResNet44():
    """
    Creates a ResNet-44 Model for CIFAR10
            (n=7)
    Returns:
        (ResNet)
    """
    # 6n + 2 = 6*7 + 2 = 44
    return ResNet(ResidualBlock, [7, 7, 7])


def ResNet56():
    """
    Creates a ResNet-56 Model for CIFAR10
            (n=9)
    Returns:
        (ResNet)
    """
    # 6n + 2 = 6*9 + 2 = 44
    return ResNet(ResidualBlock, [9, 9, 9])


def ResNet110():
    """
    Creates a ResNet-110 Model for CIFAR10
            (n=18)
    Returns:
        (ResNet)
    """
    # 6n + 2 = 6*18 + 2 = 110
    return ResNet(ResidualBlock, [18, 18, 18])


models = {
    "ResNet20": ResNet20(),
    "ResNet32": ResNet32(),
    "ResNet44": ResNet44(),
    "ResNet56": ResNet56(),
    "ResNet110": ResNet110(),
}


def load_data(path):
    """Load Data from torchvision.datasets CIFAR10

    Args:
        path (str): stored data path (PATH)
    """
    # Data augmentation as specified in Sec 4.2
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),  # CIFAR-10 mean
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    trainset = datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform
    )
    testset = datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform
    )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return trainset, testset, trainloader, testloader


def train_model(model, device, loader, criterion, optimizer, scheduler, writer):
    """
    Perform the training loop on (model)
    """
    n_steps = len(loader)
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward + update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log metrics
            epoch_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            epoch_correct += (preds == labels).sum().item()

            if (i + 1) % 100 == 0:
                avg_loss = epoch_loss / (i + 1)
                avg_acc = epoch_correct / (i + 1) * batch_size
                step = epoch * n_steps + i
                writer.add_scalar("Train/Loss", avg_loss, step)
                writer.add_scalar("Train/accuracy", avg_acc, step)

        epoch_avg_loss = epoch_loss / n_steps
        epoch_avg_acc = epoch_correct / (n_steps * batch_size)
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Epoch/Loss", epoch_avg_loss, epoch + 1)
        writer.add_scalar("Epoch/Accuracy", epoch_avg_acc, epoch + 1)
        writer.add_scalar("Train/Learning_Rate", current_lr, epoch + 1)

        # step scheduler based on epochs
        scheduler.step()


def evaluate(model, device, loader, criterion, writer):
    """
    Perform the testing loop on (model)
    Returns:
        accuracy (float)
    """
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        # Testing loop
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Get prediction
            outputs = model(images)

            # Log accuracy
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    writer.add_scalar("Test/Loss", avg_loss)
    writer.add_scalar("Test/Accuracy", accuracy)

    return accuracy


def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    p.add_argument("-e", "--epochs", type=int, default=128, help="number of epochs")
    p.add_argument(
        "modelname",
        type=str,
        choices=list(models.keys()),
        help="choose the a ResNet model to train",
    )

    return p.parse_args()


if __name__ == "__main__":
    if sys.version_info < (3, 11, 0):
        sys.stderr.write("You need python 3.11 or later to run this\n")
        sys.exit(1)
    try:
        args = cmdline_args()
        print(args)
        epochs, model = args.epochs, args.modelname
        n_epochs = epochs
        output = f"cifar10_{model}_{epochs}"

        # Set GPU if is_available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        writer = SummaryWriter(f"runs/CIFAR10/{args.modelname}|{n_epochs}")

        _, _, trainloader, testloader = load_data(DATAPATH)

        net = models[model].to(device)
        print(f"{model}:")

        # Parameter count
        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"\n{model} Total Trainable Parameters: {total_params:,}")

        # loss, optimizer and scheduler exactly as specified in Sec 4.2 (by epoch)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4
        )
        # 390 steps / epoch
        # 32k/390 = 82 | 48k/390 = 123
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[82, 123], gamma=0.1)

        # ensure model path exists
        os.makedirs(MODELPATH, exist_ok=True)

        start = time.time()
        train_model(net, device, trainloader, criterion, optimizer, scheduler, writer)
        end = time.time()
        final_accuracy = evaluate(net, device, testloader, criterion, writer)
        print(
            f"Final test accuracy: [{final_accuracy:.4f}] | Time take: [{end - start:.4f}]"
        )
        iterations_per_epoch = math.ceil(50000 / batch_size)  # ~391
        total_iterations = iterations_per_epoch * n_epochs  # 391 * 64 = 25,024
        print(
            f"Epochs: [{n_epochs}] | Iterations: [{iterations_per_epoch} / {total_iterations}]"
        )

        save_path = os.path.join(MODELPATH, f"{output}.pth")
        torch.save(net.state_dict(), save_path)
        print(f"Model state_dict saved to {save_path}")

        writer.close()
    except Exception:
        print("Usage $python src/cifar10/train_cifar.py modelname [epochs] [output]")
