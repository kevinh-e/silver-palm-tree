import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import tempfile
from functools import partial
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from ray import data, tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from pathlib import Path

import time

PATH = "./models/CIFAR10/5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 16 * 5 * 5
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
num_classes = len(classes)


def load_data(data_dir="./data/cifar10"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


class CIFARModel(nn.Module):
    def __init__(self, l1, l2, do1):
        super(CIFARModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.af1 = nn.LeakyReLU()
        self.drop = nn.Dropout(do1)
        self.fc1 = nn.Linear(input_size, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, num_classes)

    def forward(self, x):
        out = self.pool(self.af1(self.conv1(x)))
        out = self.pool(self.af1(self.conv2(out)))
        out = out.view(-1, input_size)
        out = self.af1(self.fc1(out))
        out = self.drop(out)
        out = self.af1(self.fc2(out))
        return self.fc3(out)


def train_model(config, data_dir: str):
    model = CIFARModel(
        config["l1"],
        config["l2"],
        config["do1"],
    ).to(device)
    num_epochs = config["num_epochs"]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    trainset, _ = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    train_loader = DataLoader(
        train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=4
    )

    n_steps = len(train_loader)

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        running_correct = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()

            if i % 2000 == 1999:
                print(
                    f"epoch: [{epoch + 1}/{num_epochs}], step: [{i + 1}/{n_steps}], loss: {loss.item():.4f}, accuracy: {running_correct / 100}"
                )
                running_loss = 0.0
                running_correct = 0

        with torch.no_grad():
            val_loss = 0.0
            val_steps = 0
            n_correct = 0
            n_samples = 0
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.shape[0]
                n_correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

            checkpoint_data = {
                "epoch": epoch,
                "net_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                train.report(
                    {"loss": val_loss / val_steps, "accuracy": n_correct / n_samples},
                    checkpoint=checkpoint,
                )
    print("finished training")


def test_accuracy(model):
    _, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def main(num_samples=10, max_num_epochs=20):
    data_dir = os.path.abspath("./data/cifar10")
    load_data(data_dir)
    config = {
        "l1": tune.choice([2**i for i in range(9)]),
        "l2": tune.choice([2**i for i in range(9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "do1": tune.randn(0.2, 0.05),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "num_epochs": max_num_epochs,
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    result = tune.run(
        partial(train_model, data_dir=data_dir),
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        resources_per_trial={"cpu": 4, "gpu": 1 if device.type == "cuda" else 0},
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = CIFARModel(
        best_trial.config["l1"],
        best_trial.config["l2"],
        best_trial.config["do1"],
    )
    best_trained_model.to(device)

    best_checkpoint = result.get_best_checkpoint(
        trial=best_trial, metric="accuracy", mode="max"
    )
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        test_acc = test_accuracy(best_trained_model)
        print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    start = time.time()
    main(num_samples=10, max_num_epochs=20)
    end = time.time()
    print(f"Total time: {end - start:.4f} seconds")
