import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import time

from torch.utils.tensorboard import SummaryWriter

PATH = "./models/CIFAR10/3"

start = time.time()

writer = SummaryWriter("runs/cifar10/3")

# set gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper parameters
num_epochs = 20
batch_size = 64
# tbd
input_size = 16 * 6 * 6
hidden_size1 = 128
hidden_size2 = 96
# learning rate gets changed by scheduler
learning_rate = 0.001

# import data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = datasets.CIFAR10(
    root="./data/cifar10", train=True, download=True, transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data/cifar10", train=False, download=True, transform=transform
)

# output classes
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

# data loader for batch training
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# tensorboard
example = iter(train_loader)
samples, labels = next(example)
print(f"{samples.shape}")
img_grid = torchvision.utils.make_grid(samples)
writer.add_image("cifar10_images", img_grid)


# design model
# class CIFARModel(nn.Module):
#     def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
#         super(CIFARModel, self).__init__()
#         # 3 color bits => 8
#         # [_, 3, 32, 32]
#
#         self.conv1 = nn.Conv2d(3, 8, 5)
#         # [_, 8, 28, 28]
#
#         # (2, 2) pool will half kernel size
#         self.pool = nn.MaxPool2d(2, 2)
#         # [_, 8, 14, 14]
#
#         self.conv2 = nn.Conv2d(8, 16, 3)
#         # [_, 16, 12, 12]
#         # [_, 16, 6, 6]
#
#         self.activation1 = nn.LeakyReLU()
#         # first fully connected linear layer
#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         self.fc3 = nn.Linear(hidden_size2, num_classes)
#
#     def forward(self, x):
#         # conv -> ReLu -> pool
#         out = self.pool(self.activation1(self.conv1(x)))
#         out = self.pool(self.activation1(self.conv2(out)))
#         # scale data for linear
#         out = out.view(-1, input_size)
#         out = self.activation1(self.fc1(out))
#         out = self.activation1(self.fc2(out))
#         # no activation at output
#         return self.fc3(out)


# init model and push to gpu
model = models.resnet18(pretrained=True).to(device)
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# tensorboard
# sample data
sample_input = samples.to(device)

writer.add_graph(model, sample_input)


# optimizer scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# training loop
n_steps = len(train_loader)


def train_model(model):
    # var for tensorboard
    running_loss = 0.0
    running_correct = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)

            # loss
            loss = criterion(outputs, labels)

            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:
                print(
                    f"epoch: [{epoch + 1}/{num_epochs}], step: [{i + 1}/{n_steps}], loss: {loss.item():.4f}"
                )
                writer.add_scalar(
                    "training_loss", running_loss / 100, epoch * n_steps + i
                )
                writer.add_scalar(
                    "accuracy", running_correct / 100, epoch * n_steps + i
                )
                writer.add_scalar(
                    "learning_rate", scheduler.get_last_lr()[0], epoch * n_steps + i
                )

                running_loss = 0.0
                running_correct = 0

        # step scheduler based on epochs
        scheduler.step()

        end = time.time()
        print(f"Training time: {end - start:.4f} seconds")

        # save model as state_dict
        torch.save(model.state_dict(), PATH)

    # determine model accuracy
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # this determines which class to pick (model thought was most likely)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.shape[0]
            n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy: {acc}%")


train_model(model)
