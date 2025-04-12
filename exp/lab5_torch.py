import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

# enable gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if False else "cpu")

# hyper parameters
input_size = 784  # 28 x 28
hidden_size = 500
num_epochs = 10
batch_size = 600
learning_rate = 0.001

# import data
train_dataset = torchvision.datasets.EMNIST(
    root="./data",
    split="balanced",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

test_dataset = torchvision.datasets.EMNIST(
    root="./data", split="balanced", train=False, transform=transforms.ToTensor()
)

num_classes = len(test_dataset.classes)
num_classes = 47

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#
# examples = iter(train_loader)
# samples, labels = next(examples)
#
# print(samples.shape, labels.shape)
#
# for i in range(6):
#     sub = plt.subplot(2, 3, i + 1)
#     sub.set_title(f"{labels[i]}")
#     plt.imshow(samples[i][0], cmap="gray")
# plt.show()
#


# design model
class EMNISTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(EMNISTModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # LeakyReLU activation function
        self.activation1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation1(out)
        return self.fc2(out)


# instantiate model and move to GPU
model = EMNISTModel(input_size, hidden_size, num_classes).to(device)

# loss and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_steps = len(train_loader)

# training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # move and flatten images
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # forward
        prediction = model(images)

        # loss
        loss = criterion(prediction, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()

        if i % 100 == 0:
            print(
                f"epoch: {epoch + 1}, step: {i + 1} / {n_steps}, loss: {loss.item():.6f}"
            )

end = time.time()
print(f"Training time: {end - start:.4f} seconds")

# calculate accuracy
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        predictions = model(images)

        # value and index (class)
        _, predictions = torch.max(predictions, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

acc = 100.0 * n_correct / n_samples
print(f"Accuracy: {acc}%")
