# build a simple digit classification model
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, TensorDataset, dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get data
d = load_digits()
X, y = d.data, d.target
# scale X from 0-16 to 0-1 float for better training
# X = X / 16

input_size = 64
hidden_size = 100
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 0.001

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# convert to torch tensor
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.long))
y_test = torch.from_numpy(y_test.astype(np.long))

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# samples, labels = next(iter(train_loader))
# print(samples.shape, labels.shape)
#
# for i in range(6):
#     image = samples[i].view(8, 8)
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(image, cmap="gray")
# plt.show()
#


# design model
class DigitModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(DigitModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation1(out)
        out = self.fc2(out)
        return out


# init model
model = DigitModel(input_size, hidden_size, num_classes).to(device)
n_steps = len(train_loader)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        # imgs.Size = [100, 64]
        imgs = imgs.to(device)
        labels = labels.to(device)

        # forward
        prediction = model(imgs)

        # loss
        loss = criterion(prediction, labels)

        # update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i) % 5 == 0:
            print(
                f"epoch: {epoch + 1} / {num_epochs}, step: {i}/{n_steps}, loss: {loss.item():.8f}"
            )


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # torch.max returns value and index, index is class label
        _, predictions = torch.max(outputs, 1)
        # should be 100
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"accuracy = {acc}%")
