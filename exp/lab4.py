from math import log
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# load multiclass data from sklearn
wine = datasets.load_wine()
X, y = wine.data, wine.target

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# turn data into tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.long))
y_test = torch.from_numpy(y_test.astype(np.long))

# dataloader
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=40, shuffle=True, num_workers=2)


# design the model
class WineModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(WineModel, self).__init__()
        # input layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # model layers
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation1(out)
        # if we were using manual CEL we need to do Softmax on output
        # remember pytorch CrossEntropyLoss does this for us
        return self.fc2(out)


model = WineModel(X_train.shape[1], hidden_size=10, num_classes=3)

n_epochs = 1000
learning_rate = 0.01

# setup loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(n_epochs):
    for batch_X, batch_y in dataloader:
        logits = model(batch_X)

        loss = criterion(logits, batch_y)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        print(f"epoch: {epoch + 1}, loss: {loss.item():.4f}")

# evaluate performance
with torch.no_grad():
    logits = model(X_test)
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == y_test).float().mean()
    print(f"Training accuracy: {accuracy.item() * 100:.2f}%")
