import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


# 0) prepare data and data loader
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=666
)

n_samples, n_features = X.shape


sc = StandardScaler()
# scale data
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# turn into tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# flatten the outputs
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)


# 1) design model:
# f = wx + b, add sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, in_features):
        super(LogisticRegression, self).__init__()
        # the layers in your model, in this case only 1 linear and 1 sigmoid, sigmoid is provided
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        y_pred = torch.sigmoid(y_pred)
        return y_pred


model = LogisticRegression(n_features)


# loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_epochs = 100

# training loop
for epoch in range(n_epochs):
    for batch_X, batch_y in dataloader:
        # forwads
        y_pred = model(batch_X)

        # loss
        loss = criterion(y_pred, batch_y)

        # gradients
        loss.backward()

        # update weights
        optimizer.step()

        # zero gradients
        optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"epoch: {epoch + 1}, loss: {loss.item():.4f}")

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"accuracy = {acc:.4f}")
