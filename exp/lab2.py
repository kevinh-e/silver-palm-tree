import matplotlib
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# prepare our data
X_numpy, Y_numpy = make_regression(
    n_samples=100, n_features=1, noise=10, random_state=1
)

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))

Y = Y.view(Y.shape[0], 1)

# model
n_samples, n_features = X.shape
model = nn.Linear(n_features, 1)

# loss and optimizer

learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 1000
for epoch in range(num_epochs):
    # forward
    y_pred = model.forward(X)

    # loss
    loss = criterion(y_pred, Y)

    # backpropagation
    loss.backward()

    # update weights
    optimizer.step()

    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"epoch: {epoch + 1}, loss = {loss.item():.4f}")

predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, "ro")

plt.plot(X_numpy, predicted, "b")

plt.show()
