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

# Check for GPU availability and create a device object
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert data to tensors and move them to the selected device
X = torch.from_numpy(X_numpy.astype(np.float32)).to(device)
Y = torch.from_numpy(Y_numpy.astype(np.float32)).to(device)
Y = Y.view(Y.shape[0], 1)

# model
n_samples, n_features = X.shape
model = nn.Linear(n_features, 1).to(device)


# loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 1000
for epoch in range(num_epochs):
    # forward pass
    y_pred = model(X)

    # compute loss
    loss = criterion(y_pred, Y)

    # backpropagation
    loss.backward()

    # update weights
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")

# move predictions back to CPU for visualization
predicted = model(X).cpu().detach().numpy()
plt.plot(X_numpy, Y_numpy, "ro")  # red points: original data
plt.plot(X_numpy, predicted, "b")  # blue line: predictions
plt.show()
