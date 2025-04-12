import torch

# create a tensor
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)


# perform an operation (forwards)
# y = 2x^2 + 3x + 1
def forward(x):
    return 2 * x**2 + 3 * x + 1


z = forward(x).sum()
z.backward()

print("Gradient=", x.grad)
