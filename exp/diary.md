# Model record

## CIFAR10

1. base (66.88% | 82.5065)
2. changed hidden layer dimensions (65.34% | 82.8855)
    128 -> 96 -> 10
    265 -> 128 -> 10
3. resnet18 transfer learning (82.84% | 204.3899)
4. changed hidden layer and activation functions (68.44% | 92.1414)
    256 -> 120 -> 10
    added a second ReLu(0.02) on second conv and second linear layer
