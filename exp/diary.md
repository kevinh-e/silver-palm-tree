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

## CIFAR10 | ResNet

1.ResNet-20
    Test Loss: 0.4313, Test Accuracy: 0.8651
    Final test accuracy: [0.8651] | Time take: [806.1353]
    Epochs: [64] | Iterations: [391 / 25024]
    - Better than transfer learning !!!!! (slower though)
2.ResNet-20
    Test Loss: 0.3360, Test Accuracy: 0.9110
    Final test accuracy: [0.9110] | Time take: [1607.8371]
    Epochs: [128] | Iterations: [391 / 50048]
3.ResNet-44
    Test Loss: 0.3089, Test Accuracy: 0.9281
    Final test accuracy: [0.9281] | Time take: [3171.6705]
    Epochs: [128] | Iterations: [391 / 50048]
4.ResNet-110
    Test Loss: 0.2973, Test Accuracy: 0.9293
    Final test accuracy: [0.9293] | Time take: [9066.0921]
    Epochs: [156] | Iterations: [391 / 60996]

## COVID |ResNet

1. ResNet-20 | 5 layers | 128 Epochs

2. ResNet-20 | 4 layers | 64 Epochs

3. ResNet-30 | 4 layers | 96 Epochs | conv1 and conv2 downscale from 1x -> 2x
    - output size is 8x8
