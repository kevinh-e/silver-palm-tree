import torch.nn as nn
import torch.nn.functional as F

# -- Residual block --


class ResidualBlock(nn.Module):
    """
    Basic Residual Block following ResNet-18/34
    Folows the structure describe in the paper
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """
        Initialises the ResidualBlock

        Args:
            in_planes (int): Number of input channels.
            planes (int): Number of output channels for the convolutional layers.
            stride (int): Stride for the first convolution and the shortcut connection
                          (used for downsampling). Default is 1.
        """
        super(ResidualBlock, self).__init__()

        # First conv layer [3x3 kernel | output = planes | padding = 1]
        # No bias needed (batch norm)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        # Second conv layer [stride = 1]
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection using Sequential (will be identity function)
        self.shortcut = nn.Sequential()

        # if the dimensions changed readjust the dimensions using a 1x1 conv layer
        # self.expansion*planes is just planes in ResidualBlock
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        """
        Forward pass ResidualBlock
        """
        # [Conv -> BN -> ReLU -> Conv -> BN -> Shortcut -> ReLU]
        # F(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Add back the x to make F(x) + x
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# -- ResNet Model --


class ResNet(nn.Module):
    """
    ResNet model for COVID dataset
    """

    def __init__(self, block, num_blocks, num_classes=4):
        """
        Initialises ResNet

        Args:
            block (nn.Module): The type of residual block to use (ResidualBlock).
            num_blocks (list[int]): A list containing the number of residual blocks
                                    per stack (e.g., [3, 3, 3] for ResNet-20).
            num_classes (int): Number of output classes (default 10).
        """
        super(ResNet, self).__init__()
        self.in_planes = 16

        # First conv layer [3x3 kernel | input_channels=3 | output_planes=16 | padding = 1 | stride = 1]
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Create ResidualBlock layers
        # [256x256 -> 16 filters | num_blocks = num_blocks[0] | stride = 1] *no downsampling
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)

        # [128x128 -> 32 filters | num_blocks = num_blocks[1] | stride = 2] * downsampled 2x
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)

        # [64x64 -> 64 filters | num_blocks = num_blocks[2] | stride = 2] * downsampled 2x
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # [32x32 -> 128 filters | num_blocks = num_blocks[2] | stride = 2] * downsampled 2x
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)

        # [16x16 -> 256 filters | num_blocks = num_blocks[2] | stride = 2] * downsampled 2x
        self.layer5 = self._make_layer(block, 256, num_blocks[3], stride=2)

        # final fc layer to output 10 classes
        self.fc1 = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Creates a stack of ResidualBlocks

        Args:
            block (nn.Module): The type of residual block.
            planes (int): Number of output channels for the blocks in this stack.
            num_blocks (int): Number of blocks to stack.
            stride (int): Stride for the first block in the stack (used for downsampling).

        Returns:
            nn.Sequential: A sequential container of the residual blocks.
        """

        # Calculate the number of strides for each block
        # Subsequent strides do not downsample
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            # new in_planes after layer
            self.in_planes = planes * block.expansion

        # stack the layers
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through ResNet
        """

        # [Conv -> CN -> ReLU]
        out = F.relu(self.bn1(self.conv1(x)))

        # Residual stack
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        # Global average (as described in Sec 4.2)
        # kernel_size should be 512
        out = F.avg_pool2d(out, out.size()[3])

        # Final Linear FC layer to 4 classes
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out
