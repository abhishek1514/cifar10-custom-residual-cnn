
import torch
import torch.nn as nn


# --------------------------------
# 
# --------------------------------

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=False):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        # Match input dimensions with output dimensions
        if in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        else:

            self.shortcut = nn.Identity()

        self.pool = (
            nn.MaxPool2d(kernel_size=2, stride=2)
            if downsample else nn.Identity()
        )

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity

        out = self.relu(out)

        out = self.pool(out)

        return out


# --------------------------------
# 2. CIFAR-10 CNN Version 2
# --------------------------------

class CIFAR10CNN_V2(nn.Module):

    def __init__(self):

        super(CIFAR10CNN_V2, self).__init__()

        # Block 1: 3 -> 32
        self.block1 = ResidualBlock(
            in_channels=3,
            out_channels=32,
            downsample=False
        )

        # Block 2: 32 -> 64
        self.block2 = ResidualBlock(
            in_channels=32,
            out_channels=64,
            downsample=True
        )

        # Block 3: 64 -> 128
        self.block3 = ResidualBlock(
            in_channels=64,
            out_channels=128,
            downsample=True
        )

        # Block 4: 128 -> 256
        self.block4 = ResidualBlock(
            in_channels=128,
            out_channels=256,
            downsample=True
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(p=0.4),

            nn.Linear(256, 10)

        )

    def forward(self, x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.global_pool(x)

        x = self.classifier(x)

        return x


# --------------------------------
# 3. Model Architecture Testing
# --------------------------------

if __name__ == "__main__":

    model = CIFAR10CNN_V2()

    dummy_input = torch.randn(64, 3, 32, 32)

    output = model(dummy_input)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\n========== CIFAR-10 CNN VERSION 2 ==========")

    print("Input Shape:", dummy_input.shape)

    print("Output Shape:", output.shape)

    print("Total Parameters:", total_params)

    print("Trainable Parameters:", trainable_params)

    print("\nArchitecture verification successful.")
