# models.py
import torch.nn as nn
from torchvision.models import vgg13_bn

class VGG13Mel(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # no ptretrained weights
        base = vgg13_bn(weights=None)

        # 1) change first conv layer: 3 -> 1 channel
        old_conv = base.features[0]  # Conv2d(3, 64, kernel_size=3, padding=1)
        base.features[0] = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        # 2) chage final classifier layer for 10 classes
        in_features = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_features, num_classes)

        self.backbone = base

    def forward(self, x):
        # x: [B, 1, 128, 173]
        return self.backbone(x)
