import torch.nn as nn
import torch.nn.functional as F

from .module import MiTB5
from .decode_head import Decode_Head


class SegFormer(nn.Module):
    def __init__(
        self,
        num_classes=2
    ):
        super().__init__()

        self.backbone = MiTB5()

        self.decode_head = Decode_Head(
            in_channels=(64, 128, 320, 512),
            channels=256,
            num_classes=num_classes,
            dropout=0.1
        )

    def forward(self, x):
        input_size = x.shape[2:]

        features = self.backbone(x)

        logits = self.decode_head(features)

        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False
        )

        return logits