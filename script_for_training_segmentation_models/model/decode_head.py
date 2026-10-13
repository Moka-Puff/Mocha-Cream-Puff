import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Decode_Head(nn.Module):
    def __init__(
        self,
        in_channels=(64, 128, 320, 512),
        channels=256,
        num_classes=2,
        dropout=0.1
    ):
        super().__init__()

        self.projections = nn.ModuleList([
            ConvBNReLU(c, channels)
            for c in in_channels
        ])

        self.fusion = ConvBNReLU(
            channels * 4,
            channels
        )

        self.dropout = nn.Dropout2d(dropout)

        self.classifier = nn.Conv2d(
            channels,
            num_classes,
            kernel_size=1
        )

    def forward(self, features):
        target_size = features[0].shape[2:]

        outputs = []

        for feature, projection in zip(
            features,
            self.projections
        ):
            x = projection(feature)

            if x.shape[2:] != target_size:
                x = F.interpolate(
                    x,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False
                )

            outputs.append(x)

        x = torch.cat(outputs, dim=1)

        x = self.fusion(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x