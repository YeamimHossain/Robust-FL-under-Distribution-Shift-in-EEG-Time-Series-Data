"""
EEGNet -- a compact CNN designed specifically for EEG decoding.
"""

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_channels=64, n_times=512, n_classes=2,
                 F1=8, D=2, F2=16, dropout=0.25):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times

        # Block 1: temporal conv + depthwise spatial conv
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: separable conv
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                      groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Figure out the flattened size dynamically (robust to n_times changes)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self.block2(self.block1(dummy))
            flat_size = out.numel()

        self.classifier = nn.Linear(flat_size, n_classes)

    def forward(self, x):
        # x: (batch, n_channels, n_times) -> add "1 feature-map" dim
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    # smoke test
    model = EEGNet(n_channels=64, n_times=512)
    dummy = torch.randn(4, 64, 512)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (4, 2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_params}")
