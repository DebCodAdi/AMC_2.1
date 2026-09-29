"""2D Spectrogram Convolutional Branch."""
from __future__ import annotations
import torch
import torch.nn as nn

class SpecBranch(nn.Module):
    """Processes 2D log-spectrogram slices.
    Input: (B, 1, 128, 128)
    Output: (B, out_dim) (default 256)
    """
    def __init__(self, in_chans: int = 1, out_dim: int = 256, use_timm: bool = True):
        super().__init__()
        self.use_timm = False
        if use_timm:
            try:
                import timm
                self.net = timm.create_model("efficientnet_b0", pretrained=True, in_chans=in_chans, num_classes=0)
                self.proj = nn.Linear(self.net.num_features, out_dim)
                self.use_timm = True
            except (ImportError, Exception):
                self.use_timm = False

        if not self.use_timm:
            # Built-in lightweight multi-scale 2D ConvNet
            self.net = nn.Sequential(
                nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.proj = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W)"""
        if x.ndim == 3:
            x = x.unsqueeze(1)
        feat = self.net(x)
        return self.proj(feat)
