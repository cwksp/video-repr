import torch.nn as nn

from .models import register


class PixelShuffle3d(nn.Module):

    def __init__(self, u):
        super().__init__()
        self.u = u

    def forward(self, x):
        B, C, T, H, W = x.shape
        u = self.u
        x = x.view(B, C // u**2, u, u, T, H, W)
        x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous() \
            .view(B, C // u**2, T, H * u, W * u)
        return x


@register('pixel-shuffle-upsampler')
class PixelShuffleUpsampler(nn.Sequential):
    def __init__(self, in_dim, base_dim, upsample_factors, out_dim=3):
        layers = []

        layers.append(
            nn.Conv3d(in_dim, base_dim, kernel_size=(1, 1, 1),
                      padding=(0, 0, 0))
        )

        for u in upsample_factors:
            layers.append(
                nn.Conv3d(base_dim, base_dim * u * u, kernel_size=(1, 1, 1),
                          padding=(0, 0, 0))
            )
            layers.append(nn.BatchNorm3d(base_dim * u * u))
            layers.append(nn.ReLU())
            layers.append(PixelShuffle3d(u))

        layers.append(
            nn.Conv3d(base_dim, out_dim, kernel_size=(1, 1, 1),
                      padding=(0, 0, 0))
        )

        super().__init__(*layers)
