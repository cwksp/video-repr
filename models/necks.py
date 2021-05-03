import torch
import torch.nn as nn

from .models import register


@register('disentangle-pool3d')
class DisentanglePool3D(nn.Module):

    def __init__(self, in_dim, remap3d):
        super().__init__()
        self.remap3d = remap3d
        if remap3d:
            self.out_dim = in_dim
        else:
            self.out_dim = in_dim // 2

    def forward(self, inp):
        B, C, T, H, W = inp.shape

        x_sp = inp[:, :C // 2, ...]
        x_sp = x_sp.permute(0, 1, 3, 4, 2).view(B, C // 2, H, W, -1)
        x_sp = x_sp.max(dim=-1)[0]

        x_t = inp[:, C // 2:, ...]
        x_t = x_t.view(B, C // 2, T, -1)
        x_t = x_t.max(dim=-1)[0]

        if self.remap3d:
            x_sp = x_sp.view(B, C // 2, 1, H ,W).expand(B, C // 2, T, H, W)
            x_t = x_t.view(B, C // 2, T, 1, 1).expand(B, C // 2, T, H, W)
            x = torch.cat([x_sp, x_t], dim=1)
            return x
        else:
            return x_sp, x_t


@register('linear-neck')
class LinearNeck(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_dim, out_dim, 1)
        self.out_dim = in_dim

    def forward(self, x):
        return self.conv(x)
