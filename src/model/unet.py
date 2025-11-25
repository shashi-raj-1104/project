import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=4, base_channels=64):
        super().__init__()

        # Down
        self.down1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)

        # Mid
        self.mid = nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1)

        # Up
        self.up2 = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.up1 = nn.Conv2d(base_channels + base_channels, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = t.float().view(-1,1,1,1)

        x = x + t_emb

        # Encoder
        d1 = F.relu(self.down1(x))              # 128×128
        d2 = F.relu(self.down2(F.avg_pool2d(d1, 2)))  # 64×64

        # Middle
        m = F.relu(self.mid(d2))                # 64×64

        # Decoder
        u2 = F.interpolate(m, scale_factor=2)   # 128×128
        u2 = torch.cat([u2, d1], dim=1)         # concat skip connection

        out = self.up1(u2)
        return out
