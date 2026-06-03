import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Блок пространственного внимания CBAM (Channel + Spatial)
# -------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

# -------------------------------
# Многомасштабный блок внимания (MS-CBAM)
# -------------------------------
class MultiScaleCBAM(nn.Module):
    def __init__(self, channels, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.cbams = nn.ModuleList([CBAM(channels) for _ in scales])
        self.fuse = nn.Conv2d(channels * len(scales), channels, 1)

    def forward(self, x):
        feats = []
        for s, cbam in zip(self.scales, self.cbams):
            if s == 1:
                feats.append(cbam(x))
            else:
                pooled = F.avg_pool2d(x, kernel_size=s, stride=s)
                att = cbam(pooled)
                att_up = F.interpolate(att, size=x.shape[2:], mode='bilinear', align_corners=False)
                feats.append(x * att_up)
        out = torch.cat(feats, dim=1)
        out = self.fuse(out)
        return out

# -------------------------------
# HistColorNet (U-Net + MS-CBAM)
# -------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mp = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.mp(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2,
                        diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class HistColorNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, features=[64, 128, 256, 512]):
        super().__init__()
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.bottleneck = DoubleConv(features[3], features[3]*2)

        # Многомасштабные CBAM на каждом уровне энкодера
        self.mscbam1 = MultiScaleCBAM(features[0])
        self.mscbam2 = MultiScaleCBAM(features[1])
        self.mscbam3 = MultiScaleCBAM(features[2])
        self.mscbam4 = MultiScaleCBAM(features[3])

        self.up1 = Up(features[3]*2, features[3])
        self.up2 = Up(features[3], features[2])
        self.up3 = Up(features[2], features[1])
        self.up4 = Up(features[1], features[0])

        self.outc = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x1 = self.mscbam1(x1)
        x2 = self.down1(x1)
        x2 = self.mscbam2(x2)
        x3 = self.down2(x2)
        x3 = self.mscbam3(x3)
        x4 = self.down3(x3)
        x4 = self.mscbam4(x4)
        x5 = self.bottleneck(x4)
        # decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

# -------------------------------
# Базовая U-Net (без внимания) для сравнения
# -------------------------------
class UNetBaseline(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, features=[64, 128, 256, 512]):
        super().__init__()
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.bottleneck = DoubleConv(features[3], features[3]*2)
        self.up1 = Up(features[3]*2, features[3])
        self.up2 = Up(features[3], features[2])
        self.up3 = Up(features[2], features[1])
        self.up4 = Up(features[1], features[0])
        self.outc = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# -------------------------------
# Упрощённая ChromaGAN (только генератор)
# -------------------------------
class ChromaGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # упрощённая архитектура: U-Net подобная
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.bottleneck = DoubleConv(512, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.outc = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)