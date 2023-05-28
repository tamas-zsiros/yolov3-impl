import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        half_c = channels // 2

        self.conv1 = nn.Conv2d(channels, half_c, kernel_size=1)
        self.bn1   = nn.BatchNorm2d(half_c)
        self.conv2 = nn.Conv2d(half_c, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x

        out = nn.functional.leaky_relu(self.bn1(self.conv1(x)))
        out = nn.functional.leaky_relu(self.conv2(out)) 

        out += res
        return out

def createDownsamplingConv(in_c, out_c, nr_res_blocks):
    l = [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2), nn.BatchNorm2d(out_c), nn.LeakyReLU()]
    for _ in range(nr_res_blocks):
        l.append(ResidualBlock(out_c))
    return nn.Sequential(*l)

class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.b1    = nn.BatchNorm2d(32)
        self.down1 = createDownsamplingConv(32, 64, 1)
        self.down2 = createDownsamplingConv(64, 128, 2)
        self.down3 = createDownsamplingConv(128, 256, 8)
        self.down4 = createDownsamplingConv(256, 512, 8)
        self.down5 = createDownsamplingConv(512, 1024, 4)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.b1(self.conv1(x)))
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)

        return x
    
