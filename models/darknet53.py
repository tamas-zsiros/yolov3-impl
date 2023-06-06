from .common_layers import DownsamplerLayer
from torch import nn

class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.b1    = nn.BatchNorm2d(32)
        self.down1 = DownsamplerLayer(32, 64, 1)
        self.down2 = DownsamplerLayer(64, 128, 2)
        self.down3 = DownsamplerLayer(128, 256, 8)
        self.down4 = DownsamplerLayer(256, 512, 8)
        self.down5 = DownsamplerLayer(512, 1024, 4)

        self.output_1 = None
        self.output_2 = None
        self.output_3 = None

    def forward(self, x):
        x = nn.functional.leaky_relu(self.b1(self.conv1(x)))
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)

        self.output_1 = x
        self.output_2 = self.down5.intermediate_output
        self.output_3 = self.down4.intermediate_output

        return x
    
