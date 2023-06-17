from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, add=True):
        super().__init__()

        half_c = channels // 2

        self.conv1 = nn.Conv2d(channels, half_c, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(half_c)
        self.conv2 = nn.Conv2d(half_c, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.add = add

    def forward(self, x):
        res = x

        out = nn.functional.leaky_relu(self.bn1(self.conv1(x)))
        out = nn.functional.leaky_relu(self.bn2(self.conv2(out)))

        if self.add:
            out += res
        return out


class DownsamplerLayer(nn.Module):
    def __init__(self, in_c, out_c, nr_res_blocks):
        super().__init__()

        self.conv1 = nn.Sequential(
            *[nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(out_c), nn.LeakyReLU()])
        res_blocks = []
        for _ in range(nr_res_blocks):
            res_blocks.append(ResidualBlock(out_c))
        self.res_blocks = nn.Sequential(*res_blocks)
        self.intermediate_output = None

    def forward(self, x):
        self.intermediate_output = self.conv1(x)
        out = self.res_blocks(self.intermediate_output)

        return out
