import torch
from torch import nn
from .common_layers import ResidualBlock
from .darknet53 import Darknet53


class DetectorHead(nn.Module):
    anchors = [
        [[10, 13], [16, 30], [33, 23]],  # head_3, small objects
        [[30, 61], [62, 45], [59, 119]],  # head_2, med objects
        [[116, 90], [156, 198], [373, 326]]  # head_1, large objects
    ]

    def __init__(self, class_number: int):
        super().__init__()

        self.first_head_begin = nn.Sequential(*[
            ResidualBlock(1024),
            nn.Conv2d(1024, 512, 1, 1),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 512, 1, 1),
        ])
        # upsample inbetween for next head
        self.first_head_end = nn.Conv2d(512, 1024, 3, 1, 1)

        self.second_head_begin = nn.Sequential(*[
            nn.Upsample(None, 2, 'nearest')
        ])
        # concat inbetween
        self.second_head_mid = nn.Sequential(*[
            nn.Conv2d(1024, 512, 1, 1),
            ResidualBlock(512, add=False),
            ResidualBlock(512, add=False),
            nn.Conv2d(512, 256, 1, 1)
        ])
        # upsample inbetween for next head
        self.second_head_end = nn.Conv2d(256, 512, 3, 1, 1)

        self.third_head_begin = nn.Sequential(*[
            nn.Upsample(None, 2, 'nearest')
        ])
        # concat inbetween
        self.third_head_end = nn.Sequential(*[
            nn.Conv2d(512, 256, 1, 1),
            ResidualBlock(256, add=False),
            ResidualBlock(256, add=False),
            ResidualBlock(256, add=False)
        ])

        output_number_per_head = class_number + 5  # objectness, classes, x1, y1, bw, bh
        self.head1_conv = nn.Conv2d(1024, output_number_per_head * len(DetectorHead.anchors[0]), 1)
        self.head2_conv = nn.Conv2d(512, output_number_per_head * len(DetectorHead.anchors[1]), 1)
        self.head3_conv = nn.Conv2d(256, output_number_per_head * len(DetectorHead.anchors[2]), 1)

    def forward(self, x1, x2, x3):
        head1_mid = self.first_head_begin(x1)
        head1_out = self.first_head_end(head1_mid)

        head2_beg = self.second_head_begin(head1_mid)
        concat1   = torch.concat([head2_beg, x2], dim=1)
        head2_mid = self.second_head_mid(concat1)
        head2_out = self.second_head_end(head2_mid)

        head3_beg = self.third_head_begin(head2_mid)
        concat2   = torch.concat([head3_beg, x3], dim=1)
        head3_out = self.third_head_end(concat2)

        output_1 = self.head1_conv(head1_out)
        output_2 = self.head2_conv(head2_out)
        output_3 = self.head3_conv(head3_out)

        return [output_3, output_2, output_1]


class YoloV3(nn.Module):
    def __init__(self, class_number: int, backbone: Darknet53):
        super().__init__()

        self.backbone = backbone
        self.head = DetectorHead(class_number)

    def forward(self, x):
        self.backbone(x)
        return self.head(self.backbone.output_1, self.backbone.output_2, self.backbone.output_3)

if __name__=="__main__":
    m = YoloV3(80, Darknet53())
    m.forward(torch.rand((1, 3,416, 416)))
