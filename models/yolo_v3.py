import torch
from torch import nn
from common_layers import ResidualBlock


class DetectorHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_head_begin = nn.Sequential(*[
            ResidualBlock(1024),
            nn.Conv2d(1024, 512, 1, 1),
            nn.Conv2d(512, 1024, 3, 1),
            nn.Conv2d(1024, 512, 1, 1),
        ])
        # upsample inbetween for next head
        self.first_head_end = nn.Conv2d(512, 1024, 3, 1)

        self.second_head_begin = nn.Sequential(*[
            nn.Conv2d(512, 256, 1, 1),
            nn.Upsample(None, 2, 'nearest')
        ])
        # concat inbetween
        self.second_head_mid = nn.Sequential(*[
            ResidualBlock(512, add=False),
            ResidualBlock(512, add=False),
            nn.Conv2d(512, 256, 1, 1)
        ])
        # upsample inbetween for next head
        self.second_head_end = nn.Conv2d(256, 512, 3, 1)

        self.third_head_begin = nn.Sequential(*[
            nn.Conv2d(256, 128, 1, 1),
            nn.Upsample(None, 2, 'nearest')
        ])
        # concat inbetween
        self.third_head_end = nn.Sequential(*[
            ResidualBlock(256, add=False),
            ResidualBlock(256, add=False),
            ResidualBlock(256, add=False)
        ])

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
