from typing import Literal

import torch.nn as nn

from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.models.depth_pooling import (
    DepthPoolingBlock,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.models.crackformer import Crackformer


class CNN3Dto2dCrackformer(nn.Module):
    def __init__(
        self,
        z_extent: int,
        height: int,
        width: int,
        se_type_str: str | None = None,
        in_channels: int = 1,
        out_channels: int = 1,
        depth_pool_fn: Literal["mean", "max", "attention"] = "attention",
    ):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(
            in_channels, 4, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)
        )
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))

        self.depth_pooler = DepthPoolingBlock(
            input_channels=1,
            se_type=se_type_str,
            reduction_ratio=2,
            depth_dropout=0,
            pool_fn=depth_pool_fn,
            dim=2,
            depth=z_extent,
            height=height,
            width=width,
        )

        self.xy_encoder_2d = Crackformer(in_channels=32, out_channels=out_channels)

    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output)
        output = self.depth_pooler(output)
        output = self.xy_encoder_2d(output)
        return output
