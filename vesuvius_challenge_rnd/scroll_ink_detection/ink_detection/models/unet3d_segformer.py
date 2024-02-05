from typing import Literal

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.models.depth_pooling import (
    DepthPoolingBlock,
)
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.models.squeeze_and_excitation_3d import (
    SELayer3D,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.models.unet3d import UNet3D


class UNet3DSegformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        img_size: tuple[int, int, int] = (32, 64, 64),
        n_classes: int = 1,
        unet_feature_size: int = 32,
        unet_out_channels: int = 64,
        unet_module_type: str = "resnet_se",
        dropout: float = 0.1,
        segformer_model_size: int = 5,
        se_type_str: str | None = None,
        depth_pool_fn: Literal["mean", "max", "attention"] = "max",
    ):
        super().__init__()
        self.encoder_3d = UNet3D(
            in_channels=in_channels,
            out_channels=unet_out_channels,
            basic_module_type=unet_module_type,
            f_maps=unet_feature_size,
            num_levels=5,
            num_groups=8,
            is_segmentation=False,
        )
        self.pooler = DepthPoolingBlock(
            input_channels=unet_out_channels,
            se_type=SELayer3D[se_type_str] if se_type_str is not None else None,
            reduction_ratio=2,
            depth_dropout=0,
            pool_fn=depth_pool_fn,
            dim=2,
            depth=img_size[0],
            height=img_size[1],
            width=img_size[2],
        )
        self.dropout = nn.Dropout2d(dropout)
        self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            f"nvidia/mit-b{segformer_model_size}",
            num_labels=n_classes,
            ignore_mismatched_sizes=True,
            num_channels=unet_out_channels,
        )
        self.upscaler1 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=(4, 4), stride=2, padding=1
        )
        self.upscaler2 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=(4, 4), stride=2, padding=1
        )

    def forward(self, image: torch.FloatTensor) -> torch.FloatTensor:
        output = self.encoder_3d(image).logits
        output = self.pooler(output)
        output = self.dropout(output)
        output = self.encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output
