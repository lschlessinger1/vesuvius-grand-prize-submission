from typing import Literal

import torch.nn as nn
from monai.networks.nets import UNETR
from transformers import SegformerForSemanticSegmentation

from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.models.depth_pooling import (
    DepthPoolingBlock,
)
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.models.squeeze_and_excitation_3d import (
    SELayer3D,
)


class UNETRSegformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 1,
        img_size: tuple[int, int, int] = (30, 64, 64),
        unetr_feature_size: int = 16,
        dropout: float = 0.1,
        segformer_model_size: int = 5,
        se_type_str: str | None = None,
        depth_pool_fn: Literal["mean", "max", "attention"] = "max",
    ):
        super().__init__()
        unetr_out_channels = 32
        self.encoder_3d = UNETR(
            in_channels=in_channels,
            out_channels=unetr_out_channels,
            img_size=img_size,
            feature_size=unetr_feature_size,
            conv_block=True,
        )
        self.pooler = DepthPoolingBlock(
            input_channels=unetr_out_channels,
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
            num_channels=unetr_out_channels,
        )
        self.upscaler1 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=(4, 4), stride=2, padding=1
        )
        self.upscaler2 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=(4, 4), stride=2, padding=1
        )

    def forward(self, image):
        output = self.encoder_3d(image)
        output = self.pooler(output)
        output = self.dropout(output)
        output = self.encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output
