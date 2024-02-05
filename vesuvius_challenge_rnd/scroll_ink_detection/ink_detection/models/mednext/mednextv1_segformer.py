from typing import Literal

import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.models import create_mednext_v1


class MedNextV1Segformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        mednextv1_model_id: Literal["S", "B", "M", "L"] = "M",
        kernel_size: int = 3,
        deep_supervision: bool = False,
        dropout: float = 0.1,
        segformer_model_size: int = 3,
    ):
        super().__init__()
        self.encoder_3d = create_mednext_v1(
            num_input_channels=in_channels,
            num_classes=32,
            model_id=mednextv1_model_id,
            kernel_size=kernel_size,
            deep_supervision=deep_supervision,
        )
        self.dropout = nn.Dropout2d(dropout)
        self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            f"nvidia/mit-b{segformer_model_size}",
            num_labels=1,
            ignore_mismatched_sizes=True,
            num_channels=32,
        )
        self.upscaler1 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, image):
        output = self.encoder_3d(image)
        output = output.max(axis=2)[0]
        output = self.dropout(output)
        output = self.encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output
