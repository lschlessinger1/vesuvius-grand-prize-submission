import torch.nn as nn
from monai.networks.nets import SwinUNETR
from transformers import SegformerForSemanticSegmentation


class SwinUNETRSegformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        img_size: tuple[int, int, int] = (32, 64, 64),
        dropout: float = 0.1,
        segformer_model_size: int = 5,
    ):
        super().__init__()
        swin_unetr_out_channels = 32
        self.encoder_3d = SwinUNETR(
            in_channels=in_channels,
            out_channels=swin_unetr_out_channels,
            img_size=img_size,
        )
        self.dropout = nn.Dropout2d(dropout)
        self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            f"nvidia/mit-b{segformer_model_size}",
            num_labels=1,
            ignore_mismatched_sizes=True,
            num_channels=swin_unetr_out_channels,
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
