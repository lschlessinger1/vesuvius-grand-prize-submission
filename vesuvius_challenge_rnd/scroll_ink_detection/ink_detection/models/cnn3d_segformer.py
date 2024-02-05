from typing import Literal

import torch.nn as nn
from transformers import SegformerForSemanticSegmentation


def create_cnn3d_segformer_model(model_size: Literal["b3", "b4", "b5"]) -> nn.Module:
    if model_size == "b3":
        return CNN3DSegformer()
    elif model_size == "b4":
        return CNN3DSegformerB4()
    elif model_size == "b5":
        return CNN3DSegformerB5()
    raise ValueError(f"Unknown model size {model_size}. Expected b3, b4, or b5.")


class CNN3DSegformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))

        self.xy_encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b3", num_labels=1, ignore_mismatched_sizes=True, num_channels=32
        )
        self.upscaler1 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis=2)[0]
        output = self.xy_encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output


class CNN3DSegformerB4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))

        self.xy_encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b4",
            num_labels=1,
            ignore_mismatched_sizes=True,
            num_channels=32,
        )
        self.upscaler1 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis=2)[0]
        output = self.xy_encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output


class CNN3DSegformerB5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))

        self.xy_encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b5",
            num_labels=1,
            ignore_mismatched_sizes=True,
            num_channels=32,
            classifier_dropout_prob=0.3,
            drop_path_rate=0.3,
            hidden_dropout_prob=0.3,
        )

        self.upscaler1 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis=2)[0]
        output = self.xy_encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output
