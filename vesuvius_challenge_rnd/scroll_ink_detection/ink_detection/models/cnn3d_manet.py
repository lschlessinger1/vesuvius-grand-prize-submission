import segmentation_models_pytorch as smp
import torch.nn as nn


class CNN3DMANet(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 1):
        super().__init__()
        cnn3d_out_channels = 32
        self.conv3d_1 = nn.Conv3d(
            in_channels, 4, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)
        )
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(
            16, cnn3d_out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)
        )

        self.xy_encoder_2d = smp.MAnet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=cnn3d_out_channels,
            classes=n_classes,
        )

    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis=2)[0]
        output = self.xy_encoder_2d(output)
        return output
