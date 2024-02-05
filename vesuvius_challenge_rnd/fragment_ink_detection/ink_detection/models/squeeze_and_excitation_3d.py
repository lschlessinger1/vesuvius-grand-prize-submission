"""
Adapted from: https://github.com/ai-med/squeeze_and_excitation

3D Squeeze and Excitation Modules
*****************************
3D Extensions of the following 2D squeeze and excitation blocks:

    1. `Channel Squeeze and Excitation <https://arxiv.org/abs/1709.01507>`_
    2. `Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
    3. `Channel and Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_

New Project & Excite block, designed specifically for 3D inputs
    'quote'

    Coded by -- Anne-Marie Rickmann (https://github.com/arickm)
"""

from enum import Enum

import torch
from torch import nn as nn
from torch.nn import functional as F


class ChannelSELayer3D(nn.Module):
    """3D extension of the Squeeze-and-Excitation (SE) block for channel-wise squeezing.
    Described in:
    * Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507
    * Zhu et al., AnatomyNet, arXiv:1808.05238

    Args:
        num_channels (int): Number of input channels.
        reduction_ratio (int, optional): Factor by which the number of channels should be reduced. Defaults to 2.
    """

    def __init__(self, num_channels: int, reduction_ratio: int = 2):
        """
        :param num_channels: No. of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise squeeze and excitation.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, num_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor.
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """3D extension of SE block for spatial squeezing and channel-wise excitation.
    Described in:
    * Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018

    Args:
        num_channels (int): Number of input channels.
    """

    def __init__(self, num_channels: int):
        """
        :param num_channels: No of input channels

        """
        super().__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, input_tensor: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply spatial squeeze and channel-wise excitation.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, num_channels, D, H, W).
            weights (torch.Tensor | None, optional): Weights for few-shot learning. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """3D extension for concurrent spatial and channel squeeze & excitation.
    Described in:
    * Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579

    Args:
        num_channels (int): Number of input channels.
        reduction_ratio (int, optional): Factor by which the number of channels should be reduced. Defaults to 2.
    """

    def __init__(self, num_channels: int, reduction_ratio: int = 2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super().__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply concurrent spatial and channel squeeze & excitation.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, num_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor.
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class ProjectExciteLayer(nn.Module):
    """Project & Excite Module designed for 3D inputs. A new block designed specifically for 3D inputs.

    Args:
        num_channels (int): Number of input channels.
        reduction_ratio (int, optional): Factor by which the number of channels should be reduced. Defaults to 2.
    """

    def __init__(self, num_channels: int, reduction_ratio: int = 2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super().__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(
            in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1
        )
        self.conv_cT = nn.Conv3d(
            in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply the Project & Excite operation.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, num_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor.
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum(
            [
                squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
                squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
                squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1),
            ]
        )

        # Excitation:
        final_squeeze_tensor = self.sigmoid(
            self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor)))
        )
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        return output_tensor


class SELayer3D(Enum):
    """Enum restricting the type of SE Blocks available."""

    CSE3D = "CSE3D"
    SSE3D = "SSE3D"
    CSSE3D = "CSSE3D"
    PE = "PE"


def se_layer3d_factory(se_type: SELayer3D, num_channels: int, reduction_ratio: int = 2):
    """Factory method for creating a 3D Squeeze-and-Excitation layer.

    Args:
        se_type (SELayer3D): Type of SE layer to create.
        num_channels (int): Number of input channels to the SE layer.
        reduction_ratio (int, optional): Reduction ratio for the SE layer. Default is 2.

    Returns:
        nn.Module: An instance of the specified SE layer.

    Raises:
        ValueError: If an invalid value is provided for se_type.
    """
    if se_type == SELayer3D.CSE3D:
        return ChannelSELayer3D(num_channels, reduction_ratio)
    elif se_type == SELayer3D.SSE3D:
        return SpatialSELayer3D(num_channels)
    elif se_type == SELayer3D.CSSE3D:
        return ChannelSpatialSELayer3D(num_channels, reduction_ratio)
    elif se_type == SELayer3D.PE:
        return ProjectExciteLayer(num_channels, reduction_ratio)
    else:
        raise ValueError(f"Invalid value for se_type: {se_type}")
