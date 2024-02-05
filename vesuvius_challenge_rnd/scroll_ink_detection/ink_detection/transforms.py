from typing import Optional, Union

from collections.abc import Sequence

import torch
import torch.nn as nn


class RandFlip(nn.Module):
    def __init__(self, p: float = 0.5, spatial_axis: Sequence[int] | int | None = None):
        """
        Initialize the RandFlip module for single images.

        Parameters:
        p (float): Probability of flipping an image.
        spatial_axis (Sequence[int] | int | None): Spatial axes along which to flip.
        """
        super().__init__()
        self.p = p
        self.spatial_axis = spatial_axis

    def normalize_axis(self, num_dims: int) -> list[int]:
        """Normalize the spatial_axis to always be a list of positive integers."""
        if self.spatial_axis is None:
            return list(range(num_dims))
        elif isinstance(self.spatial_axis, int):
            return [self.spatial_axis % num_dims]
        else:
            return [axis % num_dims for axis in self.spatial_axis]

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Perform the random flip operation on a single image.

        Parameters:
        img (Tensor): channel first array, must have shape: (num_channels, H[, W, ..., ]),

        Returns:
        Tensor: Augmented tensor.
        """
        should_flip = torch.rand(1).item() < self.p
        if should_flip:
            spatial_dims_to_flip = self.normalize_axis(img.dim())
            img = torch.flip(img, spatial_dims_to_flip)

        return img
