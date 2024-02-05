from typing import Literal

from functools import partial

import torch
import torch.nn as nn

from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.models.squeeze_and_excitation_3d import (
    SELayer3D,
    se_layer3d_factory,
)


class DepthPoolingBlock(nn.Module):
    """Block for pooling along the depth dimension with optional squeeze-and-excitation layer and dropout.

    Args:
        input_channels (int): Number of input channels.
        se_type (SELayer3D | None, optional): Squeeze-and-excitation layer type. Defaults to SELayer3D.CSSE3D.
        reduction_ratio (int, optional): Reduction ratio for squeeze-and-excitation layer. Defaults to 2.
        dim (int, optional): Slice/depth dimension. Defaults to 4.
        depth_dropout (float, optional): 3D dropout rate. Defaults to 0.0.
        pool_fn (Literal["mean", "max"], optional): Pooling function to use along the depth dimension. Defaults to "mean".
    """

    def __init__(
        self,
        input_channels: int,
        se_type: SELayer3D | None = SELayer3D.CSSE3D,
        reduction_ratio: int = 2,
        dim: int = 2,
        depth_dropout: float = 0.0,
        pool_fn: Literal["mean", "max", "attention"] = "mean",
        depth: int | None = None,
        height: int | None = None,
        width: int | None = None,
    ):
        super().__init__()
        self.dim = dim  # Slice/depth dimension.

        if se_type is not None:
            se_layer = se_layer3d_factory(
                se_type, num_channels=input_channels, reduction_ratio=reduction_ratio
            )
        else:
            se_layer = nn.Identity()
        self.se = se_layer

        self.dropout = nn.Dropout3d(p=depth_dropout)

        depth_pool_fn_kwargs = {"dim": self.dim, "keepdim": False}
        if pool_fn == "mean":
            self.depth_pool_fn = partial(torch.mean, **depth_pool_fn_kwargs)
        elif pool_fn == "max":
            max_partial = partial(torch.max, **depth_pool_fn_kwargs)
            # We must take the first return value because that contains the max values.
            self.depth_pool_fn = lambda x: max_partial(x)[0]
        elif pool_fn == "attention":
            if depth is None or height is None or width is None:
                raise ValueError("Depth, height and width are all required. Found None.")
            self.depth_pool_fn = AttentionPool(depth, height, width, dim=self.dim)
        else:
            raise ValueError("Only 'max', 'mean', and 'attention' pooling are supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool along the depth dimension.

        Args:
            x (torch.Tensor): N x C x D x H x W tensor.

        Returns:
            torch.Tensor: N x C x H x W tensor.
        """
        x = self.se(x)
        x = self.dropout(x)
        x = self.depth_pool_fn(x)
        return x


class DepthPooling(nn.Module):
    """Layer for depth pooling across multiple layers.

    Args:
        num_layers (int): Number of layers to pool across.
        input_channels (list[int]): List of input channels for each layer.
        slice_dim (int, optional): Slice/depth dimension. Defaults to 4.
        depth_dropout (float, optional): 3D dropout rate. Defaults to 0.0.
        pool_fn (Literal["mean", "max"], optional): Pooling function to use along the depth dimension. Defaults to "mean".
        se_type (SELayer3D | None, optional): Squeeze-and-excitation layer type. Defaults to SELayer3D.CSSE3D.
        reduction_ratio (int, optional): Reduction ratio for squeeze-and-excitation layer. Defaults to 2.
    """

    def __init__(
        self,
        num_layers: int,
        input_channels: list[int],
        slice_dim: int = 2,
        depth_dropout: float = 0.0,
        pool_fn: Literal["mean", "max", "attention"] = "mean",
        se_type: SELayer3D | None = SELayer3D.CSSE3D,
        reduction_ratio: int = 2,
        depths: list[int] | None = None,
        heights: list[int] | None = None,
        widths: list[int] | None = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.slice_dim = slice_dim

        if depths is None:
            depths = [None] * self.num_layers
        if heights is None:
            heights = [None] * self.num_layers
        if widths is None:
            widths = [None] * self.num_layers
        self.layers = nn.ModuleList(
            DepthPoolingBlock(
                input_channels=c,
                se_type=se_type,
                reduction_ratio=reduction_ratio,
                depth_dropout=depth_dropout,
                pool_fn=pool_fn,
                dim=self.slice_dim,
                depth=d,
                height=h,
                width=w,
            )
            for _, c, d, h, w in zip(
                range(self.num_layers), input_channels, depths, heights, widths
            )
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply depth pooling to the given list of features.

        Args:
            features (list[torch.Tensor]): List of feature tensors.

        Returns:
            list[torch.Tensor]: List of depth-pooled feature tensors.

        Raises:
            ValueError: If the number of features does not match the number of layers.
        """
        if len(features) != self.num_layers:
            raise ValueError(
                f"Expected to find the same number of features as layers ({self.num_layers}). Found {len(features)} features."
            )

        features_out = []
        for x, layer in zip(features, self.layers):
            features_out.append(layer(x))
        return features_out


class AttentionPool(torch.nn.Module):
    def __init__(self, depth: int, height: int, width: int, dim: int = 2):
        super().__init__()
        self.dim = dim
        self.attention_weights = nn.Parameter(torch.ones(1, 1, depth, height, width))
        self.softmax = nn.Softmax(dim=self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply softmax along the depth dimension to obtain attention weights
        attention_weights = self.softmax(self.attention_weights)
        # Perform attention pooling by multiplying the attention weights with the input tensor
        pooled_output = torch.mul(attention_weights, x)
        # Sum the pooled output along the depth dimension
        pooled_output = torch.sum(pooled_output, dim=self.dim)
        return pooled_output
