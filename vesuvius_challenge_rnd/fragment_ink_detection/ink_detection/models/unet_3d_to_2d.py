"""Adapted from https://github.com/wolny/pytorch-3dunet"""
from typing import Literal

import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.models.depth_pooling import (
    DepthPooling,
)
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.models.squeeze_and_excitation_3d import (
    ChannelSELayer3D,
    ChannelSpatialSELayer3D,
    SELayer3D,
    SpatialSELayer3D,
)


@dataclass
class UNetOutput:
    logits: torch.FloatTensor  # Classification scores for each pixel.
    encoder_features: tuple[torch.FloatTensor] | None = None


def get_number_of_learnable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def number_of_features_per_level(init_channel_number: int, num_levels: int) -> list[int]:
    return [init_channel_number * 2**k for k in range(num_levels)]


def get_class(class_name: str, modules: Iterable[str]) -> type:
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f"Unsupported dataset class: {class_name}")


def create_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, ...],
    order: str,
    num_groups: int,
    padding: int | tuple[int, ...],
    is3d: bool,
) -> list[tuple[str, nn.Module]]:
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): is3d (bool): if True use Conv3d, otherwise use Conv2d
    Return:
        list of tuple (name, module)
    """
    assert "c" in order, "Conv layer MUST be present"
    assert order[0] not in "rle", "Non-linearity cannot be the first operation in the layer"

    modules = []
    for i, char in enumerate(order):
        if char == "r":
            modules.append(("ReLU", nn.ReLU(inplace=True)))
        elif char == "l":
            modules.append(("LeakyReLU", nn.LeakyReLU(inplace=True)))
        elif char == "e":
            modules.append(("ELU", nn.ELU(inplace=True)))
        elif char == "c":
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ("g" in order or "b" in order)
            if is3d:
                conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
            else:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

            modules.append(("conv", conv))
        elif char == "g":
            is_before_conv = i < order.index("c")
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert (
                num_channels % num_groups == 0
            ), f"Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}"
            modules.append(
                (
                    "groupnorm",
                    nn.GroupNorm(num_groups=num_groups, num_channels=num_channels),
                )
            )
        elif char == "b":
            is_before_conv = i < order.index("c")
            if is3d:
                bn = nn.BatchNorm3d
            else:
                bn = nn.BatchNorm2d

            if is_before_conv:
                modules.append(("batchnorm", bn(in_channels)))
            else:
                modules.append(("batchnorm", bn(out_channels)))
        else:
            raise ValueError(
                f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']"
            )

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding
        is3d (bool): if True use Conv3d, otherwise use Conv2d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...] = 3,
        order: str = "gcr",
        num_groups: int = 8,
        padding: int | tuple[int, ...] = 1,
        is3d: bool = True,
    ):
        super().__init__()

        for name, module in create_conv(
            in_channels, out_channels, kernel_size, order, num_groups, padding, is3d
        ):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): if True use Conv3d instead of Conv2d layers
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder: bool,
        kernel_size: int | tuple[int, ...] = 3,
        order: str = "gcr",
        num_groups: int = 8,
        padding: int | tuple[int, ...] = 1,
        is3d: bool = True,
    ):
        super().__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module(
            "SingleConv1",
            SingleConv(
                conv1_in_channels,
                conv1_out_channels,
                kernel_size,
                order,
                num_groups,
                padding=padding,
                is3d=is3d,
            ),
        )
        # conv2
        self.add_module(
            "SingleConv2",
            SingleConv(
                conv2_in_channels,
                conv2_out_channels,
                kernel_size,
                order,
                num_groups,
                padding=padding,
                is3d=is3d,
            ),
        )


class ResNetBlock(nn.Module):
    """
    Residual block that can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...] = 3,
        order: str = "cge",
        num_groups: int = 8,
        is3d: bool = True,
        **kwargs,
    ):
        super().__init__()

        if in_channels != out_channels:
            # conv1x1 for increasing the number of channels
            if is3d:
                self.conv1 = nn.Conv3d(in_channels, out_channels, 1)
            else:
                self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv1 = nn.Identity()

        # residual block
        self.conv2 = SingleConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            order=order,
            num_groups=num_groups,
            is3d=is3d,
        )
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in "rel":
            n_order = n_order.replace(c, "")
        self.conv3 = SingleConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            order=n_order,
            num_groups=num_groups,
            is3d=is3d,
        )

        # create non-linearity separately
        if "l" in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif "e" in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        # apply first convolution to bring the number of channels to out_channels
        residual = self.conv1(x)

        # residual block
        out = self.conv2(residual)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class ResNetBlockSE(ResNetBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        order: str = "cge",
        num_groups: int = 8,
        se_module: str = "scse",
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            order=order,
            num_groups=num_groups,
            **kwargs,
        )
        assert se_module in ["scse", "cse", "sse"]
        if se_module == "scse":
            self.se_module = ChannelSpatialSELayer3D(num_channels=out_channels, reduction_ratio=1)
        elif se_module == "cse":
            self.se_module = ChannelSELayer3D(num_channels=out_channels, reduction_ratio=1)
        elif se_module == "sse":
            self.se_module = SpatialSELayer3D(num_channels=out_channels)

    def forward(self, x):
        out = super().forward(x)
        out = self.se_module(out)
        return out


class EncoderBlock(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    from the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a basic module (DoubleConv or ResNetBlock).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): use 3d or 2d convolutions/pooling operation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int | tuple[int, ...] = 3,
        apply_pooling: bool = True,
        pool_kernel_size: int | tuple[int, ...] = 2,
        pool_type: str = "max",
        basic_module: type[nn.Module] = DoubleConv,
        conv_layer_order: str = "gcr",
        num_groups: int = 8,
        padding: int | tuple[int, ...] = 1,
        is3d: bool = True,
    ):
        super().__init__()
        assert pool_type in ["max", "avg"]
        if apply_pooling:
            if pool_type == "max":
                if is3d:
                    self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)
            else:
                if is3d:
                    self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.AvgPool2d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(
            in_channels,
            out_channels,
            encoder=True,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
            padding=padding,
            is3d=is3d,
        )

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


def create_encoder_blocks(
    in_channels: int,
    f_maps: int | tuple[int, ...],
    basic_module: type[nn.Module],
    conv_kernel_size: int | tuple[int, ...],
    conv_padding: int | tuple[int, ...],
    layer_order: str,
    num_groups: int,
    pool_kernel_size: int | tuple[int, ...],
    is3d: bool,
):
    # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            # apply conv_coord only in the first encoder if any
            encoder = EncoderBlock(
                in_channels,
                out_feature_num,
                apply_pooling=False,  # skip pooling in the firs encoder
                basic_module=basic_module,
                conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size,
                num_groups=num_groups,
                padding=conv_padding,
                is3d=is3d,
            )
        else:
            encoder = EncoderBlock(
                f_maps[i - 1],
                out_feature_num,
                basic_module=basic_module,
                conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size,
                num_groups=num_groups,
                pool_kernel_size=pool_kernel_size,
                padding=conv_padding,
                is3d=is3d,
            )

        encoders.append(encoder)

    return nn.ModuleList(encoders)


class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample: callable):
        super().__init__()
        self.upsample = upsample

    def forward(self, encoder_features: Tensor, x: Tensor) -> Tensor:
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode: str = "nearest"):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x: Tensor, size: int | None, mode: str) -> Tensor:
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True

    """

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        kernel_size: int | tuple[int, ...] = 3,
        scale_factor: int | tuple[int, ...] = (2, 2, 2),
    ):
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        upsample = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=scale_factor,
            padding=1,
        )
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x: Tensor, size: int | None) -> Tensor:
        return x


class DecoderBlock(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation)
    followed by a basic module (DoubleConv or ResNetBlock).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upsample (bool): should the input be upsampled
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int | tuple[int, ...] = 3,
        scale_factor: tuple[int, ...] = (2, 2, 2),
        basic_module: type[nn.Module] = DoubleConv,
        conv_layer_order: str = "gcr",
        num_groups: int = 8,
        mode: str = "nearest",
        padding: int | tuple = 1,
        upsample: bool = True,
        is3d: bool = True,
    ):
        super().__init__()

        if upsample:
            if basic_module == DoubleConv:
                # if DoubleConv is the basic_module use interpolation for upsampling and concatenation joining
                self.upsampling = InterpolateUpsampling(mode=mode)
                # concat joining
                self.joining = partial(self._joining, concat=True)
            else:
                # if basic_module=ResNetBlock use transposed convolution upsampling and summation joining
                self.upsampling = TransposeConvUpsampling(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=conv_kernel_size,
                    scale_factor=scale_factor,
                )
                # sum joining
                self.joining = partial(self._joining, concat=False)
                # adapt the number of in_channels for the ResNetBlock
                in_channels = out_channels
        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, concat=True)

        self.basic_module = basic_module(
            in_channels,
            out_channels,
            encoder=False,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
            padding=padding,
            is3d=is3d,
        )

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


def create_decoder_blocks(
    f_maps: int | tuple[int, ...],
    basic_module: type[nn.Module],
    conv_kernel_size: int | tuple[int, ...],
    conv_padding: int | tuple[int, ...],
    layer_order: str,
    num_groups: int,
    is3d: bool,
):
    # create decoder path consisting of the Decoder modules. The length of the decoder list is equal to `len(f_maps) - 1`
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv:
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        decoder = DecoderBlock(
            in_feature_num,
            out_feature_num,
            basic_module=basic_module,
            conv_layer_order=layer_order,
            conv_kernel_size=conv_kernel_size,
            num_groups=num_groups,
            padding=conv_padding,
            is3d=is3d,
        )
        decoders.append(decoder)
    return nn.ModuleList(decoders)


class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution,
            otherwise apply nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True and the model is in eval mode, Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        is3d_encoder (bool): if True the model uses a 3D encoder, otherwise 2D, default: True
        is3d_decoder (bool): if True the model uses a 3D decoder, otherwise 2D, default: False
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        final_sigmoid: bool,
        basic_module: type[nn.Module],
        f_maps: int | tuple[int, ...] = 64,
        layer_order: str = "gcr",
        num_groups: int = 8,
        num_levels: int = 4,
        is_segmentation: bool = True,
        conv_kernel_size: int | tuple[int, ...] = 3,
        pool_kernel_size: int | tuple[int, ...] = 2,
        conv_padding: int | tuple[int, ...] = 1,
        is3d_encoder: bool = True,
        is3d_decoder: bool = False,
        output_features: bool | None = None,
        return_dict: bool | None = True,
    ):
        super().__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if "g" in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        self._f_maps = f_maps

        # create encoder path
        self.encoder_blocks = create_encoder_blocks(
            in_channels,
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            layer_order,
            num_groups,
            pool_kernel_size,
            is3d_encoder,
        )
        self.encoder_blocks.insert(
            0, nn.Identity()
        )  # Add this to be compatible with segmentation_models.pytorch.

        # create decoder path
        self.decoder_blocks = create_decoder_blocks(
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            layer_order,
            num_groups,
            is3d_decoder,
        )

        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        if is3d_decoder:
            self.classifier = nn.Conv3d(f_maps[0], out_channels, 1)
        else:
            self.classifier = nn.Conv2d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

        self.output_features = output_features
        self.use_return_dict = return_dict

    def encode(self, x: Tensor) -> list[Tensor]:
        stages = self.encoder_blocks

        features = []
        for encoder_block in stages:
            x = encoder_block(x)
            features.append(x)

        return features

    def decode(self, features: list[Tensor]) -> Tensor:
        # stages = self.decoder_blocks
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = head
        for decoder_block, skip in zip(self.decoder_blocks, skips):
            x = decoder_block(skip, x)
        return x

    def forward(
        self, x: Tensor, output_features: bool | None = None, return_dict: bool | None = None
    ) -> UNetOutput | tuple:
        output_features = output_features if output_features is not None else self.output_features
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        features = self.encode(x)
        x = self.decode(features)

        x = self.classifier(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction.
        # During training the network outputs logits
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        encoder_features = features if output_features else None

        if not return_dict:
            return tuple(v for v in [x, encoder_features] if v is not None)

        return UNetOutput(logits=x, encoder_features=encoder_features)


class UNet3Dto2D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        final_sigmoid: bool = False,
        f_maps: int | tuple[int, ...] = 64,
        layer_order: str = "gcr",
        num_groups: int = 8,
        num_levels: int = 4,
        is_segmentation: bool = False,
        conv_padding: int | tuple[int, ...] = 1,
        se_type_str: str | None = "CSE3D",
        reduction_ratio: int = 2,
        depth_dropout: float = 0.0,
        pool_fn: Literal["mean", "max"] = "mean",
        output_features: bool | None = None,
        return_dict: bool | None = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=DoubleConv,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            is3d_encoder=True,
            is3d_decoder=False,
            output_features=output_features,
            return_dict=return_dict,
        )

        # Pool along the slice/depth dimension.
        self.depth_pooler = DepthPooling(
            len(self.encoder_blocks) - 1,
            self._f_maps[::-1],
            slice_dim=2,
            depth_dropout=depth_dropout,
            pool_fn=pool_fn,
            se_type=SELayer3D[se_type_str] if se_type_str is not None else None,
            reduction_ratio=reduction_ratio,
        )

    def decode(self, features: list[Tensor]) -> Tensor:
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        # Pool along slice/depth dimension.
        features = self.depth_pooler(features)

        head = features[0]
        skips = features[1:]

        x = head
        for decoder_block, skip in zip(self.decoder_blocks, skips):
            x = decoder_block(skip, x)
        return x


def compute_unet_3d_to_2d_encoder_chwd(
    patch_depth: int,
    patch_height: int,
    patch_width: int,
    f_maps: int,
    encoder_level: int,
    in_channels: int,
) -> tuple[int, int, int, int]:
    """Computes the dimensions and channels at a given encoder level for a 3D U-Net.

    Args:
        patch_depth (int): Depth of the input 3D patch.
        patch_height (int): Height of the input 3D patch.
        patch_width (int): Width of the input 3D patch.
        f_maps (int): Number of feature maps at the first encoder level.
        encoder_level (int): The level of the encoder layer for which the dimensions are to be calculated.
        in_channels (int): Number of input channels.

    Returns:
        tuple[int, int, int, int]: Returns a tuple containing the number of output channels, depth, height, and width at
         the given encoder level.

    Example:
        >>> compute_unet_3d_to_2d_encoder_chwd(10, 256, 256, 32, 2, 1)
        (128, 2, 64, 64)
    """
    if encoder_level == 0:
        return in_channels, patch_depth, patch_height, patch_width

    depth = encoder_level - 1
    spatial_reduction_factor = 2**depth
    c_out = f_maps * spatial_reduction_factor
    d_out = patch_depth // spatial_reduction_factor
    h_out = patch_height // spatial_reduction_factor
    w_out = patch_width // spatial_reduction_factor
    return c_out, d_out, h_out, w_out
