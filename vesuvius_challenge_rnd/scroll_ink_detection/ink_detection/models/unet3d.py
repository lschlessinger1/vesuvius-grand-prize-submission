from torch import FloatTensor

from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.models.unet_3d_to_2d import (
    AbstractUNet,
    DoubleConv,
    ResNetBlock,
    ResNetBlockSE,
)


class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.
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
        basic_module_type: str = "double_conv",
        output_features: bool | None = None,
        return_dict: bool | None = True,
    ):
        if basic_module_type == "double_conv":
            basic_module_cls = DoubleConv
        elif basic_module_type == "resnet":
            basic_module_cls = ResNetBlock
        elif basic_module_type == "resnet_se":
            basic_module_cls = ResNetBlockSE
        else:
            raise ValueError(
                f"Unknown basic_module_type: {basic_module_type}. Expected double_conv, resnet, or resnet_se."
            )
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=basic_module_cls,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            is3d_encoder=True,
            is3d_decoder=True,
            output_features=output_features,
            return_dict=return_dict,
        )

    def decode(self, features: list[FloatTensor]) -> FloatTensor:
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = head
        for decoder_block, skip in zip(self.decoder_blocks, skips):
            x = decoder_block(skip, x)
        return x
