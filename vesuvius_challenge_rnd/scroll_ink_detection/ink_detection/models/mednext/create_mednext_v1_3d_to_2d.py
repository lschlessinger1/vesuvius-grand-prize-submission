from typing import Literal

from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.models.mednext.MedNextV1_3d_to_2d import (
    MedNeXt3dto2d,
)


def create_mednextv1_3d_to_2d_small(num_input_channels, num_classes, kernel_size=3, ds=False):
    return MedNeXt3dto2d(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=2,
        kernel_size=kernel_size,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
    )


def create_mednextv1_3d_to_2d_base(num_input_channels, num_classes, kernel_size=3, ds=False):
    return MedNeXt3dto2d(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        kernel_size=kernel_size,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
    )


def create_mednextv1_3d_to_2d_medium(num_input_channels, num_classes, kernel_size=3, ds=False):
    return MedNeXt3dto2d(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        kernel_size=kernel_size,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down=True,
        block_counts=[3, 4, 4, 4, 4, 4, 4, 4, 3],
        checkpoint_style="outside_block",
    )


def create_mednextv1_3d_to_2d_large(num_input_channels, num_classes, kernel_size=3, ds=False):
    return MedNeXt3dto2d(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        kernel_size=kernel_size,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down=True,
        block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        checkpoint_style="outside_block",
    )


def create_mednext_v1_3d_to_2d(
    num_input_channels: int,
    num_classes: int,
    model_id: Literal["S", "B", "M", "L"],
    kernel_size: int = 3,
    deep_supervision: bool = False,
) -> MedNeXt3dto2d:
    model_dict = {
        "S": create_mednextv1_3d_to_2d_small,
        "B": create_mednextv1_3d_to_2d_base,
        "M": create_mednextv1_3d_to_2d_medium,
        "L": create_mednextv1_3d_to_2d_large,
    }

    return model_dict[model_id](num_input_channels, num_classes, kernel_size, deep_supervision)
