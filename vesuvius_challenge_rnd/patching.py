def patch_index_to_pixel_position(
    patch_i: int, patch_j: int, patch_shape: tuple[int, int], patch_stride: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Convert the patch index to the top-left and bottom right pixel positions.

    Args:
        patch_i (int): The index of the patch along the vertical axis (i.e., row index).
        patch_j (int): The index of the patch along the horizontal axis (i.e., column index).
        patch_shape (tuple[int, int]): The shape of the patch as (height, width).
        patch_stride (int): The stride used for moving between patches.

    Returns:
        tuple[tuple[int, int], tuple[int, int]]: The top-left (x0, y0) and bottom-right (x1, y1) pixel positions.
    """
    y0 = patch_i * patch_stride
    y1 = y0 + patch_shape[0]
    x0 = patch_j * patch_stride
    x1 = x0 + patch_shape[1]
    return (y0, x0), (y1, x1)
