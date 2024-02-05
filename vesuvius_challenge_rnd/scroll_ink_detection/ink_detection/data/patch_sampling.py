import numpy as np
from patchify import patchify

from vesuvius_challenge_rnd.patching import patch_index_to_pixel_position
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.util import pad_to_match


def find_valid_patch_positions(
    ink_mask: np.ndarray,
    non_ink_mask: np.ndarray,
    segment_mask: np.ndarray,
    patch_size: int,
    stride: int,
    min_non_ink_coverage_frac: float = 0.5,
    ink_thresh: float = 0.05,
) -> list[tuple[int, int, int, int]]:
    """
    Identifies valid positions for patches in a given area based on ink and non-ink regions.

    This function scans an image using a square patch of a specified size and stride, and
    determines valid positions where the patch meets certain criteria of ink and non-ink coverage.
    A position is considered valid if it either contains any part of the ink region or meets the
    minimum non-ink region coverage threshold.

    Args:
        ink_mask (np.ndarray): A binary mask indicating the ink region in the image.
        non_ink_mask (np.ndarray): A binary mask indicating the non-ink region in the image.
        segment_mask (np.ndarray): A binary mask indicating the segment region of the image.
        patch_size (int): The size of the square patch (e.g., 64 for a 64x64 patch).
        stride (int): The stride length for moving the patch across the image.
        min_non_ink_coverage_frac (float): The minimum required fraction of non-ink coverage in a patch
            for it to be considered valid. Defaults to 0.5.
        ink_thresh (float): The threshold for a value to be considered ink. Defaults to 0.05.


    Returns:
        list[tuple[tuple[int, int], tuple[int, int]]]: A list of tuples, each containing two tuples. The first tuple
        in each pair represents the top-left coordinates (x, y) of a valid patch, and the second tuple represents
        the bottom-right coordinates (x + patch_size, y + patch_size).
    """
    valid_positions = []
    rows, cols = ink_mask.shape
    total_pixels = patch_size * patch_size

    for y1 in range(0, rows - patch_size + 1, stride):
        y2 = y1 + patch_size
        for x1 in range(0, cols - patch_size + 1, stride):
            x2 = x1 + patch_size
            ink_patch = ink_mask[y1:y2, x1:x2]
            non_ink_patch = non_ink_mask[y1:y2, x1:x2]
            segment_patch = segment_mask[y1:y2, x1:x2]

            # Check for overlap with non-segment region.
            no_overlap_with_non_segment_region = not np.any(segment_patch == 0)

            # Check for presence of ink region
            contains_ink = np.any(ink_patch > ink_thresh)

            # Calculate non-ink region coverage as a fraction
            non_ink_coverage_frac = np.sum(non_ink_patch > 0) / total_pixels

            # Check if non-ink
            sufficient_non_ink = non_ink_coverage_frac >= min_non_ink_coverage_frac

            if no_overlap_with_non_segment_region:
                if contains_ink or sufficient_non_ink:
                    valid_positions.append((x1, y1, x2, y2))

    return valid_positions


def sample_all_patch_positions_for_segment(
    img_stack: np.ndarray,
    segment_mask: np.ndarray,
    size: int,
    tile_size: int,
    z_extent: int,
    stride: int,
):
    """Sample all possible patch positions for a segment within the segment mask."""
    x1_list = list(range(0, img_stack.shape[1] - tile_size + 1, stride))
    y1_list = list(range(0, img_stack.shape[0] - tile_size + 1, stride))

    valid_xyxys = []
    for a in y1_list:
        for b in x1_list:
            for yi in range(0, tile_size, size):
                for xi in range(0, tile_size, size):
                    y1 = a + yi
                    x1 = b + xi
                    y2 = y1 + size
                    x2 = x1 + size

                    segment_context_window = segment_mask[a : a + tile_size, b : b + tile_size]
                    no_overlap_with_non_segment_region = not np.any(segment_context_window == 0)
                    if no_overlap_with_non_segment_region:
                        valid_xyxys.append([x1, y1, x2, y2])
                        assert x2 - x1 == size
                        assert y2 - y1 == size
                        assert img_stack[y1:y2, x1:x2].shape == (
                            size,
                            size,
                            z_extent,
                        )
    return valid_xyxys


def oversample_train_patch_positions_for_segment(
    img_stack: np.ndarray,
    ink_mask: np.ndarray,
    segment_mask: np.ndarray,
    size: int,
    tile_size: int,
    z_extent: int,
    stride: int,
):
    x1_list = list(range(0, img_stack.shape[1] - tile_size + 1, stride))
    y1_list = list(range(0, img_stack.shape[0] - tile_size + 1, stride))

    train_positions = []
    for a in y1_list:
        for b in x1_list:
            for yi in range(0, tile_size, size):
                for xi in range(0, tile_size, size):
                    y1 = a + yi
                    x1 = b + xi
                    y2 = y1 + size
                    x2 = x1 + size

                    segment_context_window = segment_mask[a : a + tile_size, b : b + tile_size]
                    no_overlap_with_non_segment_region = not np.any(segment_context_window == 0)
                    ink_context_window = ink_mask[a : a + tile_size, b : b + tile_size]
                    context_window_has_ink = not np.all(ink_context_window < 0.05)
                    if context_window_has_ink:
                        if no_overlap_with_non_segment_region:
                            train_positions.append([x1, y1, x2, y2])
                            assert x2 - x1 == size
                            assert y2 - y1 == size
                            assert img_stack[y1:y2, x1:x2].shape == (
                                size,
                                size,
                                z_extent,
                            )
    return train_positions


def get_all_ink_patch_positions(
    ink_mask: np.ndarray,
    segment_mask: np.ndarray,
    patch_size: int,
    stride: int,
    ink_thresh: float = 0.05,
) -> list[tuple[int, int, int, int]]:
    """
    Identifies valid positions for patches in a given area based on ink regions.

    This function scans an image using a square patch of a specified size and stride, and
    determines valid positions where the patch meets certain criteria of ink coverage.
    A position is considered valid if it either contains any part of the ink region.

    Args:
        ink_mask (np.ndarray): A binary mask indicating the ink region in the image.
        segment_mask (np.ndarray): A binary mask indicating the segment region of the image.
        patch_size (int): The size of the square patch (e.g., 64 for a 64x64 patch).
        stride (int): The stride length for moving the patch across the image.
        ink_thresh (float): The threshold for a value to be considered ink. Defaults to 0.05.


    Returns:
        list[tuple[tuple[int, int], tuple[int, int]]]: A list of tuples, each containing two tuples. The first tuple
        in each pair represents the top-left coordinates (x, y) of a valid patch, and the second tuple represents
        the bottom-right coordinates (x + patch_size, y + patch_size).
    """
    valid_positions = []
    rows, cols = ink_mask.shape

    for y1 in range(0, rows - patch_size + 1, stride):
        y2 = y1 + patch_size
        for x1 in range(0, cols - patch_size + 1, stride):
            x2 = x1 + patch_size
            ink_patch = ink_mask[y1:y2, x1:x2]
            segment_patch = segment_mask[y1:y2, x1:x2]

            # Check for overlap with non-segment region.
            no_overlap_with_non_segment_region = not np.any(segment_patch == 0)

            # Check for presence of ink region
            contains_ink = np.all(ink_patch > ink_thresh)

            if no_overlap_with_non_segment_region:
                if contains_ink:
                    position = (x1, y1, x2, y2)
                    valid_positions.append(position)

    return valid_positions


def get_any_ink_patch_positions(
    ink_mask: np.ndarray,
    segment_mask: np.ndarray,
    patch_size: int,
    stride: int,
    ink_thresh: float = 0.05,
) -> list[tuple[int, int, int, int]]:
    """
    Identifies valid positions for patches in a given area based on ink regions.

    This function scans an image using a square patch of a specified size and stride, and
    determines valid positions where the patch meets certain criteria of ink coverage.
    A position is considered valid if it either contains any part of the ink region.

    Args:
        ink_mask (np.ndarray): A binary mask indicating the ink region in the image.
        segment_mask (np.ndarray): A binary mask indicating the segment region of the image.
        patch_size (int): The size of the square patch (e.g., 64 for a 64x64 patch).
        stride (int): The stride length for moving the patch across the image.
        ink_thresh (float): The threshold for a value to be considered ink. Defaults to 0.05.


    Returns:
        list[tuple[tuple[int, int], tuple[int, int]]]: A list of tuples, each containing two tuples. The first tuple
        in each pair represents the top-left coordinates (x, y) of a valid patch, and the second tuple represents
        the bottom-right coordinates (x + patch_size, y + patch_size).
    """
    valid_positions = []
    rows, cols = ink_mask.shape

    for y1 in range(0, rows - patch_size + 1, stride):
        y2 = y1 + patch_size
        for x1 in range(0, cols - patch_size + 1, stride):
            x2 = x1 + patch_size
            ink_patch = ink_mask[y1:y2, x1:x2]
            segment_patch = segment_mask[y1:y2, x1:x2]

            # Check for overlap with non-segment region.
            no_overlap_with_non_segment_region = not np.any(segment_patch == 0)

            # Check for presence of any ink in the patch.
            contains_ink = np.any(ink_patch > ink_thresh)

            if no_overlap_with_non_segment_region and contains_ink:
                position = (x1, y1, x2, y2)
                valid_positions.append(position)

    return valid_positions


def get_ink_patch_positions_batched(
    ink_mask: np.ndarray,
    segment_mask: np.ndarray,
    patch_shape: tuple[int, int],
    patch_stride: int,
    ink_thresh: float = 0.05,
    chunk_size: int = 256,
    should_pad: bool = True,
    all_ink_patches: bool = False,
    skip_seg_masked_regions: bool = True,
) -> list[tuple[int, int, int, int]]:
    # Pad to same shape.
    if ink_mask.shape != segment_mask.shape:
        if should_pad:
            ink_mask = pad_to_match(ink_mask, segment_mask)
            segment_mask = pad_to_match(segment_mask, ink_mask)
        else:
            raise ValueError(
                f"ink mask and segment mask shapes ({ink_mask.shape}, {segment_mask.shape}) do not match."
                f"If this is expected, set `should_pad` to True."
            )

    patch_pos_array_full = []
    y_max, x_max = segment_mask.shape
    for y_offset in range(0, y_max, chunk_size):
        for x_offset in range(0, x_max, chunk_size):
            y1_tile = y_offset
            y2_tile = min(y1_tile + chunk_size + patch_shape[0] - patch_stride, y_max)
            x1_tile = x_offset
            x2_tile = min(x1_tile + chunk_size + patch_shape[1] - patch_stride, x_max)
            segment_mask_patch = segment_mask[y1_tile:y2_tile, x1_tile:x2_tile]
            ink_mask_patch = ink_mask[y1_tile:y2_tile, x1_tile:x2_tile]

            if np.all(segment_mask_patch == 0):  # Skip any fully masked chunk.
                continue
            elif not np.any(
                ink_mask_patch > ink_thresh
            ):  # There must be at least some ink in the chunk.
                continue

            segment_patches = patchify(segment_mask_patch, patch_shape, patch_stride)
            ink_patches = patchify(ink_mask_patch, patch_shape, patch_stride)

            xy_dims = (2, 3)
            contains_ink = ink_patches > ink_thresh
            if all_ink_patches:
                contains_ink = contains_ink.all(xy_dims)
            else:
                contains_ink = contains_ink.any(xy_dims)

            is_valid_patch = contains_ink
            if not skip_seg_masked_regions:
                no_overlap_with_non_segment_region = (segment_patches != 0).any(xy_dims)
                is_valid_patch &= no_overlap_with_non_segment_region

            patch_x, patch_y = is_valid_patch.nonzero()
            if len(patch_x) > 0 or len(patch_y) > 0:
                valid_positions = []
                for patch_i, patch_j in zip(patch_x, patch_y):
                    (y1_patch, x1_patch), (y2_patch, x2_patch) = patch_index_to_pixel_position(
                        patch_i, patch_j, patch_shape, patch_stride
                    )

                    # Adjust positions for the chunk's offset in the original image.
                    y1_patch += y_offset
                    y2_patch += y_offset
                    x1_patch += x_offset
                    x2_patch += x_offset

                    position = (x1_patch, y1_patch, x2_patch, y2_patch)
                    valid_positions.append(position)
                patch_pos_array_full += valid_positions

    return patch_pos_array_full


def get_valid_patch_positions_batched(
    ink_mask: np.ndarray,
    non_ink_mask: np.ndarray,
    segment_mask: np.ndarray,
    patch_shape: tuple[int, int],
    patch_stride: int,
    min_labeled_coverage_frac: float = 0.5,
    ink_thresh: float = 0.05,
    chunk_size: int = 256,
    should_pad: bool = True,
    skip_seg_masked_regions: bool = True,
) -> list[tuple[int, int, int, int]]:
    if not (0 < min_labeled_coverage_frac <= 1):
        raise ValueError(
            f"`min_labeled_coverage_frac` must be in the half-open interval (0, 1]. Found min_labeled_coverage_frac={min_labeled_coverage_frac}"
        )

    # Pad to same shape.
    if ink_mask.shape != segment_mask.shape:
        if should_pad:
            ink_mask = pad_to_match(ink_mask, segment_mask)
            segment_mask = pad_to_match(segment_mask, ink_mask)
            non_ink_mask = pad_to_match(non_ink_mask, ink_mask)
        else:
            raise ValueError(
                f"ink mask and segment mask shapes ({ink_mask.shape}, {segment_mask.shape}) do not match."
                f"If this is expected, set `should_pad` to True."
            )

    patch_pos_array_full = []
    total_pixels = patch_shape[0] * patch_shape[1]
    y_max, x_max = segment_mask.shape
    for y_offset in range(0, y_max, chunk_size):
        for x_offset in range(0, x_max, chunk_size):
            y1_tile = y_offset
            y2_tile = min(y1_tile + chunk_size + patch_shape[0] - patch_stride, y_max)
            x1_tile = x_offset
            x2_tile = min(x1_tile + chunk_size + patch_shape[1] - patch_stride, x_max)
            segment_mask_tile = segment_mask[y1_tile:y2_tile, x1_tile:x2_tile]
            ink_mask_tile = ink_mask[y1_tile:y2_tile, x1_tile:x2_tile]
            non_ink_tile = non_ink_mask[y1_tile:y2_tile, x1_tile:x2_tile]

            if np.all(segment_mask_tile == 0):
                # Skip any fully masked chunk.
                continue
            elif not np.any(ink_mask_tile > ink_thresh) and not np.any(non_ink_tile):
                # There must be at least some ink or non-ink in the chunk.
                continue

            segment_patches = patchify(segment_mask_tile, patch_shape, patch_stride)
            ink_patches = patchify(ink_mask_tile, patch_shape, patch_stride)
            non_ink_patches = patchify(non_ink_tile, patch_shape, patch_stride)

            xy_dims = (2, 3)
            ink_patches_binarized = ink_patches > ink_thresh
            ink_coverage = np.sum(ink_patches_binarized, axis=xy_dims)

            non_ink_patches_binarized = non_ink_patches > 0
            non_ink_coverage = np.sum(non_ink_patches_binarized > 0, axis=xy_dims)

            labeled_coverage_frac = (ink_coverage + non_ink_coverage) / total_pixels
            sufficient_labeled_area = labeled_coverage_frac >= min_labeled_coverage_frac
            is_valid_patch = sufficient_labeled_area

            if not skip_seg_masked_regions:
                no_overlap_with_non_segment_region = (segment_patches != 0).any(xy_dims)
                is_valid_patch &= no_overlap_with_non_segment_region

            if is_valid_patch.any():
                patch_x, patch_y = is_valid_patch.nonzero()
                if len(patch_x) > 0 or len(patch_y) > 0:
                    valid_positions = []
                    for patch_i, patch_j in zip(patch_x, patch_y):
                        (y1_patch, x1_patch), (y2_patch, x2_patch) = patch_index_to_pixel_position(
                            patch_i, patch_j, patch_shape, patch_stride
                        )

                        # Adjust positions for the chunk's offset in the original image.
                        y1_patch += y_offset
                        y2_patch += y_offset
                        x1_patch += x_offset
                        x2_patch += x_offset

                        position = (x1_patch, y1_patch, x2_patch, y2_patch)
                        valid_positions.append(position)
                    patch_pos_array_full += valid_positions

    return patch_pos_array_full


def get_all_patch_positions_non_masked_batched(
    segment_mask: np.ndarray,
    patch_shape: tuple[int, int],
    patch_stride: int,
    chunk_size: int = 256,
) -> list[tuple[int, int, int, int]]:
    patch_pos_array_full = []
    y_max, x_max = segment_mask.shape
    for y_offset in range(0, y_max, chunk_size):
        for x_offset in range(0, x_max, chunk_size):
            y1_tile = y_offset
            y2_tile = min(y1_tile + chunk_size + patch_shape[0] - patch_stride, y_max)
            x1_tile = x_offset
            x2_tile = min(x1_tile + chunk_size + patch_shape[1] - patch_stride, x_max)
            segment_mask_tile = segment_mask[y1_tile:y2_tile, x1_tile:x2_tile]

            if np.all(segment_mask_tile == 0):
                # Skip any fully masked chunk.
                continue

            segment_patches = patchify(segment_mask_tile, patch_shape, patch_stride)
            xy_dims = (2, 3)
            no_overlap_with_non_segment_region = (segment_patches != 0).any(xy_dims)
            if no_overlap_with_non_segment_region.any():
                patch_x, patch_y = no_overlap_with_non_segment_region.nonzero()
                if len(patch_x) > 0 or len(patch_y) > 0:
                    valid_positions = []
                    for patch_i, patch_j in zip(patch_x, patch_y):
                        (y1_patch, x1_patch), (y2_patch, x2_patch) = patch_index_to_pixel_position(
                            patch_i, patch_j, patch_shape, patch_stride
                        )

                        # Adjust positions for the chunk's offset in the original image.
                        y1_patch += y_offset
                        y2_patch += y_offset
                        x1_patch += x_offset
                        x2_patch += x_offset

                        position = (x1_patch, y1_patch, x2_patch, y2_patch)
                        valid_positions.append(position)
                    patch_pos_array_full += valid_positions

    return patch_pos_array_full
