from pathlib import Path

import cv2
import numpy as np


def load_ink_mask(
    mask_path: Path,
) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if "frag" in mask_path.stem:
        mask = cv2.resize(
            mask, (mask.shape[1] // 2, mask.shape[0] // 2), interpolation=cv2.INTER_AREA
        )
    return mask


def preprocess_ink_mask(
    ink_mask: np.ndarray,
    expected_shape: tuple[int, int],
    should_blur: bool = False,
    kernel_size: int = 17,
    min_component_size: int = 1_000,
    ink_erosion: int = 0,
    ignore_idx: int = -100,
) -> np.ndarray:
    ink_mask = remove_small_components(ink_mask, min_size=min_component_size)

    if ink_erosion > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ink_erosion, ink_erosion))
        ink_mask = cv2.erode(ink_mask, kernel, iterations=1)

    if should_blur:
        ink_mask = cv2.GaussianBlur(ink_mask, (kernel_size, kernel_size), 0)

    ink_mask = ink_mask.astype("float32")
    ink_mask /= 255

    # Pad if necessary.
    pad_y = expected_shape[0] - ink_mask.shape[0]
    pad_x = expected_shape[1] - ink_mask.shape[1]
    if pad_x < 0 or pad_y < 0:
        raise ValueError(
            f"expected shape ({expected_shape}) must be larger than raw ink mask shape ({ink_mask.shape}) for all dimensions."
        )
    ink_mask = np.pad(ink_mask, [(0, pad_y), (0, pad_x)], constant_values=ignore_idx)

    return ink_mask


def remove_small_components(mask: np.ndarray, min_size: int = 1_000) -> np.ndarray:
    if min_size == 0:
        return mask
    elif min_size < 0:
        raise ValueError(f"min_size must be nonnegative. Found value {min_size}")

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    small_component_indices = np.where(stats[1:, cv2.CC_STAT_AREA] < min_size)[0] + 1
    small_components_mask = np.isin(labels, small_component_indices)
    processed_mask = mask.copy()
    processed_mask[small_components_mask] = 0

    return processed_mask


def read_ink_mask(
    mask_path: Path,
    expected_shape: tuple[int, int],
    should_blur: bool = False,
    kernel_size: int = 17,
    min_component_size: int = 1_000,
    ink_erosion: int = 0,
    ignore_idx: int = -100,
):
    ink_mask = load_ink_mask(mask_path)
    ink_mask = preprocess_ink_mask(
        ink_mask,
        expected_shape=expected_shape,
        should_blur=should_blur,
        kernel_size=kernel_size,
        min_component_size=min_component_size,
        ink_erosion=ink_erosion,
        ignore_idx=ignore_idx,
    )

    return ink_mask


def read_papy_non_ink_labels(
    non_ink_mask_path: Path, expected_shape: tuple[int, int]
) -> np.ndarray:
    non_ink_mask = cv2.imread(str(non_ink_mask_path), cv2.IMREAD_GRAYSCALE)

    # Pad if necessary.
    pad_y = expected_shape[0] - non_ink_mask.shape[0]
    pad_x = expected_shape[1] - non_ink_mask.shape[1]
    if pad_x < 0 or pad_y < 0:
        raise ValueError(
            f"expected shape ({expected_shape}) must be larger than raw non-ink shape ({non_ink_mask.shape}) for all dimensions."
        )
    non_ink_mask = np.pad(non_ink_mask, [(0, pad_y), (0, pad_x)], constant_values=0)
    non_ink_mask = non_ink_mask.astype(bool)

    return non_ink_mask


def create_non_ink_mask_from_ink_mask(
    ink_mask: np.ndarray, dilation_kernel_size: int = 256
) -> np.ndarray:
    if ink_mask.dtype != np.uint8 or np.unique(ink_mask).size != 2:
        raise ValueError("ink_mask must be a binary mask of uint8 type")

    structuring_element = cv2.getStructuringElement(
        cv2.MORPH_RECT, (dilation_kernel_size, dilation_kernel_size)
    )
    dilated_mask = cv2.dilate(ink_mask, structuring_element, iterations=1)

    # Subtract the original mask from the dilated mask to create the expanded mask
    expanded_mask = cv2.subtract(dilated_mask, ink_mask)
    return expanded_mask


def create_non_ink_mask(
    ink_mask: np.ndarray, thresh: float = 0.5, dilation_kernel_size: int = 256
) -> np.ndarray:
    ink_mask_binarized = (ink_mask > thresh).astype(np.uint8)
    non_ink_mask = create_non_ink_mask_from_ink_mask(
        ink_mask_binarized, dilation_kernel_size=dilation_kernel_size
    ).astype(np.float32)
    return non_ink_mask
