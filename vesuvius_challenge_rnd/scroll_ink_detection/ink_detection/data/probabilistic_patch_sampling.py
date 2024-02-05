import numpy as np
from skimage.util import img_as_float32


def get_probabilistic_patch_samples(
    ink_mask_channel_0: np.ndarray,
    ink_mask_channel_2,
    non_ink_mask: np.ndarray,
    size: int,
    stride: int,
    p0: float = 0.3,
    p2: float = 0.6,
    p_non_ink: float = 0.1,
    ignore_idx: int = -100,
) -> list[tuple[int, int, int, int]]:
    ink_mask_channel_0 = ink_mask_channel_0.copy()
    ink_mask_channel_2 = ink_mask_channel_2.copy()
    ink_mask_channel_2 = ink_mask_channel_2.copy()
    ink_mask_channel_0 = np.where(ink_mask_channel_0 == ignore_idx, 0, ink_mask_channel_0)
    ink_mask_channel_2 = np.where(ink_mask_channel_2 == ignore_idx, 0, ink_mask_channel_2)
    prob_map = create_2d_probability_map(
        ink_mask_channel_0, ink_mask_channel_2, non_ink_mask, p0=p0, p2=p2, p_non_ink=p_non_ink
    )
    ink_mask_full = ink_mask_channel_0.astype(bool) | ink_mask_channel_2.astype(bool)
    height, width = prob_map.shape[0], prob_map.shape[1]
    n_samples = get_num_patches(size, stride, height, width, ink_mask_full, non_ink_mask)
    sample_yxs = sample_from_2d_probability_array(prob_map, num_samples=n_samples)
    patch_positions = centroids_to_xyxys(sample_yxs, size=size)

    # Filter patch positions exceeding image dimensions.
    x2s = patch_positions[:, 2]
    y2s = patch_positions[:, 3]
    in_bounds_mask = (x2s < width) & (y2s < height)
    patch_positions = patch_positions[in_bounds_mask]

    return patch_positions.tolist()


def create_2d_probability_map(
    ink_mask_channel_0: np.ndarray,
    ink_mask_channel_2,
    non_ink_mask: np.ndarray,
    p0: float = 0.3,
    p2: float = 0.6,
    p_non_ink: float = 0.1,
):
    prob_map = np.zeros(ink_mask_channel_0.shape, dtype=np.float32)
    prob_map[ink_mask_channel_0.astype(bool)] = p0
    prob_map[ink_mask_channel_2.astype(bool)] = p2
    prob_map[non_ink_mask.astype(bool)] = p_non_ink
    prob_map = img_as_float32(prob_map)
    return prob_map


def sample_from_2d_probability_array(
    prob_2d: np.ndarray, num_samples: int = 10
) -> list[tuple[int, int]]:
    """
    Samples a specified number of coordinates from a 2D array of probabilities.

    Parameters:
    prob_2d (np.array): A 2D numpy array of probabilities.
    num_samples (int): The number of samples to draw.

    Returns:
    list of tuples: A list of sampled 2D coordinates.
    """
    # Normalize the probabilities
    prob_flat = prob_2d.flatten()
    prob_flat /= prob_flat.sum()

    # Sample from the flattened array
    sample_indices = np.random.choice(len(prob_flat), size=num_samples, p=prob_flat)

    # Map the sampled indices back to 2D coordinates
    yxs = [np.unravel_index(index, prob_2d.shape) for index in sample_indices]

    return yxs


def get_num_windows(patch_size: tuple[int, int], stride: int, height: int, width: int) -> int:
    """Compute the number of patches for a rectangle with the given window size and stride."""
    ny = ((height - patch_size[1]) // stride) + 1
    nx = ((width - patch_size[0]) // stride) + 1
    return nx * ny


def get_num_patches(
    size: int, stride: int, height: int, width: int, ink_mask: np.ndarray, non_ink_mask: np.ndarray
) -> int:
    n_windows = get_num_windows((size, size), stride, height, width)

    # Adjust number of windows based on ink area ratio.
    ink_or_non_ink_mask = ink_mask.astype(bool) | non_ink_mask.astype(bool)
    area_segment = height * width
    area_masks = np.sum(ink_or_non_ink_mask)
    area_ratio = area_masks / area_segment
    n_samples = int(area_ratio * n_windows)
    return n_samples


def centroids_to_xyxys(centroids: np.ndarray, size: int) -> np.ndarray:
    centroids = np.asarray(centroids)

    patch_positions = np.empty((centroids.shape[0], 4), dtype=np.int32)
    half_size = size // 2
    patch_positions[:, 0] = centroids[:, 1] - half_size
    patch_positions[:, 1] = centroids[:, 0] - half_size
    patch_positions[:, 2] = centroids[:, 1] + half_size
    patch_positions[:, 3] = centroids[:, 0] + half_size

    return patch_positions
