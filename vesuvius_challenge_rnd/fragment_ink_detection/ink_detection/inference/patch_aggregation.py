import albumentations as A
import numpy as np


def resize_predictions(y_proba: np.ndarray, patch_surface_shape: tuple[int, int]) -> np.ndarray:
    """Resize patches of predicted probabilities to the specified surface shape.

    Args:
        y_proba (np.ndarray): Array of predicted probabilities.
        patch_surface_shape (tuple[int, int]): Target shape for resizing.

    Returns:
        np.ndarray: Resized array of predicted probabilities.
    """
    predict_transform = A.Resize(*patch_surface_shape, always_apply=True)
    y_proba_resized = np.empty((y_proba.shape[0], *patch_surface_shape))

    for i, y_proba_patch in enumerate(y_proba):
        transformed = predict_transform(image=y_proba_patch)
        y_proba_resized[i] = transformed["image"]
    return y_proba_resized


def average_y_proba_patches(
    y_proba_patches_resized: np.ndarray, patch_positions: np.ndarray, img_shape: tuple[int, int]
) -> np.ndarray:
    """Compute the arithmetic mean of predicted probabilities in overlapping patches.

    Args:
        y_proba_patches_resized (np.ndarray): Resized patches of predicted probabilities.
        patch_positions (np.ndarray): Positions of the patches.
        img_shape (tuple[int, int]): Shape of the original image.

    Returns:
        np.ndarray: Averaged predicted probabilities.
    """
    overlap_count = np.zeros(img_shape, dtype=float)
    y_proba_sum = np.zeros(img_shape, dtype=float)
    for y_proba_patch, ((y1, x1), (y2, x2)) in zip(y_proba_patches_resized, patch_positions):
        overlap_count[y1:y2, x1:x2] += 1
        y_proba_sum[y1:y2, x1:x2] += y_proba_patch

    y_proba_smoothed = np.divide(y_proba_sum, overlap_count, where=(overlap_count != 0))
    return y_proba_smoothed


def patches_to_y_proba(
    y_proba_patches: np.ndarray,
    patch_positions: np.ndarray,
    mask: np.ndarray,
    patch_surface_shape: tuple[int, int],
) -> np.ndarray:
    """Convert predicted (overlapping) patch probabilities to predicted probabilities for the entire image.

    Args:
        y_proba_patches (np.ndarray): Array of predicted probability of shape (num patches, down-sampled height, down-sampled width).
        patch_positions (np.ndarray): Array of predicted probability of shape (num patches, 2, 2).
        mask (np.ndarray): The papyrus mask of shape (height, width).
        patch_surface_shape (tuple[int, int]): The shape of the patch surface.

    Returns:
        np.ndarray: Predicted probabilities for the entire image.

    Raises:
        ValueError: If any dimension mismatch is found.
    """
    # Validate args.
    if y_proba_patches.ndim != 3:
        raise ValueError(
            f"Expected `y_proba_patches` to have 3 dimensions. Found {y_proba_patches.ndim}."
        )
    if patch_positions.ndim != 3:
        raise ValueError(
            f"Expected `patch_positions` to have 3 dimensions. Found {patch_positions.ndim}."
        )

    if y_proba_patches.shape[0] != patch_positions.shape[0]:
        raise ValueError(
            f"Expected first dimension of `y_proba_patches` to have the same as the first dim of `patch_positions` "
            f"dimensions. Found `y_proba_patches.shape[0]`={y_proba_patches.shape[0]} and "
            f"`patch_positions.shape[0]`={patch_positions.shape[0]}."
        )

    y_proba_patches_resized = resize_predictions(y_proba_patches, patch_surface_shape)

    y_proba_smoothed = average_y_proba_patches(y_proba_patches_resized, patch_positions, mask.shape)

    # Ensure masked region is zero.
    y_proba_smoothed[~mask] = 0.0

    return y_proba_smoothed


def parse_predictions_without_labels(predictions: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Parse patch predictions without labels.

    Args:
        predictions (list[tuple]): List of patch predictions.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing arrays of probability predictions and patch positions.
    """
    # Assumes the dataset returns two items: y_proba_patch and patch position.
    y_proba_patches = np.vstack([p[0].detach().cpu().numpy() for p in predictions])
    patch_positions = np.vstack([p[1] for p in predictions])
    return y_proba_patches, patch_positions


def parse_predictions_with_labels(
    predictions: list[tuple],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse patch predictions with labels.

    Args:
        predictions (list[tuple]): List of patch predictions.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing arrays of probability predictions, true labels, and patch positions.
    """
    # Assumes the dataset returns three items: y_proba_patch, patch ink label, and patch position.
    y_proba_patches = np.vstack([p[0].detach().cpu().numpy() for p in predictions])
    y_true_patches = np.vstack([p[1].detach().cpu().numpy() for p in predictions])
    patch_positions = np.vstack([p[2] for p in predictions])
    return y_proba_patches, y_true_patches, patch_positions


def predictions_to_y_proba(
    predictions: list[tuple], mask: np.ndarray, patch_surface_shape: tuple[int, int]
) -> np.ndarray:
    """Convert patch predictions to fragment predictions.

    Args:
        predictions (list[tuple]): List of patch predictions.
        mask (np.ndarray): The papyrus mask of shape (height, width).
        patch_surface_shape (tuple[int, int]): The shape of the patch surface.

    Returns:
        np.ndarray: Predicted probabilities for the entire fragment.
    """
    y_proba_patches, patch_positions = parse_predictions_without_labels(predictions)
    return patches_to_y_proba(y_proba_patches, patch_positions, mask, patch_surface_shape)
