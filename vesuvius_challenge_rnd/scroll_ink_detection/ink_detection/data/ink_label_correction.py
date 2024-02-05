import numpy as np
from skimage.util import img_as_float32


def apply_model_based_ink_label_correction(
    ink_labels_binarized: np.ndarray,
    non_ink_labels: np.ndarray,
    ink_pred: np.ndarray,
    model_based_ink_label_thresh: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    ink_pred = img_as_float32(ink_pred)
    ink_pred_binarized_ink_region = ink_pred > model_based_ink_label_thresh
    new_ink_labels = ink_labels_binarized & ink_pred_binarized_ink_region

    # Fix non-ink labels. Wherever it previously overlapped with old ink labels, and it was removed, assign it to BG.
    non_ink_labels = non_ink_labels.astype(bool)
    non_ink_labels = np.where(~new_ink_labels & ink_labels_binarized, 1, non_ink_labels)

    return new_ink_labels, non_ink_labels


def apply_model_based_non_ink_label_correction(
    non_ink_mask: np.ndarray,
    ink_mask: np.ndarray,
    ink_pred: np.ndarray,
    ink_idx: int = 1,
    model_based_ink_label_thresh: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    ink_pred = img_as_float32(ink_pred)
    non_ink_mask = non_ink_mask.astype(bool)
    ink_pred_binarized_non_ink_region = ink_pred > model_based_ink_label_thresh
    new_ink_mask = np.where(non_ink_mask & ink_pred_binarized_non_ink_region, ink_idx, ink_mask)
    new_non_ink_mask = ~new_ink_mask & non_ink_mask
    return new_ink_mask, new_non_ink_mask


def apply_model_based_label_correction(
    ink_mask: np.ndarray,
    non_ink_mask: np.ndarray,
    ink_pred: np.ndarray,
    ink_label_thresh: float = 0.05,
    model_based_ink_correction_thresh: float = 0.1,
    model_based_non_ink_correction_thresh: float = 0.3,
    clean_up_ink_labels: bool = True,
    clean_up_non_ink_labels: bool = True,
    ignore_idx: int = -100,
) -> tuple[np.ndarray, np.ndarray]:
    """

    :param ink_mask:
    :param non_ink_mask:
    :param ink_pred:
    :param ink_label_thresh:
    :param model_based_ink_correction_thresh: anything less than model_based_ink_label_thresh within the ink mask will
    be excluded from the ink mask.
    :param model_based_non_ink_correction_thresh: anything greater than model_based_non_ink_label_thresh within the non-ink
    mask will be added to the non-ink labels
    :param clean_up_ink_labels:
    :param clean_up_non_ink_labels:
    :param ignore_idx:
    :return:
    """
    ink_mask_orig_dtype = ink_mask.dtype
    non_ink_mask_orig_dtype = non_ink_mask.dtype
    ink_pred = img_as_float32(ink_pred)

    # Zero the ignored index regions out.
    ignore_mask = ink_mask == ignore_idx
    ink_mask = np.where(ignore_mask, 0, ink_mask)

    ink_mask = ink_mask > ink_label_thresh

    # Clean up ink labels.
    if clean_up_ink_labels:
        ink_mask, non_ink_mask = apply_model_based_ink_label_correction(
            ink_mask,
            non_ink_mask,
            ink_pred,
            model_based_ink_label_thresh=model_based_ink_correction_thresh,
        )

    # Clean up non-ink labels. (Add nearby missed ink based on model's predictions)
    if clean_up_non_ink_labels:
        ink_mask, non_ink_mask = apply_model_based_non_ink_label_correction(
            non_ink_mask,
            ink_mask,
            ink_pred,
            model_based_ink_label_thresh=model_based_non_ink_correction_thresh,
        )

    # Add the ignored index back.
    ink_mask = np.where(ignore_mask, ignore_idx, ink_mask)

    # Fix dtype.
    ink_mask = ink_mask.astype(ink_mask_orig_dtype)
    non_ink_mask = non_ink_mask.astype(non_ink_mask_orig_dtype)

    return ink_mask, non_ink_mask
