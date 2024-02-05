import numpy as np

MAX_VAL_UINT16 = np.iinfo(np.uint16).max


def preprocess_subvolume(subvolume: np.ndarray, slice_dim_last: bool = False) -> np.ndarray:
    """Preprocess a subvolume (convert to float32 and normalize to be in [0, 1]).

    This function takes a subvolume (such as a 3D array) and performs preprocessing steps on it.
    Specifically, it converts the values to float32 and normalizes them to the range [0, 1].

    Optionally, if the parameter `slice_dim_last` is set to True, the function will rearrange
    the axes of the subvolume such that the first dimension (usually corresponding to different
    slices or frames in a volumetric dataset) is moved to the last dimension. This can be useful
    in certain contexts where a specific axis ordering is required.

    Args:
        subvolume: A numpy array representing the subvolume to be preprocessed. It is expected
                   to be in an integer format, as the preprocessing includes normalization by
                   dividing by 65535.
        slice_dim_last: A boolean value that determines whether to move the first dimension to
                        the last. Default is False, meaning no rearrangement of axes.

    Returns:
        np.ndarray: A numpy array of the same shape as the input, but with values converted to
                    float32 and normalized to the range [0, 1]. The axes may also be rearranged
                    if `slice_dim_last` is True.
    """
    subvolume_pre = subvolume.astype(np.float32) / MAX_VAL_UINT16
    if slice_dim_last:
        subvolume_pre = np.moveaxis(subvolume_pre, 0, -1)
    return subvolume_pre
