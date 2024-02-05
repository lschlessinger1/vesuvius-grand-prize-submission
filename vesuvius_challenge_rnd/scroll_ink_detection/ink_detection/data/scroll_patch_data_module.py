import logging
import os
from functools import partial
from pathlib import Path
from tempfile import mkdtemp
from warnings import warn

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from skimage.util import img_as_ubyte
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from tqdm.contrib import tzip

from vesuvius_challenge_rnd import SCROLL_DATA_DIR
from vesuvius_challenge_rnd.data import Scroll, ScrollSegment
from vesuvius_challenge_rnd.data.constants import (
    KNOWN_Z_ORIENTATION_SEGMENT_IDS,
    Z_REVERSED_SEGMENT_IDS,
)
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.data.base_patch_data_module import (
    BasePatchDataModule,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.block_dataset import (
    BlockZConstantDataset,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.ink_label_correction import (
    apply_model_based_label_correction,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.memmap import (
    MEMMAP_DIR,
    delete_memmap,
    load_segment_as_memmap,
    save_segment_as_memmap,
    segment_to_memmap_path,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.patch_memmap_dataset import (
    PatchMemMapDataset,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.patch_sampling import (
    get_ink_patch_positions_batched,
    get_valid_patch_positions_batched,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.probabilistic_patch_sampling import (
    get_probabilistic_patch_samples,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.tools.ink_pred_utils import (
    read_ink_preds,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.tools.label_utils import (
    create_non_ink_mask,
    read_ink_mask,
    read_papy_non_ink_labels,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.zarr import (
    ZARR_ARRAY_DIR,
    load_segment_as_zarr_array,
    save_segment_as_zarr_array,
    segment_to_zarr_path,
)

ScrollSegmentsDataType = str | list[tuple[str, str]]


class ScrollPatchDataModule(BasePatchDataModule):
    def __init__(
        self,
        train_segment_ids: ScrollSegmentsDataType,
        val_segment_id: str | tuple[str, str],
        ink_label_dir: Path,
        data_dir: Path = SCROLL_DATA_DIR,
        z_min: int = 15,
        z_max: int = 47,
        size: int = 64,
        tile_size: int = 256,
        min_labeled_coverage_frac: float = 1,
        patch_stride_train: int = 8,
        patch_stride_val: int = 32,
        downsampling: int | None = None,
        batch_size: int = 256,
        num_workers: int = 0,
        blur_ink_labels: bool = False,
        ink_labels_blur_kernel_size: int = 17,
        ink_dilation_kernel_size: int = 256,
        min_ink_component_size: int = 1000,
        label_downscale: int = 1,
        ink_erosion: int = 0,
        ignore_idx: int = -100,
        clip_min: int = 0,
        clip_max: int = 255,
        patch_train_stride_strict: int = 8,
        patch_val_stride_strict: int = 8,
        strict_sampling_only_ink_train: bool = True,
        strict_sampling_only_ink_val: bool = True,
        min_crop_num_offset: int = 8,
        chunks_load: tuple[int, int, int] | int | bool = True,
        use_zarr: bool = False,
        zarr_dir: Path = ZARR_ARRAY_DIR,
        x_chunk_save_size: int = 512,
        y_chunk_save_size: int = 512,
        z_chunk_save_size: int = 32,
        skip_save_zarr_if_exists: bool = True,
        zarr_load_in_memory: bool = True,
        model_prediction_dir: Path | None = None,
        model_based_ink_correction_thresh_train: float = 0.1,
        model_based_ink_correction_thresh_val: float = 0.1,
        model_based_non_ink_correction_thresh_train: float = 0.3,
        model_based_non_ink_correction_thresh_val: float = 0.3,
        clean_up_ink_labels_train: bool = False,
        clean_up_ink_labels_val: bool = False,
        clean_up_non_ink_labels_train: bool = False,
        clean_up_non_ink_labels_val: bool = False,
        p_0_ink: float = 0.3,
        p_2_ink: float = 0.6,
        p_non_ink: float = 0.1,
        automatic_non_ink_labels: bool = False,
        cache_memmaps: bool = False,
        memmap_dir: Path = MEMMAP_DIR,
    ):
        if ink_labels_blur_kernel_size % 2 != 1:
            raise ValueError("`ink_labels_blur_kernel_size` must be odd")
        super().__init__(
            data_dir=data_dir,
            z_min=z_min,
            z_max=z_max,
            patch_surface_shape=(size, size),
            patch_stride=patch_stride_train,
            downsampling=downsampling,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if tile_size < size:
            raise ValueError(f"Tile size ({tile_size}) cannot be less than patch size ({size}).")

        if min_crop_num_offset > self.z_extent:
            raise ValueError(
                f"min_crop_num_offset ({min_crop_num_offset}) cannot be less than z-extent ({self.z_extent})."
            )

        if cache_memmaps and use_zarr:
            raise ValueError(
                f"zarr and memmap cache are mutually exclusive. Use neither or only one."
            )

        self.size = size
        self.tile_size = tile_size
        self.min_labeled_coverage_frac = min_labeled_coverage_frac
        self.ink_dilation_kernel_size = ink_dilation_kernel_size
        self.patch_stride_train = patch_stride_train
        self.patch_stride_val = patch_stride_val
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.ink_label_dir = ink_label_dir
        self.model_prediction_dir = model_prediction_dir
        self.model_based_ink_correction_thresh_train = model_based_ink_correction_thresh_train
        self.model_based_ink_correction_thresh_val = model_based_ink_correction_thresh_val
        self.model_based_non_ink_correction_thresh_train = (
            model_based_non_ink_correction_thresh_train
        )
        self.model_based_non_ink_correction_thresh_val = model_based_non_ink_correction_thresh_val
        self.clean_up_ink_labels_train = clean_up_ink_labels_train
        self.clean_up_ink_labels_val = clean_up_ink_labels_val
        self.clean_up_non_ink_labels_train = clean_up_non_ink_labels_train
        self.clean_up_non_ink_labels_val = clean_up_non_ink_labels_val

        self.p_0_ink = p_0_ink
        self.p_2_ink = p_2_ink
        self.p_non_ink = p_non_ink

        self.blur_ink_labels = blur_ink_labels
        self.ink_label_blur_kernel_size = ink_labels_blur_kernel_size
        self.min_ink_component_size = min_ink_component_size
        self.ink_erosion = ink_erosion
        self.ignore_idx = ignore_idx
        self.min_crop_num_offset = min_crop_num_offset
        self.patch_train_stride_strict = patch_train_stride_strict
        self.patch_val_stride_strict = patch_val_stride_strict
        self.strict_sampling_only_ink_train = strict_sampling_only_ink_train
        self.strict_sampling_only_ink_val = strict_sampling_only_ink_val

        # Set zarr parameters.
        self.use_zarr = use_zarr
        self.zarr_dir = zarr_dir
        self.chunks_load = chunks_load
        self.x_chunk_save_size = x_chunk_save_size
        self.y_chunk_save_size = y_chunk_save_size
        self.z_chunk_save_size = z_chunk_save_size
        self.skip_save_zarr_if_exists = skip_save_zarr_if_exists
        self.zarr_load_in_memory = zarr_load_in_memory

        # Memmaps parameters.
        self.cache_memmaps = cache_memmaps
        self.memmap_dir = memmap_dir

        # Create the scroll segments.
        self.segments_train = self._filter_segments(self._instantiate_segments(train_segment_ids))
        self.segment_val = self._filter_segments(self._instantiate_segments([val_segment_id]))[0]

        all_segment_names = [s.segment_name_orig for s in self.segments_train] + [
            self.segment_val.segment_name_orig
        ]

        _check_ink_labels(ink_label_dir, all_segment_names)

        if model_prediction_dir is not None:
            _check_ink_pred_dir(model_prediction_dir, all_segment_names)

        self.pred_shape = self.segment_val.surface_shape
        self.label_downscale = label_downscale

        # Check papyrus non-ink labels.
        self.automatic_non_ink_labels = automatic_non_ink_labels

        if not self.automatic_non_ink_labels:
            _check_papyrus_non_ink_labels(ink_label_dir, all_segment_names)

    def _instantiate_segments(
        self, scroll_segment_data: ScrollSegmentsDataType
    ) -> list[ScrollSegment]:
        if isinstance(scroll_segment_data, list):
            segments = []
            for scroll_id, segment_name in scroll_segment_data:
                segment = ScrollSegment(scroll_id, segment_name, scroll_dir=self.data_dir)
                segments.append(segment)
        elif scroll_segment_data in ("1", "2"):
            scroll = Scroll(scroll_id=scroll_segment_data, scroll_dir=self.data_dir)
            segments = scroll.segments
        elif scroll_segment_data.lower() == "all":
            scrolls = [Scroll(str(i + 1)) for i in range(2)]
            segments = scrolls[0].segments + scrolls[1].segments
        else:
            raise ValueError(
                "Scroll segments must be '1', '2', 'all', or a list of tuples of scroll IDs and segment names (e.g., "
                f"[(1, '20230828154913'), (1, '20230819210052')]). Found value {scroll_segment_data}."
            )

        return segments

    @property
    def train_segment_ids(self) -> list[str]:
        return [seg.segment_name for seg in self.segments_train]

    @property
    def val_segment_id(self) -> str:
        return self.segment_val.segment_name

    def _filter_segments(self, segments: list[ScrollSegment]):
        filtered_segments = []
        for segment in segments:
            if (
                segment.surface_shape[0] >= self.patch_surface_shape[0]
                and segment.surface_shape[1] >= self.patch_surface_shape[1]
            ):
                filtered_segments.append(segment)
            else:
                logging.warning(
                    f"Skipping scroll {segment.scroll_id} segment {segment.segment_name} with surface shape {segment.surface_shape} because "
                    f"it's smaller than the patch surface shape: {self.patch_surface_shape}."
                )
        return filtered_segments

    @property
    def z_extent(self) -> int:
        return self.z_max - self.z_min

    def prepare_data(self) -> None:
        """Download data to disk.

        This method checks if the data directory is empty and raises an exception
        if the data is not found.

        Raises:
            ValueError: If the data directory is empty.
        """
        super().prepare_data()
        # If using zarr, possibly create the zarr arrays.
        if self.use_zarr:
            for segment in tqdm(self.segments_train + [self.segment_val], desc="Creating zarrs..."):
                zarr_path = segment_to_zarr_path(segment, zarr_dir=self.zarr_dir)
                if zarr_path.exists() and self.skip_save_zarr_if_exists:
                    logging.info(f"Found existing zarr for segment {segment.segment_name}.")
                else:
                    logging.info(f"Creating new zarr for segment {segment.segment_name}.")
                    save_segment_as_zarr_array(
                        segment,
                        output_zarr_path=zarr_path,
                        z_chunk_size=self.z_chunk_save_size,
                        y_chunk_size=self.y_chunk_save_size,
                        x_chunk_size=self.x_chunk_save_size,
                        load_in_memory=self.zarr_load_in_memory,
                    )
        elif self.cache_memmaps:
            self.memmap_dir.mkdir(exist_ok=True, parents=True)
            for segment in tqdm(
                self.segments_train + [self.segment_val], desc="Creating memmaps..."
            ):
                memmap_path = segment_to_memmap_path(segment, memmap_dir=self.memmap_dir)
                if memmap_path.exists() and self.skip_save_zarr_if_exists:
                    logging.info(f"Found existing memmap for segment {segment.segment_name}.")
                else:
                    logging.info(f"Creating new memmap for segment {segment.segment_name}.")
                    save_segment_as_memmap(
                        segment,
                        memmap_path,
                        load_in_memory=self.zarr_load_in_memory,
                    )

    def setup(self, stage: str) -> None:
        """Set up the data for the given stage.

        Args:
            stage (str): The stage for which to set up the data (e.g., "fit").
        """
        if stage == "fit" or stage is None:
            (
                img_stack_refs_train,
                padding_list_train,
                z_reversals_train,
                segment_masks_train,
                ink_masks_train,
                ink_preds,
                non_ink_masks_train,
            ) = generate_dataset_inputs(
                self.segments_train,
                self.z_min,
                self.z_max,
                self.tile_size,
                self.ink_label_dir,
                should_blur_ink_labels=self.blur_ink_labels,
                ink_labels_blur_kernel_size=self.ink_label_blur_kernel_size,
                min_ink_component_size=self.min_ink_component_size,
                ink_erosion=self.ink_erosion,
                ignore_idx=self.ignore_idx,
                chunks_load=self.chunks_load,
                use_zarr=self.use_zarr,
                zarr_dir=self.zarr_dir,
                cache_memmaps=self.cache_memmaps,
                memmap_dir=self.memmap_dir,
                ink_preds_dir=self.model_prediction_dir,
                automatic_non_ink=self.automatic_non_ink_labels,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
            )

            train_patch_positions, non_ink_ignored_patches_train = create_train_patch_positions(
                segment_masks_train,
                ink_masks_train,
                self.ink_dilation_kernel_size,
                self.size,
                self.patch_stride_train,
                self.patch_train_stride_strict,
                strict_sampling_only_ink=self.strict_sampling_only_ink_train,
                tile_size=self.tile_size,
                min_labeled_coverage_frac=self.min_labeled_coverage_frac,
                ink_preds=ink_preds,
                model_based_ink_correction_thresh=self.model_based_ink_correction_thresh_train,
                model_based_non_ink_correction_thresh=self.model_based_non_ink_correction_thresh_train,
                clean_up_ink_labels=self.clean_up_ink_labels_train,
                clean_up_non_ink_labels=self.clean_up_non_ink_labels_train,
                p_0_ink=self.p_0_ink,
                p_2_ink=self.p_2_ink,
                p_non_ink=self.p_non_ink,
                ignore_idx=self.ignore_idx,
                non_ink_masks=non_ink_masks_train,
            )

            (
                img_stack_refs_val,
                padding_list_val,
                z_reversals_val,
                segment_masks_val,
                ink_masks_val,
                ink_preds,
                non_ink_masks_val,
            ) = generate_dataset_inputs(
                [self.segment_val],
                self.z_min,
                self.z_max,
                self.tile_size,
                self.ink_label_dir,
                min_ink_component_size=self.min_ink_component_size,
                ink_erosion=self.ink_erosion,
                ignore_idx=self.ignore_idx,
                chunks_load=self.chunks_load,
                use_zarr=self.use_zarr,
                zarr_dir=self.zarr_dir,
                cache_memmaps=self.cache_memmaps,
                memmap_dir=self.memmap_dir,
                ink_preds_dir=self.model_prediction_dir,
                automatic_non_ink=self.automatic_non_ink_labels,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
            )

            val_patch_positions, non_ink_ignored_patches_val = create_val_patch_positions(
                segment_masks_val,
                ink_masks_val,
                self.ink_dilation_kernel_size,
                self.size,
                self.patch_stride_val,
                self.patch_val_stride_strict,
                strict_sampling_only_ink=self.strict_sampling_only_ink_val,
                tile_size=self.tile_size,
                min_labeled_coverage_frac=self.min_labeled_coverage_frac,
                ink_preds=ink_preds,
                model_based_ink_correction_thresh=self.model_based_ink_correction_thresh_val,
                model_based_non_ink_correction_thresh=self.model_based_non_ink_correction_thresh_val,
                clean_up_ink_labels=self.clean_up_ink_labels_val,
                clean_up_non_ink_labels=self.clean_up_non_ink_labels_val,
                p_0_ink=self.p_0_ink,
                p_2_ink=self.p_2_ink,
                p_non_ink=self.p_non_ink,
                ignore_idx=self.ignore_idx,
                non_ink_masks=non_ink_masks_val,
            )

            ink_masks_train = max_proj_multi_channel_ink_masks(ink_masks_train)
            ink_masks_val = max_proj_multi_channel_ink_masks(ink_masks_val)

            self.data_train = create_dataset(
                img_stack_refs_train,
                train_patch_positions,
                ink_masks_train,
                z_reversals_train,
                padding_list_train,
                self.size,
                self.z_min,
                self.z_extent,
                augment=True,
                transform=self.train_transform(),
                label_downscale=self.label_downscale,
                non_ink_ignored_patches=non_ink_ignored_patches_train,
                ignore_idx=self.ignore_idx,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                min_crop_num_offset=self.min_crop_num_offset,
                use_zarr_or_mmap_cache=self.use_zarr or self.cache_memmaps,
            )

            self.data_val = create_dataset(
                img_stack_refs_val,
                val_patch_positions,
                ink_masks_val,
                z_reversals_val,
                padding_list_val,
                self.size,
                self.z_min,
                self.z_extent,
                augment=False,
                transform=self.val_transform(),
                label_downscale=self.label_downscale,
                non_ink_ignored_patches=non_ink_ignored_patches_val,
                ignore_idx=self.ignore_idx,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                min_crop_num_offset=self.min_crop_num_offset,
                use_zarr_or_mmap_cache=self.use_zarr or self.cache_memmaps,
            )

    def train_transform(self) -> A.Compose:
        return A.Compose(
            [
                A.Resize(self.size, self.size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.75),
                A.ShiftScaleRotate(rotate_limit=360, shift_limit=0.15, scale_limit=0.15, p=0.75),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10, 50)),
                        A.GaussianBlur(),
                        A.MotionBlur(),
                    ],
                    p=0.4,
                ),
                A.CoarseDropout(
                    max_holes=2,
                    max_width=int(self.size * 0.2),
                    max_height=int(self.size * 0.2),
                    mask_fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=[0] * self.z_extent, std=[1] * self.z_extent),
                ToTensorV2(transpose_mask=True),
            ]
        )

    def val_transform(self) -> A.Compose:
        return A.Compose(
            [
                A.Resize(self.size, self.size),
                A.Normalize(mean=[0] * self.z_extent, std=[1] * self.z_extent),
                ToTensorV2(transpose_mask=True),
            ]
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )


def max_proj_multi_channel_ink_masks(ink_masks: list[np.ndarray]) -> list[np.ndarray]:
    ink_masks_new = []
    for ink_mask in ink_masks:
        if ink_mask.ndim == 3:
            ink_masks_new.append(np.amax(ink_mask, axis=0))
        else:
            ink_masks_new.append(ink_mask.squeeze())
    return ink_masks_new


def create_dataset(
    img_stack_refs: list[np.memmap],
    patch_position_list: list,
    ink_masks: list[np.ndarray],
    z_reversals: list[bool],
    padding_list: list[tuple[int, int]],
    size: int,
    z_start: int,
    z_extent: int,
    augment: bool = False,
    transform=None,
    label_downscale: int = 4,
    non_ink_ignored_patches: np.ndarray | None = None,
    ignore_idx: int = -100,
    clip_min: int = 0,
    clip_max: int = 200,
    min_crop_num_offset: int = 8,
    use_zarr_or_mmap_cache: bool = True,
) -> ConcatDataset:
    datasets = []
    for (
        img_stack_ref,
        patch_positions,
        ink_mask,
        non_ink_ignored_patches_seg,
        z_reverse,
        padding_yx,
    ) in zip(
        img_stack_refs,
        patch_position_list,
        ink_masks,
        non_ink_ignored_patches,
        z_reversals,
        padding_list,
    ):
        if use_zarr_or_mmap_cache:
            dataset = BlockZConstantDataset(
                img_stack_ref,
                np.array(patch_positions),
                size,
                z_start=z_start,
                z_extent=z_extent,
                labels=ink_mask,
                augment=augment,
                transform=transform,
                label_downscale=label_downscale,
                non_ink_ignored_patches=non_ink_ignored_patches_seg,
                ignore_idx=ignore_idx,
                min_crop_num_offset=min_crop_num_offset,
                z_reverse=z_reverse,
                padding_yx=padding_yx,
                clip_min=clip_min,
                clip_max=clip_max,
            )
        else:
            dataset = PatchMemMapDataset(
                img_stack_ref,
                np.array(patch_positions),
                size,
                z_extent=z_extent,
                labels=ink_mask,
                augment=augment,
                transform=transform,
                label_downscale=label_downscale,
                non_ink_ignored_patches=non_ink_ignored_patches_seg,
                ignore_idx=ignore_idx,
                min_crop_num_offset=min_crop_num_offset,
                z_reverse=z_reverse,
                padding_yx=padding_yx,
                clip_min=clip_min,
                clip_max=clip_max,
            )
        datasets.append(dataset)
    return ConcatDataset(datasets)


def generate_dataset_inputs(
    segments: list[ScrollSegment],
    z_start: int,
    z_end: int,
    tile_size: int,
    ink_label_dir: Path,
    should_blur_ink_labels: bool = False,
    ink_labels_blur_kernel_size: int = 17,
    min_ink_component_size: int = 1000,
    ink_erosion: int = 0,
    ignore_idx: int = -100,
    chunks_load: tuple[int, int, int] | int | bool = True,
    use_zarr: bool = True,
    zarr_dir: Path = ZARR_ARRAY_DIR,
    cache_memmaps: bool = False,
    memmap_dir: Path = MEMMAP_DIR,
    ink_preds_dir: Path | None = None,
    automatic_non_ink: bool = True,
    clip_min: int = 0,
    clip_max: int = 200,
) -> tuple[
    list[np.memmap],
    list[tuple[int, int]],
    list[bool],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray] | None,
    list[np.ndarray | None],
]:
    img_stack_refs = []
    segment_masks = []
    ink_masks = []
    non_ink_masks = []
    if ink_preds_dir is not None:
        ink_preds = []
    else:
        ink_preds = None

    pbar = tqdm(segments)
    padding_list = []
    z_reversals = []
    for segment in pbar:
        segment_name = (
            segment.segment_name if not segment.is_superseded else segment.segment_name_orig
        )
        pbar.set_description(f"Generating dataset inputs for {segment_name}...")
        pad_y, pad_x = compute_xy_padding(img_shape=segment.surface_shape, tile_size=tile_size)

        z_reverse = infer_z_reversal(segment_name)

        padding_list.append((pad_y, pad_x))
        z_reversals.append(z_reverse)

        if use_zarr:
            zarr_path = segment_to_zarr_path(segment, zarr_dir=zarr_dir)
            img_stack_ref = load_segment_as_zarr_array(zarr_path, chunks=chunks_load)
        elif cache_memmaps:
            mmap_path = segment_to_memmap_path(segment, memmap_dir=memmap_dir)
            img_stack_ref = load_segment_as_memmap(mmap_path)
        else:
            logging.debug(f"Creating image stack memmap...")
            img_stack_ref = create_img_stack_memmap(
                segment,
                z_min=z_start,
                z_max=z_end,
                pad_y=pad_y,
                pad_x=pad_x,
                z_reverse=z_reverse,
                clip_min=clip_min,
                clip_max=clip_max,
            )

        logging.debug(f"Loading segment mask...")
        segment_mask = load_segment_mask(segment, pad_y=pad_y, pad_x=pad_x)

        logging.debug(f"Loading ink mask...")
        ink_mask = load_ink_mask(
            ink_label_dir,
            segment_name,
            segment_mask.shape,
            should_blur_ink_labels=should_blur_ink_labels,
            ink_labels_blur_kernel_size=ink_labels_blur_kernel_size,
            min_ink_component_size=min_ink_component_size,
            ink_erosion=ink_erosion,
            ignore_idx=ignore_idx,
        )

        # Read ink predictions.
        if ink_preds is not None:
            ink_pred = read_ink_preds(
                segment_name, ink_preds_dir, expected_shape=segment_mask.shape
            )
            ink_preds.append(ink_pred)
            if ink_pred.shape != ink_mask.shape[1:]:
                raise ValueError(
                    f"Ink predictions must have the same shape as the ink mask. Segment {segment_name} ink "
                    f"preds has shape {ink_pred.shape} and ink labels shape {ink_mask.shape}."
                )

        # Read papyrus non-ink labels.
        non_ink_mask = load_papyrus_non_ink_labels(
            ink_label_dir, segment_name, automatic_non_ink, segment_mask.shape, ink_mask.shape
        )

        img_stack_refs.append(img_stack_ref)
        segment_masks.append(segment_mask)
        ink_masks.append(ink_mask)
        non_ink_masks.append(non_ink_mask)

    return (
        img_stack_refs,
        padding_list,
        z_reversals,
        segment_masks,
        ink_masks,
        ink_preds,
        non_ink_masks,
    )


def load_ink_mask(
    ink_label_dir: Path,
    segment_name: str,
    expected_shape: tuple[int, int],
    should_blur_ink_labels: bool = False,
    ink_labels_blur_kernel_size: int = 17,
    min_ink_component_size: int = 1000,
    ink_erosion: int = 0,
    ignore_idx: int = -100,
) -> np.ndarray:
    # Channel 0: default sampling
    # Channel 1: strict sampling
    # Channel 2: probabilistic sampling
    mask_paths = []
    for i in range(3):
        base_name = f"{segment_name}_inklabels"
        mask_path = ink_label_dir / f"{base_name}_{i}.png"
        if i == 0:
            # Check that we don't simultaneously have <segment_name>_inklabels.png or <segment_name>_inklabels_0.png
            mask_path_default = ink_label_dir / f"{base_name}.png"
            if mask_path.exists() and mask_path_default.exists():
                raise ValueError(
                    f"Found ambiguous channel 0 existing masks {mask_path} and {mask_path_default}. Please choose one."
                )
            elif mask_path.exists():
                mask_paths.append(mask_path)
            elif mask_path_default.exists():
                mask_paths.append(mask_path_default)
            else:
                logging.info(f"Didn't find channel {i} ink mask for {segment_name}.")
                mask_paths.append(None)
        else:
            if mask_path.exists():
                mask_paths.append(mask_path)
            else:
                logging.info(f"Didn't find channel {i} ink mask for {segment_name}.")
                mask_paths.append(None)

    # Check that mask 0 is not None if probabilistic one is not none
    if mask_paths[2] is not None:
        if mask_paths[0] is None:
            raise ValueError(
                f"Found channel 2 ink mask without channel 0 ink mask for segment {segment_name}."
            )

    if all(p is None for p in mask_paths):
        raise ValueError(f"No ink masks found for {segment_name}.")

    ink_mask_list = []
    for mask_path in mask_paths:
        if mask_path is not None:
            mask = read_ink_mask(
                mask_path,
                expected_shape=expected_shape,
                should_blur=should_blur_ink_labels,
                kernel_size=ink_labels_blur_kernel_size,
                min_component_size=min_ink_component_size,
                ink_erosion=ink_erosion,
                ignore_idx=ignore_idx,
            )
        else:
            mask = np.zeros(expected_shape, dtype=np.float32)
        ink_mask_list.append(mask)

    return np.stack(ink_mask_list)


def load_papyrus_non_ink_labels(
    ink_label_dir: Path,
    segment_name: str,
    automatic_non_ink: bool,
    expected_shape: tuple[int, int],
    ink_mask_shape: tuple[int, int],
) -> np.ndarray | None:
    non_ink_mask_path = ink_label_dir / f"{segment_name}_papyrusnoninklabels.png"
    if non_ink_mask_path.exists():
        logging.info(f"Loading papyrus non-ink mask for segment {segment_name}.")
        non_ink_mask = read_papy_non_ink_labels(non_ink_mask_path, expected_shape=expected_shape)
        if non_ink_mask.shape != ink_mask_shape[1:]:
            raise ValueError(
                f"Non-ink mask must have the same shape as the ink mask. Segment {segment_name} ink "
                f"non-ink mask has shape {non_ink_mask.shape} and ink labels shape {ink_mask_shape}."
            )
    else:
        if not automatic_non_ink:
            raise FileNotFoundError(f"Non-ink mask file does not exist: {non_ink_mask_path}")
        else:
            logging.info(
                f"Automatically generated non-ink mask will be used for segment {segment_name}."
            )
        non_ink_mask = None
    return non_ink_mask


def infer_z_reversal(segment_name: str) -> bool:
    z_reverse = False
    segment_name_base_part = segment_name.split("_C")[0].split("_superseded")[0]
    if segment_name_base_part not in KNOWN_Z_ORIENTATION_SEGMENT_IDS:
        warn(f"Segment {segment_name} has an unknown z-orientation and will not be flipped.")

    if segment_name_base_part in Z_REVERSED_SEGMENT_IDS:
        logging.info(f"Reversing z-orientation of segment {segment_name}.")
        z_reverse = True

    return z_reverse


def default_patch_sample(
    ink_mask: np.ndarray,
    segment_mask: np.ndarray,
    size: int,
    stride: int,
    dilation_kernel_size: int,
    min_labeled_coverage_frac: float = 0.5,
    ink_label_thresh: float = 0.05,
    tile_size: int = 256,
    ink_preds: np.ndarray | None = None,
    model_based_ink_correction_thresh: float = 0.1,
    model_based_non_ink_correction_thresh: float = 0.3,
    clean_up_ink_labels: bool = True,
    clean_up_non_ink_labels: bool = True,
    ink_mask_channel_2: np.ndarray | None = None,
    p_0_ink: float = 0.3,
    p_2_ink: float = 0.6,
    p_non_ink: float = 0.1,
    ignore_idx: int = -100,
    non_ink_mask: np.ndarray | None = None,
) -> tuple[list[tuple[int, int, int, int]], list[bool]]:
    if non_ink_mask is None:
        non_ink_mask = create_non_ink_mask(ink_mask, dilation_kernel_size=dilation_kernel_size)

    if ink_preds is not None:
        ink_mask, non_ink_mask = apply_model_based_label_correction(
            ink_mask,
            non_ink_mask,
            ink_preds,
            ink_label_thresh=ink_label_thresh,
            model_based_ink_correction_thresh=model_based_ink_correction_thresh,
            model_based_non_ink_correction_thresh=model_based_non_ink_correction_thresh,
            clean_up_ink_labels=clean_up_ink_labels,
            clean_up_non_ink_labels=clean_up_non_ink_labels,
        )
    else:
        if clean_up_ink_labels or clean_up_non_ink_labels:
            logging.warning(
                f"No ink predictions were found. Skipping model-based label correction."
            )

    if ink_mask_channel_2 is None:
        patch_positions_seg = get_valid_patch_positions_batched(
            ink_mask,
            non_ink_mask,
            segment_mask,
            patch_shape=(size, size),
            patch_stride=stride,
            min_labeled_coverage_frac=min_labeled_coverage_frac,
            chunk_size=tile_size,
            should_pad=True,
            ink_thresh=ink_label_thresh,
        )
    else:
        patch_positions_seg = get_probabilistic_patch_samples(
            ink_mask,
            ink_mask_channel_2,
            non_ink_mask,
            size,
            stride,
            p0=p_0_ink,
            p2=p_2_ink,
            p_non_ink=p_non_ink,
            ignore_idx=ignore_idx,
        )

    non_ink_ignored_patches_seg = [False] * len(patch_positions_seg)
    return patch_positions_seg, non_ink_ignored_patches_seg


def all_ink_patch_sample(
    ink_mask: np.ndarray,
    segment_mask: np.ndarray,
    size: int,
    stride: int,
    tile_size: int = 256,
) -> tuple[list, list[bool]]:
    patch_positions_seg = get_ink_patch_positions_batched(
        ink_mask,
        segment_mask,
        patch_shape=(size, size),
        patch_stride=stride,
        chunk_size=tile_size,
        should_pad=True,
        all_ink_patches=True,
    )
    non_ink_ignored_patches_seg = [True] * len(patch_positions_seg)
    return patch_positions_seg, non_ink_ignored_patches_seg


def any_ink_patch_sample(
    ink_mask: np.ndarray,
    segment_mask: np.ndarray,
    size: int,
    stride: int,
    tile_size: int = 256,
) -> tuple[list, list[bool]]:
    patch_positions_seg = get_ink_patch_positions_batched(
        ink_mask,
        segment_mask,
        patch_shape=(size, size),
        patch_stride=stride,
        chunk_size=tile_size,
        should_pad=True,
        all_ink_patches=False,
    )
    non_ink_ignored_patches_seg = [True] * len(patch_positions_seg)
    return patch_positions_seg, non_ink_ignored_patches_seg


def create_patch_position_data(
    segment_masks: list[np.ndarray],
    ink_masks: list[np.ndarray],
    ink_dilation_kernel_size: int,
    size: int,
    stride: int,
    stride_strict: int | None = None,
    strict_sampling_only_ink: bool = True,
    split: str | None = None,
    tile_size: int = 256,
    min_labeled_coverage_frac: float = 0.5,
    ink_preds: list[np.ndarray] | None = None,
    model_based_ink_correction_thresh: float = 0.1,
    model_based_non_ink_correction_thresh: float = 0.3,
    clean_up_ink_labels: bool = True,
    clean_up_non_ink_labels: bool = True,
    p_0_ink: float = 0.3,
    p_2_ink: float = 0.6,
    p_non_ink: float = 0.6,
    ignore_idx: int = -100,
    non_ink_masks: list[np.ndarray | None] | None = None,
) -> tuple[list[list[int, int, int, int]], list[list[bool]]]:
    def format_str(left: str, right: str, middle: str | None = None):
        seq = (left, right) if middle is None else (left, middle, right)
        return " ".join(seq)

    if ink_preds is None:
        ink_preds_placeholder = [None] * len(segment_masks)
    else:
        ink_preds_placeholder = ink_preds

    if non_ink_masks is None:
        non_ink_masks_placeholder = [None] * len(segment_masks)
    else:
        non_ink_masks_placeholder = non_ink_masks

    non_ink_ignored_patches = []
    patch_positions = []
    pbar = tzip(
        segment_masks,
        ink_masks,
        ink_preds_placeholder,
        non_ink_masks_placeholder,
        desc=format_str(left="Creating", middle=split, right="patch positions..."),
    )
    for segment_mask, ink_mask, ink_pred, non_ink_mask in pbar:
        positions = []
        non_ink_ignored_patches_multi_channel = []

        # 1. Default sampling first.
        ink_mask_channel_0 = ink_mask[0]
        if ink_mask_channel_0[ink_mask_channel_0 != ignore_idx].any():
            ink_mask_channel_2 = ink_mask[2]
            has_ink_channel_2 = ink_mask_channel_2[ink_mask_channel_2 != ignore_idx].any()
            patch_positions_seg, non_ink_ignored_patches_seg = default_patch_sample(
                ink_mask_channel_0,
                segment_mask,
                size,
                stride,
                ink_dilation_kernel_size,
                tile_size=tile_size,
                min_labeled_coverage_frac=min_labeled_coverage_frac,
                ink_preds=ink_pred,
                model_based_ink_correction_thresh=model_based_ink_correction_thresh,
                model_based_non_ink_correction_thresh=model_based_non_ink_correction_thresh,
                clean_up_ink_labels=clean_up_ink_labels,
                clean_up_non_ink_labels=clean_up_non_ink_labels,
                ink_mask_channel_2=ink_mask_channel_2 if has_ink_channel_2 else None,
                p_0_ink=p_0_ink,
                p_2_ink=p_2_ink,
                p_non_ink=p_non_ink,
                ignore_idx=ignore_idx,
                non_ink_mask=non_ink_mask,
            )
            sampling_type = "probabilistic-based" if has_ink_channel_2 else "default"
            logging.info(
                format_str(
                    left=f"Created {len(patch_positions_seg)}",
                    middle=split,
                    right=f"{sampling_type} patch positions for ink mask channel 0.",
                )
            )
            positions += patch_positions_seg
            non_ink_ignored_patches_multi_channel += non_ink_ignored_patches_seg
        else:
            logging.info(f"Skipping default sampling for ink mask.")

        # 2. Strict sampling.
        ink_mask_channel_1 = ink_mask[1]
        if ink_mask_channel_1[ink_mask_channel_1 != ignore_idx].any():
            channel_stride = stride if stride_strict is None else stride_strict
            if strict_sampling_only_ink:
                patch_positions_seg, non_ink_ignored_patches_seg = all_ink_patch_sample(
                    ink_mask_channel_1,
                    segment_mask,
                    size,
                    channel_stride,
                    tile_size=tile_size,
                )
            else:
                patch_positions_seg, non_ink_ignored_patches_seg = any_ink_patch_sample(
                    ink_mask_channel_1,
                    segment_mask,
                    size,
                    channel_stride,
                    tile_size=tile_size,
                )
            logging.info(
                format_str(
                    left=f"Created {len(patch_positions_seg)}",
                    middle=split,
                    right=f"strict patch positions for ink mask channel 1.",
                )
            )
            positions += patch_positions_seg
            non_ink_ignored_patches_multi_channel += non_ink_ignored_patches_seg
        else:
            logging.info(f"Skipping strict sampling for ink mask.")

        non_ink_ignored_patches.append(non_ink_ignored_patches_multi_channel)
        logging.info(
            format_str(left=f"Created {len(positions)}", middle=split, right="patch positions.")
        )
        patch_positions.append(positions)

    return patch_positions, non_ink_ignored_patches


create_train_patch_positions = partial(create_patch_position_data, split="train")
create_val_patch_positions = partial(create_patch_position_data, split="validation")


def load_segment_mask(segment: ScrollSegment, pad_y: int = 0, pad_x: int = 0) -> np.ndarray:
    segment_mask = segment.load_mask()
    segment_mask = np.pad(segment_mask, [(0, pad_y), (0, pad_x)], constant_values=0)
    if "frag" in segment.segment_name:
        segment_mask = cv2.resize(
            segment_mask,
            (segment_mask.shape[1] // 2, segment_mask.shape[0] // 2),
            interpolation=cv2.INTER_AREA,
        )
    return segment_mask


def compute_xy_padding(img_shape: tuple[int, int], tile_size: int) -> tuple[int, int]:
    pad_y = tile_size - img_shape[0] % tile_size
    pad_x = tile_size - img_shape[1] % tile_size
    return pad_y, pad_x


def preprocess_image_stack_memmap(
    img_stack_ref: np.memmap,
    pad_y: int,
    pad_x: int,
    z_reverse: bool = False,
    clip_min: int = 0,
    clip_max: int = 200,
    depth_dim_last: bool = True,
    memmap_prefix: str | None = None,
):
    # Rescale intensities and convert to uint8.
    img_stack_new = img_as_ubyte(img_stack_ref)

    # Pad image.
    img_stack_new = np.pad(img_stack_new, [(0, 0), (0, pad_y), (0, pad_x)], constant_values=0)

    # Reverse along z-dimension if necessary
    if z_reverse:
        img_stack_new = img_stack_new[::-1]

    # Clip intensities
    img_stack_new = np.clip(img_stack_new, clip_min, clip_max)

    # Swap axes (move depth dimension to be last).
    if depth_dim_last:
        img_stack_new = np.moveaxis(img_stack_new, 0, -1)

    # Create new memmap
    memmap_dir = Path(os.environ.get("MEMMAP_DIR", mkdtemp()))
    prefix = Path(img_stack_ref.filename).stem if memmap_prefix is None else memmap_prefix
    file_name = memmap_dir / f"{prefix}_processed.dat"
    new_memmap = np.memmap(
        str(file_name), dtype=img_stack_new.dtype, mode="w+", shape=img_stack_new.shape
    )
    new_memmap[:] = img_stack_new

    new_memmap.flush()

    return new_memmap


def create_img_stack_memmap(
    segment: ScrollSegment,
    z_min: int | None = None,
    z_max: int | None = None,
    pad_y: int = 0,
    pad_x: int = 0,
    z_reverse: bool = False,
    clip_min: int = 0,
    clip_max: int = 200,
    depth_dim_last: bool = True,
) -> np.memmap:
    logging.debug(f"Loading original volume as memmap...")
    img_stack_orig = segment.load_volume_as_memmap(z_min, z_max)

    logging.debug(f"Creating new preprocessed image stack memmap...")
    img_stack_ref_new = preprocess_image_stack_memmap(
        img_stack_orig,
        pad_y,
        pad_x,
        z_reverse=z_reverse,
        clip_min=clip_min,
        clip_max=clip_max,
        depth_dim_last=depth_dim_last,
        memmap_prefix=segment.segment_name_orig,
    )

    logging.debug(f"Deleting original memmap...")
    delete_memmap(img_stack_orig)

    return img_stack_ref_new


def _check_ink_labels(ink_label_dir: Path, labeled_segment_names: list[str]) -> None:
    if not ink_label_dir.is_dir():
        raise ValueError("Ink label directory does not exist")

    segment_names_in_ink_labels = {
        p.stem.split("_inklabels")[0] for p in ink_label_dir.glob("*_inklabels*.png")
    }
    labeled_segment_names = set(labeled_segment_names)
    if not labeled_segment_names.issubset(segment_names_in_ink_labels):
        missing_ink_labels = labeled_segment_names - segment_names_in_ink_labels
        raise ValueError(
            f"Input segments must be a subset of ink label segments. Missing ink labels for the "
            f"following segments: {missing_ink_labels}."
        )


def _check_ink_pred_dir(ink_pred_dir: Path, labeled_segment_names: list[str]) -> None:
    if not ink_pred_dir.is_dir():
        raise ValueError("Ink prediction directory does not exist")

    segment_names_in_ink_preds = {
        p.stem.split("_inklabels")[0] for p in ink_pred_dir.glob("*_inklabels*.png")
    }
    segment_names_in_ink_labels = set(labeled_segment_names)
    if not segment_names_in_ink_labels.issubset(segment_names_in_ink_preds):
        missing_ink_preds = segment_names_in_ink_labels - segment_names_in_ink_preds
        raise ValueError(
            f"Ink label segments must be a subset of ink prediction segments. Missing predictions for the "
            f"following segments: {missing_ink_preds}."
        )


def _check_papyrus_non_ink_labels(ink_labels_dir: Path, labeled_segment_names: list[str]) -> None:
    if not ink_labels_dir.is_dir():
        raise ValueError("Ink labels directory does not exist")

    # Check the every labeled segment name has a papyrus non-ink label.
    non_ink_segment_names = {
        p.stem.split("_papyrusnoninklabels")[0]
        for p in ink_labels_dir.glob("*_papyrusnoninklabels.png")
    }
    segment_name_set = set(labeled_segment_names)
    if not segment_name_set.issubset(non_ink_segment_names):
        missing_non_ink_masks = segment_name_set - non_ink_segment_names
        raise ValueError(
            f"Each ink label must have a corresponding papyrus non-ink label. Missing non-ink "
            f"labels for the following segments: {missing_non_ink_masks}"
        )
