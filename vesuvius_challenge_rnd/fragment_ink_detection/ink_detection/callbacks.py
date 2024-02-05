from typing import Any, Optional

import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_lightning import Callback
from pytorch_lightning.core.saving import load_hparams_from_yaml
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT

from vesuvius_challenge_rnd.data import Fragment
from vesuvius_challenge_rnd.data.visualization import create_gif_from_2d_single_channel_images
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.data.base_fragment_data_module import (
    AbstractFragmentValPatchDataset,
)
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.data.patch_dataset import (
    PatchDataset,
)
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.inference.patch_aggregation import (
    patches_to_y_proba,
)


class CheckValDataModule(Callback):
    """Base callback class that ensures a data module is available."""

    def __init__(self):
        self._data_module: pl.LightningDataModule | None = None

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """Check for data module at the beginning of the validation phase."""
        data_module = getattr(trainer, "datamodule", None)
        if data_module is None:
            raise ValueError("A data module must be provided for validation in this callback.")
        self._data_module = data_module


class WandbLogPredictionSamplesCallback(CheckValDataModule):
    """
    Callback to log prediction samples to Weights & Biases during validation.
    """

    def __init__(self, downsize_factor: int = 25):
        """
        Initialize the callback for logging prediction samples to WandB.

        Args:
            downsize_factor (int, optional): The downsize factor for the output image. Defaults to 25.
        """
        super().__init__()
        self.validation_y_proba_patches: list[torch.Tensor] = []
        self.validation_patch_positions: list[torch.Tensor] = []
        self.y_proba_smoothed_seq: list[np.ndarray] = []
        self.downsize_factor = downsize_factor

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Collect validation data at the end of each validation batch.

        Args:
            trainer (pl.Trainer): PyTorch Lightning Trainer instance.
            pl_module (pl.LightningModule): The model being trained.
            outputs (STEP_OUTPUT | None): Outputs of the validation step.
            batch (Any): Current batch data.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int, optional): Index of the current dataloader. Defaults to 0.
        """
        if dataloader_idx == 0:
            y_proba_patches = F.sigmoid(outputs["logits"])
            patch_positions = outputs["patch_pos"]
            self.validation_y_proba_patches.extend(y_proba_patches)
            self.validation_patch_positions.extend(patch_positions)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        Log prediction samples at the end of the validation epoch.

        Args:
            trainer (pl.Trainer): PyTorch Lightning Trainer instance.
            pl_module (pl.LightningModule): The model being trained.
        """
        logger = trainer.logger
        if isinstance(logger, WandbLogger):
            y_proba_patches = np.stack(
                [p.detach().cpu().float().numpy() for p in self.validation_y_proba_patches], axis=0
            )
            patch_positions = np.stack(
                [p.detach().cpu().numpy() for p in self.validation_patch_positions], axis=0
            )

            dataset_val = self._get_val_dataset_if_possible()
            fragment = self._get_val_fragment_if_possible(dataset_val)

            mask = dataset_val.masks[0]
            y_proba_smoothed = patches_to_y_proba(
                y_proba_patches, patch_positions, mask, dataset_val.patch_shape
            )
            self.y_proba_smoothed_seq.append(y_proba_smoothed)

            # Apply the Viridis colormap to the normalized probability map
            cmap = plt.cm.get_cmap("viridis")
            output_image = Image.fromarray((cmap(y_proba_smoothed) * 255).astype(np.uint8))

            ink_labels_img = fragment.load_ink_labels_as_img()
            new_shape = list(x // self.downsize_factor for x in fragment.surface_shape)

            # Create thumbnails.
            ink_labels_img.thumbnail(new_shape, Image.ANTIALIAS)
            output_image.thumbnail(new_shape, Image.ANTIALIAS)
            mask_img = fragment.load_mask_as_img()
            mask_img.thumbnail(new_shape, Image.ANTIALIAS)
            image = wandb.Image(
                output_image,
                masks={
                    "ground_truth": {
                        "mask_data": np.array(ink_labels_img),
                        "class_labels": {0: "not ink", 1: "ink"},
                    },
                    "papyrus_mask": {
                        "mask_data": np.array(mask_img) + 2,  # Convert to {2, 3} for visualization.
                        "class_labels": {2: "not papyrus", 3: "papyrus"},
                    },
                },
                caption=f"Predicted fragment {fragment.fragment_id} ink probabilities",
            )

            metrics = {"Masked predictions": [image]}
            # TODO: step in w&b UI should be epoch.
            if not trainer.sanity_checking:
                logger.log_metrics(metrics)
        else:
            raise TypeError(f"Expected a WandbLogger. Found type {type(logger)}.")

        # Reset.
        self.validation_y_proba_patches = []
        self.validation_patch_positions = []

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger = trainer.logger
        if isinstance(logger, WandbLogger):
            dataset_val = self._get_val_dataset_if_possible()
            fragment = self._get_val_fragment_if_possible(dataset_val)

            logging.info(f"Saving gif of fragment {fragment.fragment_id} predictions...")

            frames = np.array(self.y_proba_smoothed_seq)
            gif_path = (
                Path(trainer.log_dir) / f"y_probas_smoothed_fragment_{fragment.fragment_id}.gif"
            )
            create_gif_from_2d_single_channel_images(
                frames,
                out_path=str(gif_path),
                scale_factor=255,
                downsample_factor=0.1,
                duration=250,
                downsample_order=2,
            )
            logging.info(
                f"Saved gif of fragment {fragment.fragment_id} predictions to {gif_path.resolve()}."
            )

            image = wandb.Image(
                str(gif_path),
                caption=f"Predicted fragment {fragment.fragment_id} ink probabilities",
            )
            metrics = {"y_probas_smoothed": [image]}
            if not trainer.sanity_checking:
                logger.log_metrics(metrics)
        else:
            raise TypeError(f"Expected a `WandbLogger`. Found type {type(logger)}.")

    def _get_val_dataset_if_possible(self) -> PatchDataset:
        datamodule = self._data_module
        if not isinstance(datamodule, AbstractFragmentValPatchDataset):
            raise TypeError(
                f"data module must be an instance of {type(AbstractFragmentValPatchDataset).__name__}. "
                f"Found a type of {type(datamodule)}."
            )
        dataset = datamodule.val_fragment_dataset
        return dataset

    def _get_val_fragment_if_possible(self, val_dataset: PatchDataset) -> Fragment:
        fragment = val_dataset.fragments[0]
        n_val_fragments = len(val_dataset.fragments)
        if n_val_fragments > 1:
            warnings.warn(
                f"This callback only supports a single validation fragment. Found {n_val_fragments} "
                f"fragments. It will default to using the first fragment ({fragment.fragment_id})."
            )
        return fragment


class WandbSavePRandROCCallback(Callback):
    """
    Callback to log Precision-Recall and ROC curves to Weights & Biases during validation.
    """

    def __init__(self, threshold: float = 0.5, ignore_index: int = -100):
        """
        Initialize the callback for saving PR and ROC curves.
        """
        if not (0 <= threshold <= 1):
            raise ValueError("`threshold` must be in the unit interval [0, 1].")
        self.validation_y_proba_patches: list[torch.Tensor] = []
        self.validation_y_true_patches: list[torch.Tensor] = []
        self.threshold = threshold
        self.ignore_index = ignore_index

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Collect validation data at the end of each validation batch.

        Args:
            trainer (pl.Trainer): PyTorch Lightning Trainer instance.
            pl_module (pl.LightningModule): The model being trained.
            outputs (STEP_OUTPUT | None): Outputs of the validation step.
            batch (Any): Current batch data.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int, optional): Index of the current dataloader. Defaults to 0.
        """
        if dataloader_idx == 0:
            y_proba_patches = F.sigmoid(outputs["logits"])
            y_true_patches = outputs["y"]
            self.validation_y_proba_patches.extend(y_proba_patches)
            self.validation_y_true_patches.extend(y_true_patches)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        Log PR and ROC curves to W&B at the end of the validation epoch.

        Args:
            trainer (pl.Trainer): PyTorch Lightning Trainer instance.
            pl_module (pl.LightningModule): The model being trained.
        """
        logger = trainer.logger
        if isinstance(logger, WandbLogger):
            y_proba_patches = np.stack(
                [p.detach().cpu().float().numpy() for p in self.validation_y_proba_patches], axis=0
            )
            y_true = np.stack(
                [p.detach().cpu().numpy() for p in self.validation_y_true_patches], axis=0
            )

            valid_indices = y_true != self.ignore_index
            y_true = y_true[valid_indices]
            y_proba_patches = y_proba_patches[valid_indices]

            y_proba_patches = y_proba_patches.flatten()
            y_true = y_true.flatten()
            y_probas_2d = np.vstack([1 - y_proba_patches, y_proba_patches]).T
            labels = ["not ink", "ink"]
            classes_to_plot = [1]  # Plot ink only.
            if not trainer.sanity_checking:
                try:
                    if np.any(y_true > 0):  # Only possible if we have some ink labels.
                        # Possibly binarize true labels.
                        unique_values = np.unique(y_true)
                        non_integer_in_y_true = np.any(unique_values != np.floor(unique_values))
                        if non_integer_in_y_true:
                            y_true = (y_true >= self.threshold).astype(int)
                            wandb.termwarn(
                                f"Binarizing labels with thresh={self.threshold} because non-integer values are present in y_true.",
                                repeat=False,
                            )

                        logger.log_metrics(
                            {
                                "roc-curve": wandb.plot.roc_curve(
                                    y_true,
                                    y_probas_2d,
                                    labels=labels,
                                    classes_to_plot=classes_to_plot,
                                )
                            }
                        )
                        logger.log_metrics(
                            {
                                "pr-curve": wandb.plot.pr_curve(
                                    y_true,
                                    y_probas_2d,
                                    labels=labels,
                                    classes_to_plot=classes_to_plot,
                                )
                            }
                        )
                    else:
                        wandb.termwarn(
                            "Skipping plotting of ROC and PR curves due no ink being present in the validation set.",
                            repeat=False,
                        )
                except FileNotFoundError:
                    wandb.termwarn(
                        "File not found error while logging ROC curve and PR curve.", repeat=False
                    )
        else:
            raise TypeError(f"Expected a WandbLogger. Found type {type(logger)}.")

        # Reset.
        self.validation_y_proba_patches = []
        self.validation_y_true_patches = []


class WandbSaveConfigCallback(Callback):
    """
    Callback to save the configuration file as an artifact in Weights & Biases.
    """

    def __init__(self, config_filename: str = "config_pl.yaml"):
        """
        Initialize the callback for saving configuration.

        Args:
            config_filename (str, optional): Name of the config file to save. Defaults to "config_pl.yaml".
        """
        self.config_filename = config_filename

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Save configuration at the start of training.

        Args:
            trainer (pl.Trainer): PyTorch Lightning Trainer instance.
            pl_module (pl.LightningModule): The model being trained.
        """
        logger = trainer.logger
        if isinstance(logger, WandbLogger):
            config_path = os.path.join(trainer.log_dir, self.config_filename)

            if os.path.isfile(config_path):
                artifact = wandb.Artifact(name="config", type="dataset")
                artifact.add_file(config_path)
                logger.experiment.log_artifact(artifact)
                logging.info(f"Saved config ({config_path}) artifact to W&B: {artifact}")

                # Log config as hyperparameters to W&B config.
                config = load_hparams_from_yaml(config_path, use_omegaconf=False)
                trainer.logger.log_hyperparams({"config": config})
