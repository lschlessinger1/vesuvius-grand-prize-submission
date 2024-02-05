from typing import Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryFBetaScore,
)

from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.models import UNet3Dto2D

BatchType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
BatchTypePrediction = tuple[torch.Tensor, torch.Tensor]


class PatchLitModel(pl.LightningModule):
    """LightningModule for handling Patch-based 3D to 2D UNet for ink detection.

    Attributes:
        model: The underlying 3D to 2D UNet model.
        loss_fn: Binary Cross Entropy loss with logits.
        lr: Learning rate.
        train_metrics: Metrics for training phase.
        val_metrics: Metrics for validation phase.
    """

    def __init__(
        self,
        loss_fn: nn.Module | None = None,
        lr: float = 1e-3,
        thresh: float = 0.5,
        f_maps: int = 64,
        layer_order: str = "gcr",
        num_groups: int = 8,
        num_levels: int = 4,
        conv_padding: int | tuple[int, ...] = 1,
        se_type_str: str | None = "CSE3D",
        reduction_ratio: int = 2,
        depth_dropout: float = 0.0,
        pool_fn: Literal["mean", "max"] = "mean",
    ):
        """Initializes the PatchLitModel.

        Args:
            loss_fn (nn.Module): Loss function. Defaults to nn.BCEWithLogitsLoss().
            lr (float): Learning rate. Defaults to 1e-3.
            thresh (float): Threshold for binary classification. Defaults to 0.5.
            f_maps (int): Number of feature maps. Defaults to 64.
            layer_order (str): Order of layer in the model. Defaults to "gcr".
            num_groups (int): Number of groups for grouped convolutions. Defaults to 8.
            num_levels (int): Number of levels in the model. Defaults to 4.
            conv_padding (int | tuple[int, ...]): Convolution padding. Defaults to 1.
            se_type_str (str | None): Squeeze-and-Excitation type string. Defaults to "CSE3D".
            reduction_ratio (int): Reduction ratio for squeeze-and-excitation. Defaults to 2.
            depth_dropout (float): Depth dropout value. Defaults to 0.0.
            pool_fn (Literal["mean", "max"]): Pooling function to be used ("mean" or "max"). Defaults to "mean".
        """
        super().__init__()
        if loss_fn is None:
            loss_fn = nn.BCEWithLogitsLoss()

        self.model = UNet3Dto2D(
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            conv_padding=conv_padding,
            se_type_str=se_type_str,
            reduction_ratio=reduction_ratio,
            depth_dropout=depth_dropout,
            pool_fn=pool_fn,
            output_features=False,
        )
        self.loss_fn = loss_fn
        self.lr = lr
        shared_metrics = MetricCollection(
            [
                BinaryAccuracy(threshold=thresh),
                BinaryFBetaScore(beta=0.5, threshold=thresh),
            ]
        )
        self.train_metrics = shared_metrics.clone(prefix="train/")
        self.val_metrics = shared_metrics.clone(prefix="val/")
        self.val_metrics.add_metrics([BinaryAUROC(), BinaryAveragePrecision()])
        self.save_hyperparameters(ignore=["loss_fn"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits.
        """
        x = x.unsqueeze(1)  # Add dummy channel dimension because it's grayscale.
        outputs = self.model(x)
        logits = outputs.logits.squeeze(1)  # Remove dummy channel
        return logits

    def _step(self, batch: BatchType) -> BatchType:
        """Handles a single step of training or validation.

        This method processes a given batch to obtain logits, labels, and patch positions.

        Args:
            batch (tuple): Batch of data.

        Returns:
            tuple: Logits, labels, and patch positions.
        """
        x, y, patch_pos = batch
        y = y.float()
        logits = self(x)
        return logits, y, patch_pos

    def training_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        """Training step for the model.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        logits, y, _ = self._step(batch)
        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss)
        train_metrics_output = self.train_metrics(logits, y)
        self.log_dict(train_metrics_output)
        return loss

    def validation_step(self, batch: BatchType, batch_idx: int) -> dict[str, torch.Tensor]:
        """Validation step for the model.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            dict: Validation outputs.
        """
        logits, y, patch_pos = self._step(batch)
        loss = self.loss_fn(logits, y)
        self.log("val/loss", loss)

        y = y.int()
        self.val_metrics.update(logits, y)
        return {"logits": logits, "y": y, "loss": loss, "patch_pos": patch_pos}

    def on_validation_epoch_end(self) -> None:
        """Method called at the end of a validation epoch."""
        val_metrics_output = self.val_metrics.compute()
        self.log_dict(val_metrics_output)
        self.val_metrics.reset()

    def predict_step(
        self, batch: BatchType | BatchTypePrediction, batch_idx: int, dataloader_idx: int = 0
    ) -> BatchTypePrediction:
        """Prediction step for the model.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.
            dataloader_idx (int, optional): Index of the data loader. Defaults to 0.

        Returns:
            tuple: Probabilities, labels, and patch positions.
        """
        if len(batch) == 3:
            logits, _, patch_pos = self._step(batch)
        elif len(batch) == 2:
            x, patch_pos = batch
            logits = self(x)
        else:
            raise ValueError("Expected number of items in a batch to be 2 or 3.")

        y_proba = F.sigmoid(logits)
        return y_proba, patch_pos

    def configure_optimizers(self) -> Optimizer:
        """Configures the optimizers.

        Returns:
            torch.optim.Optimizer: Optimizer for the model.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
