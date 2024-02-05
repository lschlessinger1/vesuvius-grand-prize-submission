from typing import Literal

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryFBetaScore,
    F1Score,
)

from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.models import UNet3Dto2D
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.models.unet_3d_to_2d import (
    compute_unet_3d_to_2d_encoder_chwd,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.models import GradientReversal
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.models.mlp import FlattenAndMLP


class LitDomainAdversarialSegmenter(pl.LightningModule):
    def __init__(
        self,
        patch_depth: int,
        patch_height: int,
        patch_width: int,
        num_classes: int = 1,
        in_channels: int = 1,
        lr: float = 0.0001,
        thresh: float = 0.5,
        task_loss_fn: nn.Module | None = None,
        f_maps: int = 64,
        layer_order: str = "gcr",
        num_groups: int = 8,
        num_levels: int = 4,
        conv_padding: int | tuple[int, ...] = 1,
        se_type_str: str | None = "PE",
        reduction_ratio: int = 2,
        depth_dropout: float = 0.0,
        pool_fn: Literal["mean", "max"] = "mean",
        gamma: float = 10.0,  # Adaptation factor
        dc_input_encoder_depth: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        if task_loss_fn is None:
            task_loss_fn = nn.BCEWithLogitsLoss()

        self.segmenter = UNet3Dto2D(
            in_channels=in_channels,
            out_channels=num_classes,
            se_type_str=se_type_str,
            depth_dropout=depth_dropout,
            pool_fn=pool_fn,
            f_maps=f_maps,
            num_levels=num_levels,
            reduction_ratio=reduction_ratio,
            conv_padding=conv_padding,
            num_groups=num_groups,
            layer_order=layer_order,
            output_features=True,
        )

        if dc_input_encoder_depth > num_levels:
            raise ValueError(
                f"Domain classifier input encoder depth ({dc_input_encoder_depth}) cannot be greater than U-Net depth (num_levels={num_levels})."
            )

        self.dc_input_encoder_depth = dc_input_encoder_depth
        c, h, w, d = compute_unet_3d_to_2d_encoder_chwd(
            patch_depth,
            patch_height,
            patch_width,
            f_maps,
            encoder_level=self.dc_input_encoder_depth,
            in_channels=in_channels,
        )
        self.domain_classifier = FlattenAndMLP(
            input_dim=int(c * h * w * d),
            output_dim=num_classes,
        )
        self.loss_fn_sup = task_loss_fn
        self.loss_fn_dc = nn.BCEWithLogitsLoss()
        self.grl = GradientReversal(alpha=0.0)

        self.lr = lr
        shared_fragment_metrics = MetricCollection(
            [
                BinaryAccuracy(threshold=thresh),
                BinaryFBetaScore(beta=0.5, threshold=thresh),
            ]
        )
        self.train_fragment_metrics = shared_fragment_metrics.clone(prefix="train/")
        self.val_fragment_metrics = shared_fragment_metrics.clone(prefix="val/")
        self.val_fragment_metrics.add_metrics([BinaryAUROC(), BinaryAveragePrecision()])

        # Domain classifier metrics.
        shared_dc_metrics = MetricCollection(
            [
                BinaryAccuracy(threshold=thresh),
                F1Score(task="binary", threshold=thresh, num_classes=2),
                BinaryAUROC(),
                BinaryAveragePrecision(),
            ]
        )
        self.train_dc_metrics = shared_dc_metrics.clone(prefix="train/domain_classifier/")

        # B x C x D x H x W
        self.example_input_array = torch.Tensor(
            5, in_channels, patch_depth, patch_height, patch_width
        )
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.ndim == 4:
            x = x.unsqueeze(1)  # Add dummy channel dimension because it's grayscale.
        output = self.segmenter(x)
        logits = output.logits.squeeze(1)
        features = output.encoder_features[self.dc_input_encoder_depth]
        return logits, features

    def adversarial_classifier(self, x: torch.Tensor) -> torch.Tensor:
        x = self.grl(x)
        logits = self.domain_classifier(x)
        return logits

    def training_step(self, batch, batch_idx: int):
        # Compute new alpha.
        num_batches = self.trainer.num_training_batches
        current_steps = self.current_epoch * num_batches
        max_steps = self.trainer.max_epochs * num_batches
        p = (batch_idx + current_steps) / max_steps
        alpha = calculate_dann_alpha(self.gamma, p)
        self.grl.update_alpha(alpha)
        self.log("alpha", alpha)

        # Prepare batch.
        source_patch, source_labels, _ = batch["source"]
        target_patch, _ = batch["target"]
        patches = torch.cat((source_patch, target_patch), dim=0)
        source_batch_size = source_patch.shape[0]
        source_domain_labels = torch.zeros(source_batch_size, dtype=torch.float, device=self.device)
        target_domain_labels = torch.ones(
            target_patch.shape[0], dtype=torch.float, device=self.device
        )
        domain_labels = torch.cat((source_domain_labels, target_domain_labels))
        source_labels = source_labels.float()

        # Label predictor.
        logits, features = self(patches)
        source_logits = logits[:source_batch_size]
        loss_sup = self.loss_fn_sup(source_logits, source_labels)
        self.log("train/task_loss", loss_sup, prog_bar=True)
        train_fragment_metrics_output = self.train_fragment_metrics(source_logits, source_labels)
        self.log_dict(train_fragment_metrics_output)

        # Domain classifier.
        domain_logits = self.adversarial_classifier(features).squeeze()
        loss_dc = self.loss_fn_dc(domain_logits, domain_labels)
        self.log(f"{self.train_dc_metrics.prefix}loss", loss_dc)
        train_dc_metrics_output = self.train_dc_metrics(domain_logits, domain_labels.int())
        self.log_dict(train_dc_metrics_output)

        loss = loss_sup + loss_dc
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            # For now, only validate on source/fragments.
            source_patch, source_labels, source_patch_pos = batch
            source_labels = source_labels.float()
            source_logits, _ = self(source_patch)

            task_loss = self.loss_fn_sup(source_logits, source_labels)
            self.log("val/loss", task_loss)

            source_labels = source_labels.int()
            self.val_fragment_metrics.update(source_logits, source_labels)
            return {
                "logits": source_logits,
                "y": source_labels,
                "loss": task_loss,
                "patch_pos": source_patch_pos,
            }

    def predict_step(
        self, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prediction step for the model.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.
            dataloader_idx (int, optional): Index of the data loader. Defaults to 0.

        Returns:
            tuple: Probabilities and patch positions.
        """
        if len(batch) == 2:
            x, patch_pos = batch
        else:
            raise ValueError("Expected number of items in a batch to be 2.")

        logits, _ = self(x)
        y_proba = F.sigmoid(logits)
        return y_proba, patch_pos

    def on_validation_epoch_end(self) -> None:
        """Method called at the end of a validation epoch."""
        val_metrics_output = self.val_fragment_metrics.compute()
        self.log_dict(val_metrics_output)
        self.val_fragment_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


def calculate_dann_alpha(gamma: float, p: float) -> float:
    """Compute the domain adaptation parameter alpha (lambda) for Domain-Adversarial Neural Networks (DANN).

    This function calculates alpha using a scaled and shifted version of the logistic (sigmoid) function.

    Args:
        gamma (float): Scaling factor that modulates the sigmoid function.
        p (float): Progress variable, typically representing the training progress.

    Returns:
        float: The computed value of alpha, in the range of [-1, 1].

    Example:
        >>> calculate_dann_alpha(10, 0.1)
        0.9757230564143254
    """
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1
