from typing import Literal

import logging
import os

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryFBetaScore,
)

from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.models.depth_pooling import (
    DepthPooling,
)
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.models.squeeze_and_excitation_3d import (
    SELayer3D,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.models.i3dall import InceptionI3d
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.schedulers import (
    GradualWarmupSchedulerV2,
)


class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        encoder_dims[i] + encoder_dims[i - 1],
                        encoder_dims[i - 1],
                        3,
                        1,
                        1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(encoder_dims[i - 1]),
                    nn.ReLU(inplace=True),
                )
                for i in range(1, len(encoder_dims))
            ]
        )

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            f_down = self.convs[i - 1](f)
            feature_maps[i - 1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask


class RegressionPLModel(LightningModule):
    def __init__(
        self,
        pred_shape: tuple[int, int],
        size: int = 64,
        with_norm: bool = False,
        smooth_factor: float = 0.25,
        dice_weight: float = 0.5,
        lr: float = 2e-5,
        bce_pos_weight: float | list[float] | None = None,
        metric_thresh: float = 0.5,
        metric_gt_ink_thresh: float = 0.05,
        z_extent: int = 30,
        se_type_str: str | None = None,
        in_channels: int = 1,
        depth_pool_fn: Literal["mean", "max", "attention"] = "mean",
        ignore_idx: int = -100,
        ckpt_path: str | None = None,
    ):
        super().__init__()
        # Parse bce_pos_weight.
        if bce_pos_weight is not None:
            if isinstance(bce_pos_weight, float):
                bce_pos_weight = [bce_pos_weight]
            bce_pos_weight = torch.tensor(bce_pos_weight)

        self.pred_shape = pred_shape
        self.size = size
        self.with_norm = with_norm
        self.lr = lr
        self.ignore_idx = ignore_idx
        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape, dtype=np.float32)
        self.mask_count = np.zeros(self.hparams.pred_shape, dtype=np.int32)

        self.loss_func1 = smp.losses.DiceLoss(mode="binary", ignore_index=self.ignore_idx)
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(
            smooth_factor=smooth_factor, pos_weight=bce_pos_weight, ignore_index=self.ignore_idx
        )
        bce_weight = 1 - dice_weight
        self.loss_func = lambda x, y: dice_weight * self.loss_func1(
            x, y
        ) + bce_weight * self.loss_func2(x, y)

        self.backbone = InceptionI3d(in_channels=in_channels, num_classes=512)

        self.example_input_array = torch.ones(4, in_channels, z_extent, self.size, self.size)
        encoder_output_shapes = np.array(
            [x.shape[1:] for x in self.backbone(self.example_input_array)]
        )
        encoder_dims = encoder_output_shapes[:, 0].tolist()
        encoder_depths = encoder_output_shapes[:, 1].tolist()
        encoder_heights = encoder_output_shapes[:, 2].tolist()
        encoder_widths = encoder_output_shapes[:, 3].tolist()
        self.depth_pooler = DepthPooling(
            len(encoder_dims),
            encoder_dims,
            slice_dim=2,
            se_type=SELayer3D[se_type_str] if se_type_str is not None else None,
            pool_fn=depth_pool_fn,
            depths=encoder_depths,
            heights=encoder_heights,
            widths=encoder_widths,
        )

        self.decoder = Decoder(
            encoder_dims=encoder_dims,
            upscale=1,
        )

        if ckpt_path is not None:
            logging.info("Loading model parameters from checkpoint.")
            self.load_weights(ckpt_path)

        if self.hparams.with_norm:
            self.normalization = nn.BatchNorm3d(num_features=1)

        self.metric_gt_ink_thresh = metric_gt_ink_thresh
        shared_metrics = MetricCollection(
            [
                BinaryAccuracy(threshold=metric_thresh, ignore_index=self.ignore_idx),
                BinaryFBetaScore(beta=0.5, threshold=metric_thresh, ignore_index=self.ignore_idx),
            ]
        )
        self.train_metrics = shared_metrics.clone(prefix="train/")
        self.val_metrics = shared_metrics.clone(prefix="val/")
        self.val_metrics.add_metrics(
            [
                BinaryAUROC(ignore_index=self.ignore_idx),
                BinaryAveragePrecision(ignore_index=self.ignore_idx),
            ]
        )

        mean_fbeta_auprc = (
            self.val_metrics["BinaryFBetaScore"] + self.val_metrics["BinaryAveragePrecision"]
        ) / 2
        self.val_metrics.add_metrics({"mean_fbeta_auprc": mean_fbeta_auprc})
        self.validation_step_outputs = []

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]
        if self.hparams.with_norm:
            x = self.normalization(x)
        feat_maps = self.backbone(x)
        feat_maps_pooled = self.depth_pooler(feat_maps)
        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)
        if torch.isnan(loss):
            logging.warning("Loss nan encountered")
        self.log(
            "train/total_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        y_binarized = (y > self.metric_gt_ink_thresh).int()
        train_metrics_output = self.train_metrics(logits, y_binarized)
        self.log_dict(train_metrics_output)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y, xyxys = batch
        logits = self(x)
        loss = self.loss_func(logits, y)

        self.log(
            "val/total_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        y_binarized = (y > self.metric_gt_ink_thresh).int()
        self.val_metrics.update(logits, y_binarized)

        self.validation_step_outputs.append((logits.detach(), xyxys.detach().cpu()))

        return {
            "logits": logits,
            "y": y,
            "loss": loss,
            "patch_pos": xyxys,
        }

    def on_validation_epoch_end(self):
        outputs = self.all_gather(self.validation_step_outputs)
        logits = [t[0] for t in outputs]
        xyxys = [t[1] for t in outputs]
        self.validation_step_outputs.clear()  # free memory

        if self.trainer.world_size == 1:
            # Create a dummy "device" dimension.
            logits = [t.unsqueeze(0) for t in logits]
            xyxys = [t.unsqueeze(0) for t in xyxys]

        logits = torch.cat(logits, dim=1)  # D x N x C x patch height x patch width (C=1)
        xyxys = torch.cat(xyxys, dim=1)  # D x N x 4

        if self.trainer.is_global_zero:
            logits = logits.view(-1, *logits.shape[-3:])
            xyxys = xyxys.view(-1, xyxys.shape[-1])
            y_preds = torch.sigmoid(logits).cpu()
            for i, (x1, y1, x2, y2) in enumerate(xyxys):
                self.mask_pred[y1:y2, x1:x2] += (
                    F.interpolate(y_preds[i].unsqueeze(0).float(), scale_factor=4, mode="bilinear")
                    .squeeze(0)
                    .squeeze(0)
                    .numpy()
                )
                self.mask_count[y1:y2, x1:x2] += np.ones(
                    (self.hparams.size, self.hparams.size), dtype=self.mask_count.dtype
                )
            self.mask_pred = np.divide(
                self.mask_pred,
                self.mask_count,
                out=np.zeros_like(self.mask_pred),
                where=self.mask_count != 0,
            )
            logger = self.logger
            if isinstance(logger, WandbLogger):
                logger.log_image(
                    key="masks", images=[np.clip(self.mask_pred, 0, 1)], caption=["probs"]
                )

            # Reset mask.
            self.mask_pred = np.zeros_like(self.mask_pred)
            self.mask_count = np.zeros_like(self.mask_count)

        val_metrics_output = self.val_metrics.compute()
        self.log_dict(val_metrics_output)
        self.val_metrics.reset()

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
            raise ValueError(f"Expected number of items in a batch to be 2. Found {len(batch)}.")

        logits = self(x)
        y_proba = F.sigmoid(logits)
        return y_proba, patch_pos

    def configure_optimizers(self):
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)

        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs - 2, eta_min=1e-6
        )
        scheduler = GradualWarmupSchedulerV2(
            optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine
        )

        return [optimizer], [scheduler]

    def load_weights(self, checkpoint_path: str | os.PathLike) -> None:
        # Load the entire checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Extract the state dictionary corresponding to the entire model
        state_dict = checkpoint["state_dict"]

        # Filter out and load weights for the backbone, decoder, and depth pooler
        for component in ["backbone", "decoder", "depth_pooler"]:
            component_dict = {
                k.replace(f"{component}.", ""): v
                for k, v in state_dict.items()
                if k.startswith(f"{component}.")
            }
            getattr(self, component).load_state_dict(component_dict)
