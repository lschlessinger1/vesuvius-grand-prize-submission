from typing import Literal

import logging

import numpy as np
import segmentation_models_pytorch as smp
import torch
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

from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.models.cnn3d_segformer import (
    create_cnn3d_segformer_model,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.schedulers import (
    GradualWarmupSchedulerV2,
)


class CNN3DSegformerPLModel(LightningModule):
    def __init__(
        self,
        pred_shape: tuple[int, int],
        size: int = 64,
        smooth_factor: float = 0.25,
        dice_weight: float = 0.5,
        lr: float = 2e-5,
        bce_pos_weight: torch.Tensor | None = None,
        metric_thresh: float = 0.5,
        metric_gt_ink_thresh: float = 0.05,
        model_size: Literal["b3", "b4", "b5"] = "b3",
    ):
        super().__init__()
        self.pred_shape = pred_shape
        self.size = size
        self.lr = lr
        self.save_hyperparameters()

        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode="binary")
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(
            smooth_factor=smooth_factor, pos_weight=bce_pos_weight
        )
        bce_weight = 1 - dice_weight
        self.loss_func = lambda x, y: dice_weight * self.loss_func1(
            x, y
        ) + bce_weight * self.loss_func2(x, y)

        self.model = create_cnn3d_segformer_model(model_size)

        self.metric_gt_ink_thresh = metric_gt_ink_thresh
        shared_metrics = MetricCollection(
            [
                BinaryAccuracy(threshold=metric_thresh),
                BinaryFBetaScore(beta=0.5, threshold=metric_thresh),
            ]
        )
        self.train_metrics = shared_metrics.clone(prefix="train/")
        self.val_metrics = shared_metrics.clone(prefix="val/")
        self.val_metrics.add_metrics([BinaryAUROC(), BinaryAveragePrecision()])

        self.example_input_array = torch.ones(4, 1, 32, self.size, self.size)
        self.validation_step_outputs = []

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]
        pred_mask = self.model(x)
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
                    y_preds[i].unsqueeze(0).float().squeeze(0).squeeze(0).numpy()
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
