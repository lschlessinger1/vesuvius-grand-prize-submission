import logging

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


class Encoder3dDecoder2dModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = InceptionI3d(in_channels=1, num_classes=512)

        # Ideally, refactor the Decoder to adapt to input dimensions dynamically.
        # If that's not feasible, use a dummy tensor for dimension calculation.
        dummy_input = torch.rand(1, 1, 20, 256, 256)
        encoder_dims = [x.size(1) for x in self.encoder(dummy_input)]
        self.decoder = Decoder(encoder_dims=encoder_dims, upscale=1)

    def forward(self, x):
        feat_maps = self.encoder(x)
        feat_maps_pooled = self.pool_feat_maps(feat_maps)
        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask

    @staticmethod
    def pool_feat_maps(feat_maps: list[torch.Tensor]) -> list[torch.Tensor]:
        return [torch.mean(f, dim=2) for f in feat_maps]


class I3DMeanTeacherPLModel(LightningModule):
    def __init__(
        self,
        pred_shape: tuple[int, int],
        size: int = 64,
        with_norm: bool = False,
        smooth_factor: float = 0.25,
        dice_weight: float = 0.5,
        lr: float = 2e-5,
        bce_pos_weight: torch.Tensor | None = None,
        metric_thresh: float = 0.5,
        metric_gt_ink_thresh: float = 0.05,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.pred_shape = pred_shape
        self.size = size
        self.with_norm = with_norm
        self.lr = lr
        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode="binary")
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(
            smooth_factor=smooth_factor, pos_weight=bce_pos_weight
        )
        bce_weight = 1 - dice_weight
        self.supervised_loss_fn = lambda x, y: dice_weight * self.loss_func1(
            x, y
        ) + bce_weight * self.loss_func2(x, y)

        self.model = Encoder3dDecoder2dModel()
        self.ema_model = Encoder3dDecoder2dModel()
        # Detach all parameters in the EMA model
        for param in self.ema_model.parameters():
            param.detach_()

        if self.hparams.with_norm:
            self.normalization = nn.BatchNorm3d(num_features=1)

        self.ema_decay = ema_decay

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

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]
        if self.hparams.with_norm:
            x = self.normalization(x)

        pred_mask = self.model(x)
        return pred_mask

    def training_step(self, batch, batch_idx):
        # Prepare batch.
        x_labeled, y = batch["labeled"]
        x_unlabeled = batch["unlabeled"]

        # Calculate supervised loss.
        logits_labeled = self(x_labeled)
        loss_supervised = self.supervised_loss_fn(logits_labeled, y)
        if torch.isnan(loss_supervised):
            logging.warning("Loss nan encountered")
        self.log(
            "train/sup_total_loss",
            loss_supervised.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # Calculate consistency loss.
        noise = torch.clamp(torch.randn_like(x_unlabeled) * 0.1, -0.2, 0.2)
        # This noise could also be from dropout, stochastic depth, or more data augmentation
        ema_inputs = x_unlabeled + noise

        outputs_soft = torch.softmax(logits_labeled, dim=1)
        with torch.no_grad():
            ema_output = torch.softmax(self.ema_model(ema_inputs), dim=1)

        consistency_weight = get_current_consistency_weight(self.global_step // 150)
        if self.global_step < 1000:
            consistency_loss = 0.0
        else:
            consistency_loss = torch.mean((outputs_soft - ema_output) ** 2)
        loss = loss_supervised + consistency_weight * consistency_loss

        update_ema_variables(self.model, self.ema_model, self.ema_decay, self.global_step)

        # Log other metrics.
        y_binarized = (y > self.metric_gt_ink_thresh).int()
        train_metrics_output = self.train_metrics(logits_labeled, y_binarized)
        self.log_dict(train_metrics_output)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y, xyxys = batch
        logits = self(x)
        loss = self.supervised_loss_fn(logits, y)
        y_preds = torch.sigmoid(logits).to("cpu")
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            self.mask_pred[y1:y2, x1:x2] += (
                F.interpolate(y_preds[i].unsqueeze(0).float(), scale_factor=4, mode="bilinear")
                .squeeze(0)
                .squeeze(0)
                .numpy()
            )
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/total_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        y_binarized = (y > self.metric_gt_ink_thresh).int()
        self.val_metrics.update(logits, y_binarized)

        return {
            "logits": logits,
            "y": y,
            "loss": loss,
            "patch_pos": xyxys,
        }

    def on_validation_epoch_end(self):
        self.mask_pred = np.divide(
            self.mask_pred,
            self.mask_count,
            out=np.zeros_like(self.mask_pred),
            where=self.mask_count != 0,
        )
        logger = self.logger
        if isinstance(logger, WandbLogger):
            logger.log_image(key="masks", images=[np.clip(self.mask_pred, 0, 1)], caption=["probs"])

        # reset mask
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

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


def sigmoid_rampup(current: float, rampup_length: float) -> float:
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(
    epoch: int, consistency: float = 0.1, consistency_rampup: float = 200.0
):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def update_ema_variables(
    model: nn.Module, ema_model: nn.Module, alpha: float, global_step: int
) -> None:
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
