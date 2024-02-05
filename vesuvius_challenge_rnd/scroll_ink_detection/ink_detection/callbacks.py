from typing import Any, Literal, Optional

from collections.abc import Sequence
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter


class PredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "epoch",
    ):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Sequence[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        batch_pred_dir = self.output_dir / str(dataloader_idx)
        batch_pred_dir.mkdir(exist_ok=True, parents=True)
        out_path = batch_pred_dir / f"{batch_idx}.pt"
        torch.save(prediction, out_path)

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Sequence[Any] | None,
    ) -> None:
        self.output_dir.mkdir(exist_ok=True, parents=True)
        out_path = self.output_dir / "predictions.pt"
        torch.save(predictions, out_path)
