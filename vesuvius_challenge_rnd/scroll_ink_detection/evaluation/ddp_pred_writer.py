import os

import torch
from pytorch_lightning.callbacks import BasePredictionWriter


class PredictionWriterDDP(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of its respective rank
        torch.save(
            predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt")
        )

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(
            batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt")
        )