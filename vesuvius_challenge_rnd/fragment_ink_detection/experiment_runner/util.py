from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


class TrainerWandb(Trainer):
    """Customized trainer for W&B logger that fixes artifacts from experiment dir
    to root dir."""

    @property
    def log_dir(self) -> str | None:
        """The directory for the current experiment. Use this to save images to, etc...

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                img = ...
                save_img(img, self.trainer.log_dir)
        """
        logger = self.logger
        if logger is not None:
            if isinstance(logger, WandbLogger):
                dirpath = logger.experiment.dir
            elif not isinstance(logger, TensorBoardLogger):
                dirpath = logger.save_dir
            else:
                dirpath = logger.log_dir
        else:
            dirpath = self.default_root_dir

        dirpath = self.strategy.broadcast(dirpath)
        return dirpath
