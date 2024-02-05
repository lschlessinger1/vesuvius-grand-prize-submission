"""
Gradual Warmup Scheduler for PyTorch's Optimizer.
Adapted from https://github.com/ildoonet/pytorch-gradual-warmup-lr
"""
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


class GradualWarmupScheduler(LRScheduler):
    """Gradually warm-up (increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: Target learning rate multiplier. target learning rate = base lr * multiplier if multiplier > 1.0.
            if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epochs: Epochs to gradually reach target learning rate.
        after_scheduler: Scheduler to use after total_epochs (e.g. ReduceLROnPlateau).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        multiplier: float,
        total_epochs: int,
        after_scheduler: LRScheduler | None = None,
    ):
        if multiplier < 1.0:
            raise ValueError("multiplier should be >= 1.")

        self.multiplier = multiplier
        self.total_epochs = total_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Calculate the learning rate based on the current epoch.

        Returns:
            List[float]: List of learning rates for each parameter group.
        """
        if self.last_epoch > self.total_epochs:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epochs) for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epochs + 1.0)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics: Any, epoch: int | None = None) -> None:
        """Special step method to handle ReduceLROnPlateau scheduling.

        Args:
            metrics (float): Metric value used to compute learning rate.
            epoch (int, optional): Current epoch number.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epochs:
            warmup_lr = [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epochs + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        elif self.after_scheduler is not None:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epochs)

    def step(self, epoch: int | None = None, metrics: Any = None) -> None:
        """Update the learning rate using the GradualWarmupScheduler.

        Args:
            epoch (int, optional): Current epoch number.
            metrics (float, optional): Metric value used to compute learning rate.
        """
        if isinstance(self.after_scheduler, ReduceLROnPlateau):
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super().step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class ReduceLROnPlateauWarmup(ReduceLROnPlateau):
    """A scheduler that combines ReduceLROnPlateau with warm-up.

    This class first warms up the learning rate linearly over a number of epochs,
    and then uses ReduceLROnPlateau to reduce the learning rate when a metric has stopped improving.

    Attributes:
        warmup_epochs (int): Number of epochs for the warm-up phase.
    """

    def __init__(self, optimizer: Optimizer, warmup_epochs: int, **kwargs) -> None:
        """Initialize the ReduceLROnPlateauWarmup scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Number of epochs for the warm-up phase.
            **kwargs: Additional keyword arguments for ReduceLROnPlateau.
        """
        super().__init__(optimizer, **kwargs)
        self.warmup_epochs = warmup_epochs
        self.last_epoch = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_rop(self.mode_worse, False)

    def warmup_lr(self, epoch: int) -> None:
        """Warm up the learning rate linearly over the warm-up phase.

        Args:
            epoch (int): Current epoch number.
        """
        factor = epoch / self.warmup_epochs
        self.last_epoch = epoch
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.base_lrs[i] * factor

    def step_rop(self, metrics: Any, evaluate: bool) -> None:
        """Step method for ReduceLROnPlateauWarmup scheduler.

        This method updates the learning rate according to the warm-up phase and evaluation condition.

        Args:
            metrics (Any): Metric value used to compute learning rate.
            evaluate (bool): Whether to evaluate the metric and update the learning rate.
        """
        epoch = self.last_epoch + 1

        if epoch <= self.warmup_epochs:
            self.warmup_lr(epoch)
        elif evaluate:
            super().step(metrics, epoch=epoch)
