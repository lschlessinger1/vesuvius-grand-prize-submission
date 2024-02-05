import logging

import segmentation_models_pytorch.losses  # noqa: F401
from dotenv import load_dotenv
from pytorch_lightning.cli import ArgsType, LightningCLI
from pytorch_lightning.loggers import WandbLogger

import vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.schedulers  # noqa: F401
from vesuvius_challenge_rnd.fragment_ink_detection.experiment_runner.util import TrainerWandb
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.util import compile_if_possible

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
)


class FragmentLightningCLI(LightningCLI):
    def before_fit(self):
        """Method to be run before the fitting process."""
        load_dotenv()  # take environment variables from .env.

        if isinstance(self.trainer.logger, WandbLogger):
            # log gradients and model topology
            self.trainer.logger.watch(self.model, log_graph=False)

        self.model = compile_if_possible(self.model)

    def after_fit(self):
        """Method to be run after the fitting process."""
        checkpoint_callback = self.trainer.checkpoint_callback
        if checkpoint_callback is not None:
            logging.info(f"Best model saved to: {checkpoint_callback.best_model_path}")


def cli_main(args: ArgsType | None = None) -> FragmentLightningCLI:
    """Main CLI entry point.

    Args:
        args (ArgsType, optional): Command-line arguments.

    Returns:
        FragmentLightningCLI: An instance of the FragmentLightningCLI class.
    """
    return FragmentLightningCLI(
        trainer_class=TrainerWandb,
        trainer_defaults={"max_epochs": 100, "precision": "16-mixed", "benchmark": True},
        save_config_kwargs={
            "config_filename": "config_pl.yaml",
            "overwrite": True,
        },
        args=args,
    )


if __name__ == "__main__":
    cli_main()
