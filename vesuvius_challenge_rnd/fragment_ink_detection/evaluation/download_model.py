import os
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv

from vesuvius_challenge_rnd.util import download_wandb_artifact, get_wandb_artifact, get_wandb_run


def download_model_and_config_from_wandb(
    run_id: str, entity: str, project: str, alias: str = "best", output_dir: str | None = None
) -> tuple[Path, Path]:
    if output_dir is None:
        output_dir = "."

    # Download model.
    model_artifact_name = f"model-{run_id}:{alias}"
    ckpt_artifact = get_wandb_artifact(entity, project, model_artifact_name, artifact_type="model")
    ckpt_artifact.download()
    ckpt_dir_path = download_wandb_artifact(ckpt_artifact, output_dir=output_dir)
    if output_dir is None:
        ckpt_path = ckpt_dir_path / "artifacts" / model_artifact_name / "model.ckpt"
    else:
        ckpt_path = ckpt_dir_path / "model.ckpt"
    print(f"Downloaded model checkpoint to {ckpt_path}.")

    # Download config.
    run = get_wandb_run(entity, project, run_id)
    if run is None:
        raise ValueError(f"Run {run_id} not found for {entity}/{project}.")

    config_filename = "config_pl.yaml"
    config_path = Path(output_dir) / config_filename
    config_file = run.file(config_filename)
    config_file.download(root=output_dir, exist_ok=True, replace=True)
    print(f"Downloaded model config to {config_path.resolve()}")

    return ckpt_path, config_path


def _set_up_parser() -> ArgumentParser:
    """Set up argument parser for command-line interface.

    Returns:
        ArgumentParser: Configured argument parser.
    """
    parser = ArgumentParser(description="Download a model fom Weights and Biases (W&B).")
    parser.add_argument("run_id", type=str, help="Run ID")
    parser.add_argument("-a", "--alias", type=str, help="Model alias", default="best")
    parser.add_argument(
        "-e", "--entity", type=str, help="W&B entity", default=os.getenv("WANDB_ENTITY")
    )
    parser.add_argument(
        "-p", "--project", type=str, help="W&B project", default=os.getenv("WANDB_PROJECT")
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, help="Model artifact output directory", default="downloads"
    )
    return parser


def main() -> None:
    load_dotenv()
    parser = _set_up_parser()
    args = parser.parse_args()
    download_model_and_config_from_wandb(
        args.run_id, args.entity, args.project, alias=args.alias, output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
