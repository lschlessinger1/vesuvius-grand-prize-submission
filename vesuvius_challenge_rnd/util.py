from pathlib import Path

import wandb
from wandb.apis.public import Run


def get_wandb_artifact(
    entity: str,
    project: str,
    artifact_name: str,
    api: None | wandb.Api = None,
    artifact_type: str | None = None,
) -> wandb.Artifact:
    """Get a Weights and Biases Artifact."""
    if api is None:
        api = wandb.Api()
    return api.artifact(f"{entity}/{project}/{artifact_name}", type=artifact_type)


def get_wandb_run(entity: str, project: str, run_id: str, api: None | wandb.Api = None) -> Run:
    """Get a Weights and Biases Run."""
    if api is None:
        api = wandb.Api()
    return api.run(f"{entity}/{project}/{run_id}")


def download_wandb_artifact(artifact: wandb.Artifact, output_dir: str | None = None) -> Path:
    """Download a Weights and Biases Artifact to a local directory."""
    output_path = Path(artifact.download(root=output_dir)).resolve()
    return output_path
