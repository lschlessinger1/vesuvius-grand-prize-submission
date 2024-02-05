import argparse
import logging
import shutil
from pathlib import Path

from tqdm import tqdm

from vesuvius_challenge_rnd.data.constants import Z_NON_REVERSED_SEGMENT_IDS, Z_REVERSED_SEGMENT_IDS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_OUTPUT_DIR = Path("model_predictions")


def main():
    parser = _set_up_parser()
    args = parser.parse_args()
    generate_flat_model_prediction_dir(
        args.prediction_base_dir,
        args.model_id,
        args.scroll_id,
        args.start_idx,
        args.output_dir,
        args.stride,
    )


def _find_segment_pred_path(
    segment_pred_dir: Path, stride: int, start_idx: int, z_reversed: bool = False
) -> Path:
    seg_str = f"{segment_pred_dir.name}_{stride}_{start_idx}_r{int(z_reversed)}"
    pred_paths = [p for p in segment_pred_dir.glob("*.png") if p.stem.startswith(seg_str)]
    if len(pred_paths) != 1:
        raise ValueError(
            f"Found {len(pred_paths)} segment prediction paths starting with {seg_str}. Expected only one."
        )
    return pred_paths[0]


def _infer_z_reversal(segment_name: str) -> bool | None:
    if segment_name in Z_REVERSED_SEGMENT_IDS:
        return True
    elif segment_name in Z_NON_REVERSED_SEGMENT_IDS:
        return False
    else:
        return None


def generate_flat_model_prediction_dir(
    prediction_base_dir: str,
    run_id: str,
    scroll_id: str,
    start_idx: int,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    stride: int = 32,
) -> None:
    if isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    prediction_base_dir_path = Path(prediction_base_dir)
    prediction_dir_path = prediction_base_dir_path / run_id / scroll_id

    segment_pred_paths = []
    for path in prediction_dir_path.glob("*"):
        if path.is_dir():
            segment_name = path.name
            z_reverse = _infer_z_reversal(segment_name)
            if z_reverse is None:
                logger.info(
                    f"Could not determine z_reverse for {segment_name}. Defaulting to non-reversed."
                )
                z_reverse = True

            pred_path = _find_segment_pred_path(
                path, stride=stride, start_idx=start_idx, z_reversed=z_reverse
            )
            if not pred_path.is_file():
                raise FileNotFoundError(
                    f"Could not find segment {segment_name} prediction for path {pred_path}."
                )
            segment_pred_paths.append(pred_path)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)

    # Copy files.
    for segment_pred_path in tqdm(segment_pred_paths, desc="Copying segment predictions..."):
        destination_path = output_dir_path / segment_pred_path.name
        shutil.copy(segment_pred_path, destination_path)


def _set_up_parser() -> argparse.ArgumentParser:
    """Set up argument parser for command-line interface.

    Returns:
        ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate a flattened directory of predictions for a given model."
    )
    parser.add_argument(
        "prediction_base_dir", type=str, help="Segment prediction base directory path"
    )
    parser.add_argument("model_id", type=str, help="Model ID")
    parser.add_argument(
        "--scroll_id", type=str, default="1", help="The scroll ID for which the segment belongs"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output path"
    )
    parser.add_argument(
        "--start_idx",
        default=15,
        type=int,
        help="Z-plane start index.",
    )
    parser.add_argument(
        "--stride",
        default=32,
        type=int,
        help="XY-plane stride.",
    )
    return parser


if __name__ == "__main__":
    main()
