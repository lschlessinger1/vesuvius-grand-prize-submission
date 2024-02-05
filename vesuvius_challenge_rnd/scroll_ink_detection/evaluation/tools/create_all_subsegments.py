import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from vesuvius_challenge_rnd.scroll_ink_detection.evaluation.tools import create_subsegment

DEFAULT_SUBSEGMENT_INFO_PATH = Path(__file__).parent / "subsegment_info.csv"


def create_all_subsegments(
    subsegment_info_path: Path = DEFAULT_SUBSEGMENT_INFO_PATH,
    skip_if_exists: bool = True,
    use_memmap: bool = True,
) -> None:
    print(f"Creating subsegments from {subsegment_info_path}")
    subsegment_info_df = pd.read_csv(subsegment_info_path).sort_values(
        by=["scroll_id", "segment_id", "column_id"]
    )

    n_rows = len(subsegment_info_df)
    print(f"Found {n_rows} sub-segments in file.")
    for _, row in tqdm(subsegment_info_df.iterrows(), total=n_rows):
        try:
            create_subsegment(
                str(row.segment_id),
                str(row.scroll_id),
                int(row.u1),
                int(row.u2),
                int(row.v1),
                int(row.v2),
                str(row.column_id),
                skip_if_exists=skip_if_exists,
                use_memmap=use_memmap,
            )
        except Exception as e:
            print(f"Failed to create sub-segment {row}: {e}")


def main():
    load_dotenv()
    parser = _set_up_parser()
    args = parser.parse_args()
    create_all_subsegments(
        args.subsegment_info_path,
        skip_if_exists=args.skip_if_exists,
        use_memmap=not args.load_in_memory,
    )


def _set_up_parser() -> argparse.ArgumentParser:
    """Set up argument parser for command-line interface.

    Returns:
        ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate subsegment for scroll segments from a CSV file."
    )
    parser.add_argument(
        "--subsegment_info_path",
        type=str,
        default=DEFAULT_SUBSEGMENT_INFO_PATH,
        help="The sub-segment info CSV file path.",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip the creation of a sub-segment if it already exists.",
    )
    parser.add_argument(
        "--load_in_memory",
        action="store_true",
        help="Load the image stack in RAM instead of as a memory-mapped file.",
    )
    return parser


if __name__ == "__main__":
    main()
