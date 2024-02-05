import argparse
from pathlib import Path


def get_segment_ids(directory: str, with_subsegments: bool = False):
    # Create a Path object for the directory
    dir_path = Path(directory)

    # Initialize a set for unique segment IDs
    segment_ids = set()

    # Patterns to match specific file names
    patterns = ["*_inklabels.png", "*_papyrusnoninklabels.png"]

    # Search for files matching each pattern
    for pattern in patterns:
        for file_path in dir_path.glob(pattern):
            # Extract the segment ID from the file stem
            stem = file_path.stem
            if not with_subsegments:
                segment_id = (
                    stem.split("_C")[0].split("_inklabels")[0].split("_papyrusnoninklabels")[0]
                )
            else:
                segment_id = stem.split("_inklabels")[0].split("_papyrusnoninklabels")[0]
            segment_ids.add(segment_id)

    return segment_ids


def main():
    parser = argparse.ArgumentParser(description="Extract segment IDs from specific PNG filenames.")
    parser.add_argument("directory", type=str, help="Directory containing PNG files")
    parser.add_argument("--with_subsegments", help="Include sub-segments", action="store_true")

    args = parser.parse_args()
    directory = args.directory

    segment_ids = get_segment_ids(directory, args.with_subsegments)
    print(" ".join(segment_ids))


if __name__ == "__main__":
    main()
