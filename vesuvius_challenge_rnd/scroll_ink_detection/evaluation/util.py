from argparse import ArgumentTypeError
from pathlib import Path


def tuple_parser(input_str: str) -> tuple[int, str]:
    try:
        scroll_id, segment_id = input_str.split(",")
        return int(scroll_id), segment_id
    except Exception:
        raise ValueError("Invalid tuple format. Expected 'scroll_id,segment_id'.")


def validate_file_path(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise ArgumentTypeError(f"The path {file_path} does not exist.")
    elif not path.is_file():
        raise ArgumentTypeError(f"The path {file_path} exists but is not a file.")
    return file_path


def validate_positive_int(value: str) -> int:
    try:
        int_value = int(value)
        if int_value <= 0:
            raise ValueError
    except ValueError:
        raise ArgumentTypeError(f"Invalid value: {value}. Must be a positive integer.")
    return int_value


def validate_nonnegative_int(value: str) -> int:
    try:
        int_value = int(value)
        if int_value < 0:
            raise ValueError
    except ValueError:
        raise ArgumentTypeError(f"Invalid value: {value}. Must be a positive integer.")
    return int_value
