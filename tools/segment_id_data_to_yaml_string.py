import argparse


def convert_set_to_string_list(segment_ids: set, scroll_id: str = "1"):
    """
    Convert a set of strings to a string that visually represents a list of strings.

    Args:
    input_set (set): A set of strings.

    Returns:
    str: A string representing the list of strings.
    """
    return "[" + ", ".join(f"['{scroll_id}', '" + str(id_) + "']" for id_ in segment_ids) + "]"


def main():
    parser = argparse.ArgumentParser(description="Convert a set of scroll IDs.")
    parser.add_argument("segment_ids", nargs="+", type=str, help="List of segment IDs")
    parser.add_argument("--scroll_id", type=str, default="1", help="Scroll ID.")

    args = parser.parse_args()

    out_str = convert_set_to_string_list(set(args.segment_ids), scroll_id=args.scroll_id)
    print(out_str)


if __name__ == "__main__":
    main()
