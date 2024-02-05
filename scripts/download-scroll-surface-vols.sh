#!/usr/bin/env bash

# Check command-line arguments
if [ $# -eq 0 ]; then
  echo "Please provide at least one scroll ID (1, 2, PHerc1667, or PHerc0332) to download."
  exit
fi

HOST=dl.ash2txt.org
SCRIPT_DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 || exit ; pwd -P )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

SCROLL_DIRNAME=scrolls
DEFAULT_OUTPUT_DATA_DIR="$WORKSPACE_DIR/data"
OUTPUT_DATA_DIR="${DATA_DIR:-$DEFAULT_OUTPUT_DATA_DIR}"
OUTPUT_DIR_ABSOLUTE="$OUTPUT_DATA_DIR/$SCROLL_DIRNAME"

excluded_extensions=("obj" "cpp" "ppm" "vcps" "orig" "mtl")  # List of file extensions to exclude

# Join the elements of the array with commas
extension_string=$(IFS=,; echo "${excluded_extensions[*]}")

for i in "$@"
do
  SCROLL_DIR="$OUTPUT_DIR_ABSOLUTE/$i"
  # Check if $i is an integer
  if [[ $i =~ ^-?[0-9]+$ ]]; then
      SCROLL_NAME="Scroll$i"
  else
      SCROLL_NAME="$i"
  fi
  REMOTE_PATH=full-scrolls/$SCROLL_NAME.volpkg/paths
  echo "Downloading scroll $i surface volumes to $SCROLL_DIR..."

  rclone copy :http:/"$REMOTE_PATH" "$SCROLL_DIR" --http-url http://$USER:$PASS@$HOST/ \
  --progress --multi-thread-streams=4 --transfers=4 --size-only --exclude "*.{$extension_string}"
done
