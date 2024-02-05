#!/usr/bin/env bash

# Check command-line arguments
if [ $# -eq 0 ]; then
  echo "Please provide at least one scroll ID to download."
  exit
fi

HOST=dl.ash2txt.org
SCRIPT_DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 || exit ; pwd -P )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

SCROLL_DIRNAME=scrolls
DEFAULT_OUTPUT_DATA_DIR="$WORKSPACE_DIR/data"
OUTPUT_DATA_DIR="${DATA_DIR:-$DEFAULT_OUTPUT_DATA_DIR}"
OUTPUT_DIR_ABSOLUTE="$OUTPUT_DATA_DIR/$SCROLL_DIRNAME"

for i in "$@"
do
  SCROLL_DIR="$OUTPUT_DIR_ABSOLUTE/$i"
  REMOTE_PATH=full-scrolls/Scroll$i.volpkg/paths
  echo "Downloading scroll $i mesh files to $SCROLL_DIR..."

  mkdir -p "$SCROLL_DIR"  # Ensure that the destination directory exists

  rclone copy :http:/"$REMOTE_PATH" "$SCROLL_DIR" --http-url http://$USER:$PASS@$HOST/ \
  --filter "+ /[0-9]*/{{\d+}}.tif" \
  --filter "+ /[0-9]*/{{\d+}}.obj" \
  --filter "+ /[0-9]*/{{\d+}}.mtl" \
  --filter "+ /[0-9]*/{{\d+}}.ppm" \
  --filter "- *" \
  --transfers=4 --progress --multi-thread-streams=4 --size-only
done
