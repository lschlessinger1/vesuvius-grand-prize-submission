#!/usr/bin/env bash

# Check command-line arguments
if [ $# -eq 0 ]; then
  echo "Please provide at least one fragment ID (1, 2, or 3) to download."
  exit
fi

HOST=dl.ash2txt.org
SCRIPT_DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 || exit ; pwd -P )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

FRAGMENT_DIRNAME=fragments
DEFAULT_OUTPUT_DATA_DIR="$WORKSPACE_DIR/data"
OUTPUT_DATA_DIR="${DATA_DIR:-$DEFAULT_OUTPUT_DATA_DIR}"
OUTPUT_DIR_ABSOLUTE="$OUTPUT_DATA_DIR/$FRAGMENT_DIRNAME"

for i in "$@"
do
  FRAGMENT_DIR="$OUTPUT_DIR_ABSOLUTE/$i"
  REMOTE_PATH=fragments/Frag$i.volpkg/working/54keV_exposed_surface

  # Download ppm, obj, tif, and mtl files.
  rclone copy :http:/"$REMOTE_PATH" "$FRAGMENT_DIR" --http-url http://$USER:$PASS@$HOST/ \
    --include "{result.ppm,result.obj,result.tif,result.mtl}" --transfers=4 --progress --multi-thread-streams=4 --size-only
done
