#!/usr/bin/env bash

# Check command-line arguments
if [ $# -eq 0 ]; then
  echo "Please provide at least one fragment ID (1, 2, 3, or 4) to download."
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

  # 1. Download surface volumes.
  SURFACE_VOL_PATH=$REMOTE_PATH/surface_volume
  echo "Downloading fragment $i surface volumes to $FRAGMENT_DIR..."

  rclone copy :http:/"$SURFACE_VOL_PATH" "$FRAGMENT_DIR"/surface_volume --http-url http://$USER:$PASS@$HOST/ \
  --progress --multi-thread-streams=4 --transfers=4 --size-only --exclude "*.json"

  # 2. Download ink labels, IR image, and mask.
  other_files=("inklabels.png" "ir.png" "mask.png")
  for f in "${other_files[@]}"
  do
    rclone copyto :http:/"$REMOTE_PATH/$f" "$FRAGMENT_DIR"/"$f" --http-url http://$USER:$PASS@$HOST/
  done
done
