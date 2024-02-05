#!/usr/bin/env bash

# Check command-line arguments
if [ $# -eq 0 ]; then
  echo "Usage: $0 <scroll_id> <segment_id>..."
  echo "Please provide at least one scroll ID (1, 2, PHerc1667, or PHerc0332) to download."
  exit
fi

# Access the scroll ID
scroll_id=$1
# Check if the scroll ID is an integer
if [[ $scroll_id =~ ^-?[0-9]+$ ]]; then
    SCROLL_NAME="Scroll$scroll_id"
else
    SCROLL_NAME="$scroll_id"
fi
echo "Scroll name: $SCROLL_NAME"

# Shift the arguments so that we can iterate over segment IDs
shift

# Check if at least one segment ID is provided
if [ $# -lt 1 ]; then
    echo "Error: No segment IDs provided."
    echo "Please provide at least one segment ID after the scroll ID."
    exit 1
fi

HOST=dl.ash2txt.org
SCRIPT_DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 || exit ; pwd -P )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

SCROLL_DIRNAME=scrolls
DEFAULT_OUTPUT_DATA_DIR="$WORKSPACE_DIR/data"
OUTPUT_DATA_DIR="${DATA_DIR:-$DEFAULT_OUTPUT_DATA_DIR}"
OUTPUT_DIR_ABSOLUTE="$OUTPUT_DATA_DIR/$SCROLL_DIRNAME"

SCROLL_DIR="$OUTPUT_DIR_ABSOLUTE/$scroll_id"
for segment_id in "$@"; do
    echo "Downloading segment ID: $segment_id"
    REMOTE_PATH=full-scrolls/$SCROLL_NAME.volpkg/paths/$segment_id

    # 1. Download surface volumes
    SURFACE_VOL_PATH=$REMOTE_PATH/layers
    SEGMENT_DIR="$SCROLL_DIR/${segment_id}"

    echo "Downloading surface volumes to $SEGMENT_DIR..."
    rclone copy :http:/"$SURFACE_VOL_PATH" "$SEGMENT_DIR"/layers --http-url http://$USER:$PASS@$HOST/ \
    --progress --multi-thread-streams=4 --transfers=4 --size-only

    # 2. Download mask.
    # Check if the segment_id ends with '_superseded' and modify accordingly
    if [[ $segment_id == *_superseded ]]; then
        mask_name_prefix="${segment_id%_superseded}"
    else
        mask_name_prefix="$segment_id"
    fi

    MASK_PATH="$REMOTE_PATH/${mask_name_prefix}_mask.png"
    rclone copy :http:/"$MASK_PATH" "$SEGMENT_DIR" --http-url http://$USER:$PASS@$HOST/ --progress

    # 3. Download meta.json, area, and author.
    rclone copy :http:/"$REMOTE_PATH" "$SEGMENT_DIR" --http-url http://$USER:$PASS@$HOST/ --progress \
    --multi-thread-streams=4 --transfers=4 --size-only --include "*.{json,txt}"
done
