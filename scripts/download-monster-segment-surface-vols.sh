HOST=dl.ash2txt.org

SCRIPT_DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 || exit ; pwd -P )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

SCROLL_1_DIRNAME=scrolls/1
DEFAULT_OUTPUT_DATA_DIR="$WORKSPACE_DIR/data"
OUTPUT_DATA_DIR="${DATA_DIR:-$DEFAULT_OUTPUT_DATA_DIR}"
OUTPUT_DIR_ABSOLUTE="$OUTPUT_DATA_DIR/$SCROLL_1_DIRNAME"

if [ "$#" -eq 0 ]; then
    echo "Please provide at least one orientation (recto or verso) as an argument."
    exit 1
fi

for ORIENTATION in "$@"; do
    if [[ "$ORIENTATION" != "recto" && "$ORIENTATION" != "verso" ]]; then
        echo "Invalid orientation: $ORIENTATION. Accepted values are recto or verso."
        exit 1
    fi
    REMOTE_PATH=stephen-parsons-uploads/$ORIENTATION

    # 1. Download surface volumes and meta.json.
    SCROLL_SUBPART_NAME="Scroll1_part_1_wrap"
    SCROLL_PART_NAME=${SCROLL_SUBPART_NAME}_${ORIENTATION}
    SURFACE_VOL_PATH=$REMOTE_PATH/${SCROLL_PART_NAME}_surface_volume
    SEGMENT_DIR="$OUTPUT_DIR_ABSOLUTE/${SCROLL_PART_NAME}"

    echo "Downloading monster segment surface volumes to $SEGMENT_DIR..."

    rclone copy :http:/"$SURFACE_VOL_PATH" "$SEGMENT_DIR"/surface_volume --http-url http://$USER:$PASS@$HOST/ \
    --progress --multi-thread-streams=4 --transfers=4 --size-only

    # 2. Download mask.
    MASK_PATH="$REMOTE_PATH/${SCROLL_PART_NAME}_mask.png"
    rclone copy :http:/"$MASK_PATH" "$SEGMENT_DIR" --http-url http://$USER:$PASS@$HOST/ --progress
done
