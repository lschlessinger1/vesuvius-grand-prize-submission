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
    EXTRAS_PATH=$REMOTE_PATH/extras
    SEGMENT_DIR="$OUTPUT_DIR_ABSOLUTE/${SCROLL_PART_NAME}"

    echo "Downloading monster segment mesh data to $SEGMENT_DIR..."

    # 1. Download obj, tif, and mtl.
    rclone copy :http:/"$EXTRAS_PATH" "$SEGMENT_DIR" --http-url http://$USER:$PASS@$HOST/ \
    --filter "+ $SCROLL_PART_NAME.tif" \
    --filter "+ $SCROLL_PART_NAME.obj" \
    --filter "+ $SCROLL_PART_NAME.mtl" \
    --filter "- *" \
    --transfers=4 --progress --multi-thread-streams=4 --size-only

    # 2. Download ppm.
    PPM_PATH="$EXTRAS_PATH/${SCROLL_PART_NAME}.ppm"
    rclone copy :http:/"$PPM_PATH" "$SEGMENT_DIR" --http-url http://$USER:$PASS@$HOST/ --progress --multi-thread-streams=4 --transfers=4 --size-only
done
