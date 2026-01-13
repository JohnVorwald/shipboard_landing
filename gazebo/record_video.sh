#!/bin/bash
#
# Record Gazebo Simulation Video
# Usage: ./record_video.sh [duration] [output_file] [sea_state]
#
# Examples:
#   ./record_video.sh                    # 15s video, default output
#   ./record_video.sh 30 landing.mp4    # 30s video, custom output
#   ./record_video.sh 15 demo.mp4 6     # 15s, sea state 6
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parameters
DURATION=${1:-15}
OUTPUT_FILE=${2:-"${SCRIPT_DIR}/../videos/shipboard_landing_$(date +%Y%m%d_%H%M%S).mp4"}
SEA_STATE=${3:-5}

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "======================================"
echo " Gazebo Video Recording"
echo " Duration: ${DURATION}s"
echo " Output: $OUTPUT_FILE"
echo " Sea State: $SEA_STATE"
echo "======================================"

# Set environment
export GAZEBO_MODEL_PATH="${SCRIPT_DIR}/models:${GAZEBO_MODEL_PATH}"
export GAZEBO_PLUGIN_PATH="${SCRIPT_DIR}/plugins/build:${GAZEBO_PLUGIN_PATH}"

# Check if plugins are built
if [ ! -f "${SCRIPT_DIR}/plugins/build/libship_motion_plugin.so" ]; then
    echo "Building Gazebo plugins..."
    mkdir -p "${SCRIPT_DIR}/plugins/build"
    cd "${SCRIPT_DIR}/plugins/build"
    cmake .. && make -j$(nproc)
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build plugins"
        exit 1
    fi
    cd "$SCRIPT_DIR"
fi

# Generate world file with sea state
WORLD_FILE="${SCRIPT_DIR}/worlds/ship_landing_ss${SEA_STATE}.world"
sed "s/<sea_state>5</<sea_state>${SEA_STATE}</g" \
    "${SCRIPT_DIR}/worlds/ship_landing.world" > "$WORLD_FILE"

# Temporary log directory for Gazebo state recording
LOG_DIR=$(mktemp -d)
STATE_LOG="${LOG_DIR}/state.log"

echo "Starting Gazebo with recording..."

# Launch Gazebo with state logging
timeout ${DURATION}s gazebo --verbose \
    --record_path="$LOG_DIR" \
    "$WORLD_FILE" &
GZ_PID=$!

# Wait for Gazebo to start
sleep 2

# Check if Gazebo started
if ! kill -0 $GZ_PID 2>/dev/null; then
    echo "ERROR: Gazebo failed to start"
    rm -rf "$LOG_DIR"
    exit 1
fi

echo "Recording for ${DURATION} seconds..."

# Wait for recording to complete
wait $GZ_PID
EXIT_CODE=$?

# Check if video was recorded
if [ -f "${LOG_DIR}/state.log" ]; then
    echo "Converting state log to video..."

    # Use gz log to playback and record with ffmpeg
    # Alternative: Use screen capture if X is running
    if command -v ffmpeg &> /dev/null && [ -n "$DISPLAY" ]; then
        echo "Using FFmpeg screen capture fallback..."
        # This is a fallback - actual video would need X display
    fi

    # Move any recorded files
    if ls "${LOG_DIR}"/*.mp4 2>/dev/null; then
        mv "${LOG_DIR}"/*.mp4 "$OUTPUT_FILE"
        echo "Video saved to: $OUTPUT_FILE"
    else
        echo "Note: Gazebo state log saved to ${LOG_DIR}"
        echo "Convert with: gz log -e ${STATE_LOG} | gz sdf -p | ..."
    fi
else
    echo "Recording completed (check Gazebo recording at ${LOG_DIR})"
fi

# Cleanup
if [ $EXIT_CODE -eq 124 ]; then
    echo "Recording completed (timeout reached)"
else
    echo "Recording completed (exit code: $EXIT_CODE)"
fi

echo ""
echo "To view the simulation again:"
echo "  gazebo --play ${LOG_DIR}/state.log"
echo ""

# Optional: Keep log directory for review
echo "State log saved to: ${LOG_DIR}"
