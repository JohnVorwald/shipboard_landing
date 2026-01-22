#!/bin/bash
# Record ship landing mission to video
# Usage: ./record_mission.sh [output_file.mp4]

OUTPUT_FILE="${1:-ship_landing_$(date +%Y%m%d_%H%M%S).mp4}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# PIDs for cleanup
GZ_PID=""
FFMPEG_PID=""

cleanup() {
    echo ""
    echo "Cleaning up..."

    # Stop ffmpeg gracefully (send quit command)
    if [ -n "$FFMPEG_PID" ] && kill -0 $FFMPEG_PID 2>/dev/null; then
        echo "Stopping recording..."
        kill -INT $FFMPEG_PID 2>/dev/null
        sleep 2
    fi

    # Stop Gazebo
    if [ -n "$GZ_PID" ] && kill -0 $GZ_PID 2>/dev/null; then
        echo "Stopping Gazebo..."
        kill $GZ_PID 2>/dev/null
    fi

    pkill -9 -f "gz sim" 2>/dev/null
    pkill -9 -f "arducopter" 2>/dev/null
    pkill -9 -f "sim_vehicle" 2>/dev/null

    if [ -f "$OUTPUT_FILE" ]; then
        SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
        echo "Video saved: $OUTPUT_FILE ($SIZE)"
    fi

    exit 0
}

# Trap Ctrl+C and other signals
trap cleanup SIGINT SIGTERM EXIT

echo "=============================================="
echo "  Ship Landing Mission Recording"
echo "=============================================="
echo "Output file: $OUTPUT_FILE"
echo "Press Ctrl+C to stop and save video"
echo ""

# Clean up any existing processes
pkill -9 -f "gz sim" 2>/dev/null
pkill -9 -f "arducopter" 2>/dev/null
pkill -9 -f "sim_vehicle" 2>/dev/null
sleep 2

# Auto-detect display if not set
if [ -z "$DISPLAY" ]; then
    DETECTED=$(xdpyinfo 2>/dev/null | grep "name of display" | awk '{print $4}')
    if [ -n "$DETECTED" ]; then
        export DISPLAY="$DETECTED"
    else
        export DISPLAY=:0
    fi
fi
echo "Using DISPLAY=$DISPLAY"

# Get display info
DISPLAY_RES=$(xdpyinfo 2>/dev/null | grep dimensions | awk '{print $2}')
if [ -z "$DISPLAY_RES" ]; then
    DISPLAY_RES="1920x1080"
fi
echo "Display resolution: $DISPLAY_RES"

# Start Gazebo with GUI
echo ""
echo "Starting Gazebo with GUI..."

# Set up Gazebo environment (from /tmp/run_gz.sh - working config)
export GZ_IP=127.0.0.1
export GZ_PARTITION=gazebo_default
export GZ_SIM_PHYSICS_ENGINE_PATH="/usr/lib/x86_64-linux-gnu/gz-physics-8/engine-plugins"
export GZ_SIM_RENDER_ENGINE_PATH="/usr/lib/x86_64-linux-gnu/gz-rendering-9/engine-plugins"

# Graphics fixes for NVIDIA hybrid (CRITICAL)
export LIBGL_DRI3_DISABLE=1
export QT_QPA_PLATFORM=xcb

# Use ogre render engine (not ogre2), -r to auto-start physics
cd "$SCRIPT_DIR/../gazebo"
gz sim -r --render-engine ogre worlds/ship_simple.world &
GZ_PID=$!
echo "Gazebo PID: $GZ_PID"

# Wait for Gazebo to fully start
echo "Waiting 15 seconds for Gazebo to initialize..."
sleep 15

# Check if Gazebo is still running
if ! kill -0 $GZ_PID 2>/dev/null; then
    echo "ERROR: Gazebo failed to start"
    exit 1
fi

# Start screen recording with ffmpeg
echo ""
echo "Starting screen recording..."
ffmpeg -y -f x11grab -framerate 30 -video_size $DISPLAY_RES -i $DISPLAY \
    -c:v libx264 -preset ultrafast -crf 23 \
    "$OUTPUT_FILE" 2>/dev/null &
FFMPEG_PID=$!
echo "FFmpeg PID: $FFMPEG_PID"
sleep 2

# Run the mission (skip gazebo since it's already running)
echo ""
echo "Running ship landing mission..."
echo "Ship and quad positions will be printed during flight."
echo ""
cd "$SCRIPT_DIR"
python3 ship_landing_mission.py --skip-gazebo --ship-speed 0 --gui 2>&1

# Wait a moment for final frames
sleep 3

# Cleanup will be called by trap
