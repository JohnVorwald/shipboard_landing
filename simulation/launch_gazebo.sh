#!/bin/bash
# Wrapper script to launch Gazebo with correct environment variables
# Called by ship_landing_demo.py

echo "=== LAUNCHING GAZEBO WITH ARDUPILOT PLUGIN ==="

# Set required environment variables
export GZ_SIM_RESOURCE_PATH="/home/john/gz_ws/src/ardupilot_gazebo/models:/home/john/gz_ws/src/ardupilot_gazebo/worlds:/home/john/github/shipboard_landing/gazebo/models"
export GZ_SIM_SYSTEM_PLUGIN_PATH="/home/john/gz_ws/src/ardupilot_gazebo/build"
export GZ_IP="127.0.0.1"
export GZ_PARTITION="gazebo_default"

# Display fixes for remote desktop
export LIBGL_DRI3_DISABLE=1
export QT_QPA_PLATFORM=xcb

# Use display from argument or detect
if [ -n "$1" ]; then
    export DISPLAY="$1"
fi

echo "DISPLAY: $DISPLAY"
echo "GZ_SIM_SYSTEM_PLUGIN_PATH: $GZ_SIM_SYSTEM_PLUGIN_PATH"
echo "GZ_SIM_RESOURCE_PATH: $GZ_SIM_RESOURCE_PATH"

# World file (can be overridden by second argument)
WORLD_FILE="${2:-/home/john/github/shipboard_landing/gazebo/worlds/ship_landing_ardupilot.sdf}"

# GUI config (optional third argument)
GUI_CONFIG="$3"

echo "World file: $WORLD_FILE"
echo ""

# Build command
CMD="gz sim -r"
if [ -n "$GUI_CONFIG" ]; then
    CMD="$CMD --gui-config $GUI_CONFIG"
fi
CMD="$CMD $WORLD_FILE"

echo "Running: $CMD"
echo "=== Gazebo starting... ==="
echo ""

# Execute Gazebo (exec replaces this shell process)
exec $CMD
