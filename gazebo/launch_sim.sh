#!/bin/bash
#
# Shipboard Landing Gazebo Simulation Launcher
# Usage: ./launch_sim.sh [sea_state] [--headless]
#
# Examples:
#   ./launch_sim.sh           # Sea state 5, GUI mode
#   ./launch_sim.sh 3         # Sea state 3, GUI mode
#   ./launch_sim.sh 6 --headless  # Sea state 6, headless mode
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default parameters
SEA_STATE=${1:-5}
HEADLESS=false

# Check for headless flag
for arg in "$@"; do
    if [ "$arg" == "--headless" ]; then
        HEADLESS=true
    fi
done

echo "======================================"
echo " Shipboard Landing Simulation"
echo " Sea State: $SEA_STATE"
echo " Headless: $HEADLESS"
echo "======================================"

# Set Gazebo model path
export GAZEBO_MODEL_PATH="${SCRIPT_DIR}/models:${GAZEBO_MODEL_PATH}"

# Set plugin path (after building)
export GAZEBO_PLUGIN_PATH="${SCRIPT_DIR}/plugins/build:${GAZEBO_PLUGIN_PATH}"

# Check if plugins are built
if [ ! -f "${SCRIPT_DIR}/plugins/build/libship_motion_plugin.so" ]; then
    echo ""
    echo "WARNING: Plugins not built. Building now..."
    echo ""

    mkdir -p "${SCRIPT_DIR}/plugins/build"
    cd "${SCRIPT_DIR}/plugins/build"
    cmake ..
    make -j$(nproc)

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build plugins"
        exit 1
    fi

    cd "$SCRIPT_DIR"
    echo "Plugins built successfully."
fi

# Generate world file with specified sea state
WORLD_FILE="${SCRIPT_DIR}/worlds/ship_landing_ss${SEA_STATE}.world"

# Modify sea state in world file
sed "s/<sea_state>5</<sea_state>${SEA_STATE}</g" \
    "${SCRIPT_DIR}/worlds/ship_landing.world" > "$WORLD_FILE"

echo "Generated world file: $WORLD_FILE"

# Launch Gazebo
if [ "$HEADLESS" = true ]; then
    echo "Starting Gazebo in headless mode..."
    gzserver --verbose "$WORLD_FILE" &
    GZ_PID=$!
else
    echo "Starting Gazebo with GUI..."
    gazebo --verbose "$WORLD_FILE" &
    GZ_PID=$!
fi

echo "Gazebo PID: $GZ_PID"

# Wait a moment for Gazebo to start
sleep 3

# Function to send landing command
land_uav() {
    local duration=${1:-10}
    echo "Sending landing command (duration: ${duration}s)..."
    gz topic -p /gazebo/default/quadrotor/land -m 'data: '$duration
}

# Function to set hover target
set_hover() {
    local x=${1:-0}
    local y=${2:-0}
    local z=${3:-15}
    echo "Setting hover target: ($x, $y, $z)"
    gz topic -p /gazebo/default/quadrotor/trajectory -m "x: $x, y: $y, z: $z"
}

# Export functions for interactive use
export -f land_uav
export -f set_hover

echo ""
echo "Commands available:"
echo "  land_uav [duration]     - Initiate landing (default 10s)"
echo "  set_hover x y z         - Set hover position"
echo ""
echo "Press Ctrl+C to stop simulation"

# Wait for Gazebo to exit
wait $GZ_PID
