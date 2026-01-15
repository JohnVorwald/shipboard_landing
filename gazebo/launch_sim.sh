#!/bin/bash
#
# Shipboard Landing Gazebo Simulation Launcher
# Usage: ./launch_sim.sh [sea_state] [--headless] [--software-render]
#
# Examples:
#   ./launch_sim.sh           # Sea state 5, GUI mode
#   ./launch_sim.sh 3         # Sea state 3, GUI mode
#   ./launch_sim.sh 6 --headless  # Sea state 6, headless mode
#   ./launch_sim.sh --software-render  # Use software rendering (for remote X)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default parameters
SEA_STATE=${1:-5}
HEADLESS=false
SOFTWARE_RENDER=false

# Check for flags
for arg in "$@"; do
    if [ "$arg" == "--headless" ]; then
        HEADLESS=true
    fi
    if [ "$arg" == "--software-render" ]; then
        SOFTWARE_RENDER=true
    fi
done

# Enable software rendering for remote X sessions
if [ "$SOFTWARE_RENDER" = true ]; then
    echo "Enabling software rendering (Mesa)..."
    export LIBGL_ALWAYS_SOFTWARE=1
    export MESA_GL_VERSION_OVERRIDE=3.3
fi

# Gazebo Ionic fixes for NVIDIA hybrid graphics
export GZ_IP=127.0.0.1                    # Fix GUI/Server communication
export GZ_PARTITION=gazebo_default        # Fix GUI/Server communication
export LIBGL_DRI3_DISABLE=1               # Fix GLX/dri3 crashes
export QT_QPA_PLATFORM=xcb                # Fix frozen/copied background window

# Gazebo plugin paths for Ubuntu packages
export GZ_SIM_PHYSICS_ENGINE_PATH="/usr/lib/x86_64-linux-gnu/gz-physics-8/engine-plugins"
export GZ_RENDERING_ENGINE_PATH="/usr/lib/x86_64-linux-gnu/gz-rendering-9/engine-plugins"

echo "======================================"
echo " Shipboard Landing Simulation"
echo " Sea State: $SEA_STATE"
echo " Headless: $HEADLESS"
echo " Software Render: $SOFTWARE_RENDER"
echo "======================================"

# Set Gazebo Ionic paths
ASV_WAVE_SIM_DIR="$HOME/gz_ws/src/asv_wave_sim/gz-waves-models"

# Model paths: local models + asv_wave_sim models
export GZ_SIM_RESOURCE_PATH="${SCRIPT_DIR}/models:${ASV_WAVE_SIM_DIR}/models:${ASV_WAVE_SIM_DIR}/world_models:${GZ_SIM_RESOURCE_PATH}"

# Plugin paths: local plugins + asv_wave_sim plugins
export GZ_SIM_SYSTEM_PLUGIN_PATH="${SCRIPT_DIR}/plugins/build:$HOME/gz_ws/install/lib:${GZ_SIM_SYSTEM_PLUGIN_PATH}"

# Library path for asv_wave_sim
export LD_LIBRARY_PATH="$HOME/gz_ws/install/lib:${LD_LIBRARY_PATH}"

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
    gz sim -s --verbose "$WORLD_FILE" &
    GZ_PID=$!
else
    echo "Starting Gazebo with GUI..."
    gz sim -g --verbose "$WORLD_FILE" &
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
