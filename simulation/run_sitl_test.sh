#!/bin/bash
# SITL Test Launch Script
# Coordinates Gazebo, ArduPilot SITL, and test script startup

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAZEBO_DIR="$(dirname "$SCRIPT_DIR")/gazebo"
ARDUPILOT_DIR="$HOME/ardupilot"
WORLD_FILE="worlds/ship_landing_ardupilot.sdf"

# Environment for Gazebo
export GZ_SIM_SYSTEM_PLUGIN_PATH=/home/john/gz_ws/src/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
export GZ_SIM_RESOURCE_PATH=/home/john/gz_ws/src/ardupilot_gazebo/models:/home/john/gz_ws/src/ardupilot_gazebo/worlds:$GZ_SIM_RESOURCE_PATH

# Log files
GZ_LOG="/tmp/gz_sitl_test.log"
SITL_LOG="/tmp/sitl_test.log"

cleanup() {
    echo "Cleaning up..."
    # Kill any existing processes
    pkill -f "gz sim" 2>/dev/null || true
    pkill -f "arducopter" 2>/dev/null || true
    pkill -f "mavproxy" 2>/dev/null || true
    sleep 1
}

# Cleanup on exit
trap cleanup EXIT

echo "=== SITL Test Launch Script ==="
echo ""

# Clean up any existing processes
cleanup

echo "[1/4] Starting Gazebo server (headless + running)..."
cd "$GAZEBO_DIR"
# -s = server only (no GUI), -r = run immediately (not paused)
gz sim -s -r "$WORLD_FILE" > "$GZ_LOG" 2>&1 &
GZ_PID=$!
echo "  Gazebo PID: $GZ_PID"
echo "  Log: $GZ_LOG"

# Wait for Gazebo to initialize
echo "  Waiting for Gazebo to initialize..."
sleep 8

# Check if Gazebo is still running
if ! kill -0 $GZ_PID 2>/dev/null; then
    echo "  ERROR: Gazebo failed to start!"
    cat "$GZ_LOG"
    exit 1
fi

# Check for topics
if gz topic -l 2>/dev/null | grep -q "ship_landing_ardupilot"; then
    echo "  Gazebo ready - topics available"
else
    echo "  WARNING: Gazebo topics not found, but process is running"
fi

echo ""
echo "[2/4] Starting ArduPilot SITL..."
cd "$ARDUPILOT_DIR/ArduCopter"

# Start SITL with MAVProxy (needed for test script connection on port 5760)
../Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON > "$SITL_LOG" 2>&1 &
SITL_PID=$!
echo "  SITL PID: $SITL_PID"
echo "  Log: $SITL_LOG"

# Wait for SITL to initialize
echo "  Waiting for SITL to initialize..."
sleep 10

# Check if SITL is running
if ! pgrep -f "arducopter" > /dev/null; then
    echo "  ERROR: ArduCopter SITL failed to start!"
    tail -50 "$SITL_LOG"
    exit 1
fi
echo "  SITL ready"

echo ""
echo "[3/4] System ready for testing"
echo "  - Gazebo server running (PID: $GZ_PID)"
echo "  - ArduCopter SITL running"
echo ""

# Check what test to run
TEST_SCRIPT="${1:-static_target_test.py}"

echo "[4/4] Running test: $TEST_SCRIPT"
cd "$SCRIPT_DIR"

# Run the test script
python3 "$TEST_SCRIPT"
TEST_EXIT=$?

echo ""
echo "=== Test Complete (exit code: $TEST_EXIT) ==="

exit $TEST_EXIT
