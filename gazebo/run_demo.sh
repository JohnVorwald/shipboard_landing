#!/bin/bash
# Autonomous Landing Demo
# Starts Gazebo and runs the landing controller

cd "$(dirname "$0")"

# Environment
export GZ_IP=127.0.0.1
export GZ_PARTITION=gazebo_default
export DISPLAY=${DISPLAY:-:10.0}
export LIBGL_DRI3_DISABLE=1
export QT_QPA_PLATFORM=xcb

# Kill any existing Gazebo
pkill -9 -f "gz sim" 2>/dev/null
sleep 2

echo "=============================================="
echo "  Autonomous Shipboard Landing Demo"
echo "=============================================="
echo ""
echo "Starting Gazebo..."

# Start Gazebo (paused)
gz sim --render-engine ogre worlds/ship_landing_simple.world &
GZ_PID=$!

echo "Waiting for Gazebo to initialize..."
sleep 8

# Check if running
if ! kill -0 $GZ_PID 2>/dev/null; then
    echo "ERROR: Gazebo failed to start"
    exit 1
fi

echo ""
echo "=============================================="
echo "  PRESS PLAY (▶️) IN GAZEBO NOW!"
echo "=============================================="
echo ""
echo "Then the controller will start in 5 seconds..."
sleep 5

# Run controller
python3 autonomous_landing.py

# Cleanup
echo ""
echo "Demo complete. Closing Gazebo in 10 seconds..."
sleep 10
kill $GZ_PID 2>/dev/null
