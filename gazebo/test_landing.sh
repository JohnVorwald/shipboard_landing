#!/bin/bash
# Test landing controller in headless mode
cd "$(dirname "$0")"

export GZ_IP=127.0.0.1
export GZ_PARTITION=gazebo_default

# Kill any existing
killall -9 gz ruby-gz 2>/dev/null
sleep 2

echo "Starting Gazebo headless (paused)..."
gz sim -s worlds/ship_landing_simple.world &
GZ_PID=$!
sleep 5

echo "Starting controller (will unpause simulation)..."
python3 autonomous_landing_with_ship_motion.py &
CTRL_PID=$!

# Give controller time to initialize
sleep 1

echo "Unpausing simulation..."
gz service -s /world/ship_landing_simple/control --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean --timeout 2000 --req 'pause: false' 2>/dev/null || echo "Using gz topic instead..."

# Wait for controller
wait $CTRL_PID
RESULT=$?

echo ""
echo "Controller exit code: $RESULT"

# Cleanup
kill $GZ_PID 2>/dev/null
killall -9 gz ruby-gz 2>/dev/null

exit $RESULT
