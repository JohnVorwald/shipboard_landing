#!/bin/bash
# Wrapper script to launch ArduPilot SITL with MAVProxy
# Called by ship_landing_demo.py

echo "=== LAUNCHING ARDUPILOT SITL ==="

# Use display from argument or detect
if [ -n "$1" ]; then
    export DISPLAY="$1"
fi

echo "DISPLAY: $DISPLAY"
echo ""

cd /home/john/ardupilot

echo "Running: sim_vehicle.py -v ArduCopter -f JSON --model JSON --console"
echo "=== ArduPilot SITL starting... ==="
echo ""

# Execute sim_vehicle.py (exec replaces this shell process)
exec ./Tools/autotest/sim_vehicle.py -v ArduCopter -f JSON --model JSON --console
