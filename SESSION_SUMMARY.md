# Ship Landing Project - Session Summary

**Date:** January 22, 2026
**Status:** ArduPilot + Gazebo integration WORKING with manual commands

---

## Current Session Progress

### What's Working

1. **Gazebo + ArduPilot Connection** - CONFIRMED WORKING
   - Gazebo Harmonic v8.10.0 with ship world and iris_with_gimbal
   - ArduPilot SITL connects via JSON interface
   - EKF initializes and becomes healthy
   - Arming works (with force arm)
   - Takeoff and landing work

2. **Manual Flight Test** - SUCCESS
   ```python
   # Connected to tcp:127.0.0.1:5760
   # Armed on attempt 3
   # Took off to 10m
   # Landed
   ```

3. **Environment Variables** - RESOLVED
   - Must prepend custom paths to existing GZ_SIM_RESOURCE_PATH
   - Direct subprocess.Popen with env dict works reliably
   - gnome-terminal approach doesn't pass env vars correctly

### Working Launch Commands

**Terminal 1 - Gazebo:**
```bash
export GZ_SIM_RESOURCE_PATH=/home/john/gz_ws/src/ardupilot_gazebo/models:/home/john/gz_ws/src/ardupilot_gazebo/worlds:/home/john/github/shipboard_landing/gazebo/models
export GZ_SIM_SYSTEM_PLUGIN_PATH=/home/john/gz_ws/src/ardupilot_gazebo/build
export DISPLAY=:10  # or :0 for local
export GZ_IP=127.0.0.1 GZ_PARTITION=gazebo_default LIBGL_DRI3_DISABLE=1 QT_QPA_PLATFORM=xcb
gz sim -r /home/john/github/shipboard_landing/gazebo/worlds/ship_landing_ardupilot.sdf
```

**Terminal 2 - ArduPilot (after Gazebo shows ship):**
```bash
cd /home/john/ardupilot
./Tools/autotest/sim_vehicle.py -v ArduCopter -f JSON --model JSON --no-mavproxy
```

**Python Connection:**
```python
from pymavlink import mavutil
master = mavutil.mavlink_connection('tcp:127.0.0.1:5760', source_system=255)
master.wait_heartbeat(timeout=15)  # Wait for system 1
# Then: set_mode(4), arm, takeoff, etc.
```

### Automated Demo Script

`simulation/ship_landing_demo.py` - Needs timing adjustment

**Current Issue:**
Script connects to ArduPilot before Gazebo-ArduPilot JSON connection is established.
ArduPilot reports port 5760 open quickly, but heartbeat not available until
JSON interface with Gazebo is working.

**Fix Needed:**
- Wait for ArduPilot to show "JSON received" in output before connecting
- Or wait for heartbeat with longer timeout and more retries

### Key Findings from Debugging

1. **Port 9002 is NOT a TCP listener** - ArduPilotPlugin uses UDP for JSON
2. **Heartbeat only available after Gazebo connection** - ArduPilot won't send
   heartbeats until it receives JSON data from Gazebo
3. **EKF may timeout without EKF_STATUS_REPORT** - Use longer waits
4. **Force arm (21196) works** when normal arm fails due to checks

### Files Modified

| File | Status |
|------|--------|
| `simulation/ship_landing_demo.py` | Updated - needs timing fix |
| `gazebo/launch_ship.sh` | Working launch script |
| `STATUS.md` | Updated with commands |
| `SESSION_SUMMARY.md` | This file |

---

## Next Steps

1. **Fix automated demo timing** - Wait for ArduPilot-Gazebo sync before connect
2. **Add verbose flight output** - Print arming attempts, altitude, position
3. **Video recording** - Verify screen capture and telemetry overlay work
4. **Ship motion** - Re-enable moving ship for landing tests

## Quick Test

```bash
# Clean start
pkill -9 -f 'gz sim'; pkill -9 -f arducopter; sleep 2

# Launch Gazebo
bash /home/john/github/shipboard_landing/gazebo/launch_ship.sh &
sleep 15

# Launch ArduPilot
cd /home/john/ardupilot && ./Tools/autotest/sim_vehicle.py -v ArduCopter -f JSON --model JSON --no-mavproxy &
sleep 30

# Connect and fly (see STATUS.md for commands)
```
