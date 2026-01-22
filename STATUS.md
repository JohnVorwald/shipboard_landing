# Shipboard Landing Project Status

**Date:** 2026-01-21

## Current Status: ArduPilot + Gazebo Harmonic Working

### What's Working

#### 1. ArduPilot SITL with Gazebo Harmonic
- **Gazebo Harmonic (v8.10.0)** - Required (Ionic v9 has altitude bugs)
- **ArduPilot SITL** connected via JSON interface
- **EKF healthy** (flags 831)
- **Quad takeoff, hover, position control** all working
- **Ship world with helipad** - quad spawns on deck

#### 2. Pure Python Controllers (100% Success)
- **ZEM/ZEV** (Georgia Tech style): 100% success rate
- **Tau-Based** (Penn State): Works well
- **PMP** (Pontryagin): 100% success rate

---

## Quick Start Commands

### Terminal Setup for Remote Desktop

If connecting via remote desktop (VNC/RDP), first find your display:
```bash
echo $DISPLAY
# Usually :10 or :10.0 for remote desktop
```

### Option 1: Use Launch Script (Recommended)

**Terminal 1 - Gazebo:**
```bash
/home/john/github/shipboard_landing/gazebo/launch_ship.sh
```

**Terminal 2 - ArduPilot:**
```bash
cd /home/john/ardupilot && ./Tools/autotest/sim_vehicle.py -v ArduCopter -f JSON --model JSON --console
```

### Option 2: Manual Commands

**Terminal 1 - Gazebo with GUI:**
```bash
export GZ_SIM_RESOURCE_PATH=/home/john/gz_ws/src/ardupilot_gazebo/models:/home/john/gz_ws/src/ardupilot_gazebo/worlds:/home/john/github/shipboard_landing/gazebo/models
export GZ_SIM_SYSTEM_PLUGIN_PATH=/home/john/gz_ws/src/ardupilot_gazebo/build
export DISPLAY=:10.0
export GZ_IP=127.0.0.1
export GZ_PARTITION=gazebo_default
export LIBGL_DRI3_DISABLE=1
export QT_QPA_PLATFORM=xcb
gz sim -r /home/john/github/shipboard_landing/gazebo/worlds/ship_landing_ardupilot.sdf
```

**Terminal 1 - Gazebo Headless (no GUI):**
```bash
export GZ_SIM_RESOURCE_PATH=/home/john/gz_ws/src/ardupilot_gazebo/models:/home/john/gz_ws/src/ardupilot_gazebo/worlds:/home/john/github/shipboard_landing/gazebo/models
export GZ_SIM_SYSTEM_PLUGIN_PATH=/home/john/gz_ws/src/ardupilot_gazebo/build
export GZ_IP=127.0.0.1
export GZ_PARTITION=gazebo_default
gz sim -s -r /home/john/github/shipboard_landing/gazebo/worlds/ship_landing_ardupilot.sdf
```

**Terminal 2 - ArduPilot SITL:**
```bash
cd /home/john/ardupilot
./Tools/autotest/sim_vehicle.py -v ArduCopter -f JSON --model JSON --console
```

---

## MAVProxy Commands

### Basic Flight
```bash
mode guided              # Enable position control
arm throttle force       # Arm motors (force bypasses checks)
takeoff 10               # Takeoff to 10m altitude
mode land                # Land
mode loiter              # Hold position
disarm                   # Disarm motors
```

### Position Control
**Note: `position` command is RELATIVE to current position (NED frame)**
```bash
position 10 0 0          # Move 10m North
position 0 10 0          # Move 10m East
position 0 0 -5          # Move 5m Up (Z negative = up in NED)
position 0 0 5           # Move 5m Down
position -10 -10 0       # Move 10m South and 10m West
```

### Show Current Position
```bash
status GLOBAL_POSITION_INT    # Show lat/lon/alt
status LOCAL_POSITION_NED     # Show x/y/z in meters
status                        # Show all telemetry
watch GLOBAL_POSITION_INT     # Continuous position updates
nowatch GLOBAL_POSITION_INT   # Stop watching
```

### Other Useful Commands
```bash
mode stabilize           # Manual control mode
mode rtl                 # Return to launch
param show FRAME_CLASS   # Show parameter
param set FRAME_CLASS 1  # Set parameter
arm list                 # Show arm checks
```

---

## Display Troubleshooting

### Remote Desktop Shows Frozen/Copied Background
Add these environment variables:
```bash
export LIBGL_DRI3_DISABLE=1
export QT_QPA_PLATFORM=xcb
```

### Find Correct Display Number
```bash
echo $DISPLAY              # Check current display
w                          # Show logged in users and displays
```

### Switch Display
```bash
export DISPLAY=:10.0       # For remote desktop (usually :10)
export DISPLAY=:0          # For local console
export DISPLAY=:1          # Alternative local
```

### Kill Stuck Gazebo Processes
```bash
killall -9 gz 2>/dev/null
pgrep -a gz                # Verify all killed
```

---

## Key Files

| File | Purpose |
|------|---------|
| `gazebo/launch_ship.sh` | Launch script with all env vars |
| `gazebo/worlds/ship_landing_ardupilot.sdf` | Ship world with helipad |
| `/home/john/gz_ws/src/ardupilot_gazebo/` | ArduPilot Gazebo plugin |

---

## Coordinate Frames

### Gazebo (ENU)
- X = East
- Y = North
- Z = Up

### ArduPilot (NED)
- X = North
- Y = East
- Z = Down (negative = up)

### MAVProxy `position` Command
- Uses NED frame
- **RELATIVE to current position**
- `position 0 0 -10` = move up 10m from current location

---

## Troubleshooting

### Quad Falls Through World
- Missing ground/deck collision
- Check ship has `<collision>` element for deck

### EKF Flags = 1024 (Bad)
- GPS/altitude issue
- Switch to Gazebo Harmonic (v8), not Ionic (v9)

### "Address already in use" Error
- Old ArduCopter process running
- `killall -9 arducopter`

### Gazebo GUI Crashes (Ogre Assertion)
- Use `--render-engine ogre` or run headless with `-s` flag
- Reduce water plane size (500x500 max)

---

## Next Steps

1. **Enable ship motion** - Re-enable velocity control plugin for moving deck
2. **Autonomous landing** - Connect guidance controllers to ArduPilot
3. **Test precision landing** - Use IR beacon or visual markers
