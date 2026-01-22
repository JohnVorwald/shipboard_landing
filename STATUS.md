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
mode land                # Land at current position
mode loiter              # Hold position
disarm                   # Disarm motors
```

### GUIDED Mode Flying

#### Position Control (RELATIVE to current position)
**IMPORTANT: `position` command moves RELATIVE to where you are now, NOT absolute!**
```bash
position 10 0 0          # Move 10m North from current position
position 0 10 0          # Move 10m East from current position
position 0 0 -5          # Move 5m Up (Z negative = up in NED)
position 0 0 5           # Move 5m Down
position -10 -10 0       # Move 10m South and 10m West
```

#### Return to Home/Helipad
```bash
# Option 1: RTL mode (automatic return and land)
mode rtl                 # Return to launch point, then land

# Option 2: Manual return using LOCAL_POSITION_NED
# First check your position relative to home:
status LOCAL_POSITION_NED
# Shows: x (North), y (East), z (Down, negative=up)
# To return to home at current altitude:
# If you're at x=20, y=30, z=-10: use position -20 -30 0

# Option 3: Land mode at current position
mode land                # Land where you are
```

#### Show Position Relative to Home
```bash
# LOCAL_POSITION_NED shows position relative to home/launch point
status LOCAL_POSITION_NED
# Output example:
#   LOCAL_POSITION_NED {x: 15.2, y: -3.4, z: -10.5, vx: 0.1, vy: 0.0, vz: 0.0}
#   x = 15.2m North of home
#   y = -3.4m East of home (negative = West)
#   z = -10.5m Down (negative = 10.5m UP/altitude)

# Continuous monitoring:
watch LOCAL_POSITION_NED
nowatch LOCAL_POSITION_NED   # Stop watching
```

#### Show GPS Position
```bash
status GLOBAL_POSITION_INT    # Show lat/lon/alt
# Output includes:
#   lat, lon: GPS coordinates (divide by 1e7 for degrees)
#   alt: MSL altitude in mm
#   relative_alt: Altitude above home in mm

watch GLOBAL_POSITION_INT     # Continuous GPS updates
```

#### Altitude Control
```bash
# In GUIDED mode, altitude is the negative Z component
position 0 0 -15         # Go up to 15m altitude (relative move)
position 0 0 -5          # Go up 5m from current altitude
position 0 0 5           # Go down 5m from current altitude

# To go to specific altitude, calculate from current:
# If at z=-10 (10m alt) and want 20m: position 0 0 -10
```

#### Example Flight Sequence
```bash
mode guided              # Enable GUIDED mode
arm throttle force       # Arm with force (bypasses pre-arm checks)
takeoff 10               # Takeoff to 10m

# Fly a square pattern (each move is RELATIVE)
position 10 0 0          # 10m North
position 0 10 0          # 10m East (now at N=10, E=10)
position -10 0 0         # 10m South (now at N=0, E=10)
position 0 -10 0         # 10m West (back to N=0, E=0 = home)

mode land                # Land
# or
mode rtl                 # Return to launch and land
```

### Other Useful Commands
```bash
mode stabilize           # Manual control mode
mode loiter              # Hold current position
mode rtl                 # Return to launch
param show FRAME_CLASS   # Show parameter
param set FRAME_CLASS 1  # Set parameter
arm list                 # Show arm checks
status                   # Show all telemetry
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
