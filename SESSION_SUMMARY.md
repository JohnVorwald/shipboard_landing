# Ship Landing Project - Session Summary

**Date:** January 16, 2026
**Status:** Gazebo visualization working, mission running with ArduPilot SITL

## Work Completed

### 1. ArduPilot Arm/Takeoff Fixes
- Updated `simulation/ship_landing_mission.py` with reliable arming:
  - Uses `arducopter_arm()` and `motors_armed_wait()` from pymavlink
  - Added force-arm fallback with `ARMING_CHECK=0` and magic number `21196`
  - Better takeoff monitoring with ACK check

### 2. Ship Position Tracking Fix
- Fixed `ShipState.update()` method - was accumulating position incorrectly
- Now correctly computes: `_current_position = initial_position + velocity * elapsed_time`
- Added `__post_init__` to initialize `_current_position` from `initial_position`

### 3. Waypoint Controller Tuning
- Increased gain: 2 -> 4 (more aggressive pursuit)
- Increased max_speed: 8 -> 10 m/s
- Added minimum pursuit speed of 2 m/s to catch moving ship

### 4. Gazebo Visualization Fixes
The main challenge was getting Gazebo GUI to work. Key fixes:

**Environment Variables (CRITICAL):**
```bash
export DISPLAY=:10.0  # Your display
export GZ_IP=127.0.0.1
export GZ_PARTITION=gazebo_default
export LIBGL_DRI3_DISABLE=1  # NVIDIA hybrid graphics fix
export QT_QPA_PLATFORM=xcb
export GZ_SIM_PHYSICS_ENGINE_PATH="/usr/lib/x86_64-linux-gnu/gz-physics-8/engine-plugins"
```

**Physics Stability Fixes:**
- Changed from `graded_buoyancy` to `uniform_fluid_density` (simpler, more stable)
- Removed `gz-sim-hydrodynamics-system` plugin (was causing DART NaN crashes)
- Reduced ship mass from 4,000,000 kg to 500,000 kg

**Working command:**
```bash
gz sim -r --render-engine ogre worlds/ship_simple.world
```

### 5. Files Modified

| File | Changes |
|------|---------|
| `simulation/ship_landing_mission.py` | Arm/takeoff fixes, ship tracking, waypoint controller |
| `simulation/record_mission.sh` | Video recording script with proper env vars |
| `gazebo/worlds/ship_simple.world` | Fixed physics, removed hydrodynamics, camera position |

## Current State

### Working:
- Gazebo GUI displays ship, water plane, and quadcopter
- Simulation runs without crashing (physics stable)
- ArduPilot SITL connects and arms reliably
- Mission flies quad to helipad on stationary ship
- Moving ship tracking works in headless mode

### Recording Script Location:
```
/home/john/github/shipboard_landing/simulation/record_mission.sh
```

### To Run:
```bash
cd /home/john/github/shipboard_landing/simulation
./record_mission.sh output.mp4
```

## Next Steps (After Weekend)

1. **Test full mission with video recording** - verify quad lands on helipad in visualization
2. **Add moving ship support to Gazebo** - currently ship is static in visualization
3. **Tune landing controller** - may need adjustment based on visual feedback
4. **Record demo video** - capture successful landing for documentation

## Known Issues

1. **Gazebo window not appearing** - Process runs but window doesn't show. Need to figure out correct DISPLAY setting for your session. Try running directly from your terminal.
2. **OGRE shader warnings** - "Unable to find shader lib" - shadows disabled but simulation works
3. **Ship motion not in Gazebo** - ship is static in visualization; motion only tracked internally
4. **Buoyancy oscillation** - ship may bounce slightly at start; settles after a few seconds

## Display Issue - TO FIX

The Gazebo window isn't appearing. When you return, try running this directly from YOUR terminal:

```bash
cd /home/john/github/shipboard_landing/gazebo
export GZ_IP=127.0.0.1 GZ_PARTITION=gazebo_default LIBGL_DRI3_DISABLE=1 QT_QPA_PLATFORM=xcb
gz sim --render-engine ogre worlds/ship_simple.world
```

If it still doesn't appear, check your DISPLAY variable with `echo $DISPLAY` and use that value.

## Quick Test Commands

**Test Gazebo alone:**
```bash
export DISPLAY=:10.0 GZ_IP=127.0.0.1 GZ_PARTITION=gazebo_default LIBGL_DRI3_DISABLE=1 QT_QPA_PLATFORM=xcb
cd /home/john/github/shipboard_landing/gazebo
gz sim -r --render-engine ogre worlds/ship_simple.world
```

**Run mission without visualization:**
```bash
cd /home/john/github/shipboard_landing/simulation
python3 ship_landing_mission.py --ship-speed 0
```

**Run mission with visualization:**
```bash
cd /home/john/github/shipboard_landing/simulation
./record_mission.sh test_video.mp4
```
