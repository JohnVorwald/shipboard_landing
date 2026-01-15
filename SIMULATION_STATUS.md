# Shipboard Landing Simulation - Status

**Last Updated:** 2026-01-15

## Current Status: PARTIALLY WORKING

Ship and landing pad visible. Ship sinks when simulation runs (physics works but no buoyancy). Ocean shows as white plane instead of animated waves.

## What's Working

- Gazebo GUI opens and renders 3D scene
- Ship model visible with helipad
- Physics simulation runs (gravity works)
- ShipMotionPlugin loads (sea state 5)
- All environment variables configured

## Remaining Issues

### 1. Ocean Waves Not Rendering (White Plane)
The asv_wave_sim visual plugin needs OGRE2 but can't find `libOgreNextMain.so.2.3.1`.

**Fix:** Add OGRE2 path to LD_LIBRARY_PATH:
```bash
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/OGRE-2.3:$LD_LIBRARY_PATH"
```

### 2. Ship Sinks (No Buoyancy)
The WavesModel physics (buoyancy) may not be connecting to ship. Need to verify ship has buoyancy plugin attached.

### 3. Plugin Symlinks Required
Ubuntu packages don't create unversioned symlinks. These were created manually:
```bash
sudo ln -sf libgz-physics-dartsim-plugin.so.8 /usr/lib/x86_64-linux-gnu/gz-physics-8/engine-plugins/libgz-physics-dartsim-plugin.so
sudo ln -sf libgz-rendering9-ogre.so.9 /usr/lib/x86_64-linux-gnu/gz-rendering-9/engine-plugins/libgz-rendering-ogre.so
sudo ln -sf libgz-rendering9-ogre2.so.9 /usr/lib/x86_64-linux-gnu/gz-rendering-9/engine-plugins/libgz-rendering-ogre2.so
```

## Environment Variables (All Required)

```bash
# GUI/Server communication
export GZ_IP=127.0.0.1
export GZ_PARTITION=gazebo_default

# Graphics fixes for NVIDIA hybrid
export LIBGL_DRI3_DISABLE=1
export QT_QPA_PLATFORM=xcb

# Plugin paths
export GZ_SIM_PHYSICS_ENGINE_PATH="/usr/lib/x86_64-linux-gnu/gz-physics-8/engine-plugins"
export GZ_SIM_RENDER_ENGINE_PATH="/usr/lib/x86_64-linux-gnu/gz-rendering-9/engine-plugins"

# OGRE2 for waves visual (CRITICAL)
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/OGRE-2.3:$HOME/gz_ws/install/lib:$LD_LIBRARY_PATH"

# Resource paths
export GZ_SIM_RESOURCE_PATH="models:$HOME/gz_ws/src/asv_wave_sim/gz-waves-models/models"
export GZ_SIM_SYSTEM_PLUGIN_PATH="plugins/build:$HOME/gz_ws/install/lib"
```

## Launch Script Location

`/tmp/run_gz.sh` - Contains all environment variables and launches with ogre2

## How to Test

```bash
DISPLAY=:0 /tmp/run_gz.sh
```

Or from project directory:
```bash
cd ~/github/shipboard_landing/gazebo
./launch_sim.sh
```

## Next Steps to Fix

1. **Test with OGRE2 path added** - May fix ocean wave rendering
2. **Check ship buoyancy configuration** - Ship needs buoyancy plugin in world file
3. **Consider using ogre2 render engine** - Add `--render-engine ogre2` flag

## Files Modified

- `worlds/ship_landing.world` - Changed physics to `dart`, sensors to `ogre`
- `~/.gz/sim/9/gui.config` - Changed engine to `ogre`
- `/tmp/run_gz.sh` - Test launch script with all fixes
