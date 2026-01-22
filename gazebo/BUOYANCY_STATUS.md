# Gazebo Ship Buoyancy - WORKING

**Date:** 2026-01-15
**Status:** Ship floats and settles correctly

## Final Configuration

### Hydrodynamics Plugin (at model level)
```xml
<plugin filename="gz-sim-hydrodynamics-system" name="gz::sim::systems::Hydrodynamics">
  <link_name>hull</link_name>
  <xDotU>-2.0e4</xDotU>
  <yDotV>-2.0e4</yDotV>
  <zDotW>-5.0e6</zDotW>   <!-- Heave damping -->
  <kDotP>-5.0e6</kDotP>   <!-- Roll damping -->
  <mDotQ>-5.0e7</mDotQ>   <!-- Pitch damping -->
  <nDotR>-5.0e7</nDotR>   <!-- Yaw damping -->
  <xUabsU>-1.0e4</xUabsU>
  <yVabsV>-1.0e4</yVabsV>
  <zWabsW>-5.0e5</zWabsW>
</plugin>
```

### Graded Buoyancy (world level)
```xml
<plugin filename="gz-sim-buoyancy-system" name="gz::sim::systems::Buoyancy">
  <graded_buoyancy>
    <default_density>1025</default_density>
    <density_change>
      <above_depth>0</above_depth>
      <density>1</density>
    </density_change>
  </graded_buoyancy>
  <enable>ship</enable>
</plugin>
```

### Ship Parameters
- Mass: 4,000,000 kg
- Hull: 50m x 12m x 8m box collision
- Initial pose: z=0 (at waterline)
- Equilibrium: z ≈ -0.96 (settles in ~6 seconds)

## Test Results (Headless)

```
Time(s)  Z-position
   2     -0.73
   4     -0.99
   6     -0.96
   8     -0.95
  10     -0.97
  14     -0.98
  16     -0.97
```

Ship settles to equilibrium with minimal oscillation.

## Launch Commands

```bash
cd /home/john/github/shipboard_landing/gazebo

# Environment
export GZ_IP=127.0.0.1
export GZ_PARTITION=gazebo_default
export GZ_SIM_PHYSICS_ENGINE_PATH="/usr/lib/x86_64-linux-gnu/gz-physics-8/engine-plugins"
export GZ_SIM_RENDER_ENGINE_PATH="/usr/lib/x86_64-linux-gnu/gz-rendering-9/engine-plugins"

# With GUI
export LIBGL_DRI3_DISABLE=1
export QT_QPA_PLATFORM=xcb
gz sim --render-engine ogre worlds/ship_simple.world

# Headless
gz sim -s worlds/ship_simple.world
```

## Key Lessons Learned

1. **velocity_decay** is insufficient for large vessels - use **hydrodynamics plugin**
2. **Graded buoyancy** needed for water/air interface (not uniform_fluid_density)
3. **Hydrodynamics plugin** must be at **model level**, not inside link
4. **zDotW** (heave damping) is critical for vertical settling
5. Start ship at waterline (z=0) to minimize initial oscillation

## Remaining Optional Tasks

- Add ocean wave visuals (asv_wave_sim needs rebuild)
- Add quadrotor model for landing simulation
- Add ship motion (forward movement, wave response)
