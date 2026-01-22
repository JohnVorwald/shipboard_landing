# Comprehensive Gemini AI Assistance Request: Shipboard Landing Project

## Project Overview

I'm developing an autonomous quadcopter landing system for a moving ship deck using optimal control theory. The project integrates:

1. **Gazebo Ionic (gz-sim 9.5.0)** for physics simulation
2. **ArduPilot SITL** for flight control and MAVLink communication
3. **Pontryagin Maximum Principle (PMP)** optimal control for trajectory generation
4. **Ship motion simulation** with realistic buoyancy and wave effects

## Current Project Status

### Working Components

1. **Ship Buoyancy in Gazebo** (WORKING)
   - Ship floats at z ≈ -0.96 with proper water/air interface
   - Graded buoyancy: water (1025 kg/m³) below z=0, air (1 kg/m³) above
   - Hydrodynamics damping plugin settles ship in ~6 seconds
   - World file: `gazebo/worlds/ship_simple.world`

2. **ArduPilot SITL Connection** (WORKING)
   - MAVLink connection on `tcp:127.0.0.1:5760`
   - Proper EKF and GPS initialization waiting (45+ seconds)
   - Vehicle arms successfully after GPS fix and EKF healthy

3. **Existing Controllers** (IMPLEMENTED)
   - PMP optimal control: `optimal_control/pmp_controller.py`
   - Trajectory planner: `optimal_control/trajectory_planner.py`
   - Tau guidance: `guidance/tau_guidance.py`
   - ZEM guidance: `guidance/zem_guidance.py`

### Issues Needing Resolution

1. **ArduPilot SITL Takeoff Not Responding**
   - Vehicle arms successfully but takeoff command doesn't work
   - Altitude stays at 0 despite MAV_CMD_NAV_TAKEOFF being sent
   - SITL runs with: `sim_vehicle.py -v ArduCopter -f quad --no-mavproxy`
   - Possible issue with model configuration or vehicle state

2. **Gazebo Physics Crashes with Quadcopter**
   - ODE collision detector crashes with large meshes (ship + quadcopter)
   - Error: `ODE INTERNAL ERROR 1: assertion "aabbBound >= dMinIntExact && aabbBound < dMaxIntExact"`
   - Only DART physics available (no Bullet or TPE in gz-sim 9)

3. **Gazebo-ArduPilot Integration**
   - Need to connect ArduPilot SITL to Gazebo for visual feedback
   - ArduPilot Gazebo Plugin (ardupilot_gazebo) needs proper model setup
   - Currently running ArduPilot standalone without Gazebo visualization

## Technical Details

### File Structure
```
/home/john/github/shipboard_landing/
├── optimal_control/
│   ├── pmp_controller.py       # PMP optimal control implementation
│   ├── trajectory_planner.py   # Min-snap trajectory generation
│   ├── pmp_landing.py          # Landing-specific PMP
│   └── pontryagin.py           # Core Pontryagin theory
├── guidance/
│   ├── tau_guidance.py         # Tau-based guidance law
│   ├── zem_guidance.py         # Zero-Effort-Miss guidance
│   └── higher_order_tau.py     # Higher-order tau guidance
├── quad_dynamics/
│   └── quadrotor.py            # Quadrotor dynamics model
├── ship_motion/
│   └── ddg_motion.py           # DDG-51 destroyer motion model
├── simulation/
│   ├── sitl_landing_test.py    # SITL testing script (NEW)
│   ├── ardupilot_pmp_gazebo.py # ArduPilot + PMP execution
│   ├── landing_sim.py          # Pure Python landing sim
│   └── mpc_landing.py          # MPC-based landing
├── gazebo/
│   ├── worlds/
│   │   └── ship_simple.world   # Gazebo world with ship + buoyancy
│   └── plugins/
│       └── quadrotor_controller_plugin.cc  # Stub for GZ Ionic
└── tests/
    └── *.py                    # Unit tests
```

### Gazebo World Configuration (ship_simple.world)

```xml
<!-- Physics -->
<physics name="dart" type="dart">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
</physics>

<!-- Buoyancy Plugin -->
<plugin filename="gz-sim-buoyancy-system" name="gz::sim::systems::Buoyancy">
  <graded_buoyancy>
    <default_density>1025</default_density>  <!-- Water -->
    <density_change>
      <above_depth>0</above_depth>
      <density>1</density>  <!-- Air -->
    </density_change>
  </graded_buoyancy>
  <enable>ship</enable>
</plugin>

<!-- Hydrodynamics Damping -->
<plugin filename="gz-sim-hydrodynamics-system" name="gz::sim::systems::Hydrodynamics">
  <link_name>hull</link_name>
  <zDotW>-5.0e6</zDotW>   <!-- Heave damping -->
  <kDotP>-5.0e6</kDotP>   <!-- Roll damping -->
  <mDotQ>-5.0e7</mDotQ>   <!-- Pitch damping -->
  <nDotR>-5.0e7</nDotR>   <!-- Yaw damping -->
</plugin>
```

### ArduPilot SITL Commands

```bash
# Start SITL
cd /home/john/ardupilot
./Tools/autotest/sim_vehicle.py -v ArduCopter -f quad --no-mavproxy

# Connect via pymavlink
from pymavlink import mavutil
master = mavutil.mavlink_connection('tcp:127.0.0.1:5760')
master.wait_heartbeat()

# Wait for EKF (critical - needs 45+ seconds)
# Required EKF flags: ATTITUDE | VELOCITY_HORIZ | VELOCITY_VERT | POS_HORIZ_REL | POS_HORIZ_ABS
```

### EKF Waiting Pattern (from ArduPilot autotest)

```python
# Required EKF flags for arming:
required_flags = (
    EKF_ATTITUDE |           # 0x001
    EKF_VELOCITY_HORIZ |     # 0x002
    EKF_VELOCITY_VERT |      # 0x004
    EKF_POS_HORIZ_REL |      # 0x008
    EKF_POS_HORIZ_ABS |      # 0x010
    EKF_POS_VERT_ABS |       # 0x020
    EKF_PRED_POS_HORIZ_REL | # 0x100
    EKF_PRED_POS_HORIZ_ABS   # 0x200
)

# Error flags (must NOT be set):
error_flags = (
    EKF_CONST_POS_MODE |     # 0x080
    EKF_GPS_GLITCH |         # 0x400
    EKF_ACCEL_ERROR          # 0x800
)
```

## Specific Questions for Gemini

### 1. ArduPilot SITL Takeoff Issue

The vehicle arms successfully but the takeoff command has no effect. Here's my takeoff code:

```python
def takeoff(self, altitude: float = 10.0):
    self.master.mav.command_long_send(
        self.master.target_system,
        self.master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude
    )
```

**Questions:**
- Do I need to set mode to GUIDED before takeoff?
- Should I use `MAV_CMD_NAV_TAKEOFF_LOCAL` instead?
- Is there a required sequence: arm -> mode -> takeoff?
- Are there specific parameters I need to check (like `PILOT_TKOFF_ALT`)?

### 2. Gazebo-ArduPilot Integration

I want to visualize ArduPilot flight in Gazebo. Options seem to be:

**Option A**: Use ardupilot_gazebo plugin
- Requires specific Iris model with motor plugins
- How to set up for Gazebo Ionic (gz-sim 9)?

**Option B**: External bridge
- Write Python script to read ArduPilot state and update Gazebo model pose
- Use `gz topic` to publish model poses

**Option C**: JSON interface
- Use ArduPilot's JSON SITL interface
- Connect Gazebo as external physics sim

**Questions:**
- Which approach is most reliable for Gazebo Ionic?
- How do I configure the ardupilot_gazebo plugin for gz-sim 9?
- Can you provide example code for Option B (bridge script)?

### 3. ODE Collision Crash

When adding the quadcopter model to the ship world, Gazebo's ODE collision detector crashes:

```
ODE INTERNAL ERROR 1: assertion "aabbBound >= dMinIntExact && aabbBound < dMaxIntExact" failed
```

**Questions:**
- Is this a known issue with large coordinate differences?
- Can I use a different collision detector with DART physics?
- Should I simplify collision geometry for the ship?
- Is there a way to disable collision checking between specific models?

### 4. PMP Trajectory Execution

My PMP controller generates optimal trajectories, but I'm not sure how to best execute them on ArduPilot:

```python
# Current approach: send position targets at 10Hz
for waypoint in trajectory:
    self.master.mav.set_position_target_local_ned_send(
        ..., x, y, z, vx, vy, vz, ...
    )
    time.sleep(0.1)
```

**Questions:**
- Should I use position targets (SET_POSITION_TARGET_LOCAL_NED) or attitude targets?
- What's the optimal update rate for trajectory following?
- How do I handle the time synchronization between planned and actual trajectory?
- Should I implement a tracking controller on top of ArduPilot's position controller?

### 5. Ship Motion Prediction

For landing on a moving deck, I need to predict ship motion. Currently I have:
- DDG-51 destroyer motion model
- Spectral analysis for periodic motion
- But no real-time prediction integration with ArduPilot

**Questions:**
- How should I integrate ship motion prediction with the PMP trajectory planner?
- Should I replan the trajectory continuously or use a feedback controller?
- What's a good approach for handling uncertain ship motion?

## Code Snippets for Reference

### PMP Controller Interface

```python
class PMPController:
    def compute_optimal_trajectory(self, x0, xf, tf):
        """
        Compute minimum-effort trajectory using Pontryagin's Maximum Principle.

        Args:
            x0: Initial state [pos, vel]
            xf: Final state [pos, vel]
            tf: Time horizon

        Returns:
            trajectory: (times, states, controls, costates)
        """
        pass
```

### Ship Motion Model

```python
class DDGMotion:
    def predict_deck_state(self, t, forecast_horizon=5.0):
        """
        Predict deck position and velocity.

        Returns:
            deck_pos: [x, y, z]
            deck_vel: [vx, vy, vz]
            deck_acc: [ax, ay, az]
        """
        pass
```

## Environment Information

- **OS**: Ubuntu 24.04 LTS
- **Gazebo**: Ionic (gz-sim 9.5.0)
- **ArduPilot**: Latest master branch
- **Python**: 3.12
- **pymavlink**: 2.4.49

## Desired Outcome

I need help to:

1. **Get ArduPilot SITL takeoff working** - Currently arms but doesn't takeoff
2. **Integrate ArduPilot with Gazebo Ionic** - For visualization and closed-loop testing
3. **Execute PMP trajectories** - Send optimal trajectory commands to ArduPilot
4. **Implement ship deck tracking** - Land on moving platform

## Additional Context

The project is for research on autonomous shipboard landing using optimal control theory. The key innovation is using Pontryagin's Maximum Principle to generate fuel-optimal trajectories that account for ship deck motion. The control architecture is:

```
Ship Motion Predictor → PMP Trajectory Planner → ArduPilot → Quadcopter
         ↑                      ↑
    Deck Sensors           State Feedback
```

Please provide detailed code examples and configuration files where possible. I prefer Python for control scripts and am comfortable with C++ for Gazebo plugins if needed.
