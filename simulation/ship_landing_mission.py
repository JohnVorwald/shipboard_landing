#!/usr/bin/env python3
"""
Ship Landing Mission - Complete Autonomous Landing Simulation with Multiple Guidance Controllers

This script performs a complete autonomous shipboard landing with support for
evaluating different guidance/control algorithms:

Supported Controllers:
- waypoint: Simple waypoint-following (baseline)
- pmp: Pontryagin Maximum Principle optimal control
- tau: Tau guidance law (time-to-contact based)
- zem: Zero-Effort-Miss guidance
- mpc: Variable horizon Model Predictive Control

Flight Profile:
1. Starts Gazebo with the ship simulation
2. Starts ArduPilot SITL
3. Positions quadcopter 20ft aft and 20ft above ship
4. Flies level to descent intercept point
5. Descends at user-defined angle (default 60°) to hover point above deck
6. Vertical descent to touchdown
7. Lands and disarms

Usage:
    python3 ship_landing_mission.py [options]

Examples:
    # Run with PMP controller
    python3 ship_landing_mission.py --controller pmp

    # Run with Tau guidance at 45° descent
    python3 ship_landing_mission.py --controller tau --descent-angle 45

    # Compare controllers
    python3 ship_landing_mission.py --compare pmp tau waypoint

Author: Autonomous Landing Project
"""

import numpy as np
import time
import subprocess
import signal
import sys
import os
import json
import argparse
import atexit
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, List, Dict
from datetime import datetime

# Global mission reference for cleanup on exit
_active_mission = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymavlink import mavutil

# Try to import guidance modules
try:
    from optimal_control.pmp_controller import PMPController
    from optimal_control.trajectory_planner import LandingTrajectoryPlanner
    from quad_dynamics.quadrotor import QuadrotorParams
    HAS_PMP = True
except ImportError:
    HAS_PMP = False
    print("Warning: PMP modules not available")

try:
    from guidance.tau_guidance import TauGuidance
    HAS_TAU = True
except ImportError:
    HAS_TAU = False

try:
    from guidance.zem_guidance import ZEMGuidance
    HAS_ZEM = True
except ImportError:
    HAS_ZEM = False

# =============================================================================
# Constants
# =============================================================================
FT_TO_M = 0.3048
DEG_TO_RAD = np.pi / 180

# Ship/helipad configuration
HELIPAD_X = 15.0
HELIPAD_Y = 0.0
DECK_HEIGHT = 1.5

# EKF flags
EKF_ATTITUDE = 0x001
EKF_VELOCITY_HORIZ = 0x002
EKF_VELOCITY_VERT = 0x004
EKF_POS_HORIZ_REL = 0x008
EKF_POS_HORIZ_ABS = 0x010
EKF_POS_VERT_ABS = 0x020
EKF_CONST_POS_MODE = 0x080
EKF_GPS_GLITCH = 0x400
EKF_ACCEL_ERROR = 0x800


@dataclass
class MissionConfig:
    """Mission configuration parameters"""
    descent_angle_deg: float = 60.0
    hover_height_m: float = 1.0
    start_height_ft: float = 20.0
    start_aft_ft: float = 20.0
    approach_speed: float = 3.0
    descent_speed: float = 2.0
    vertical_speed: float = 0.5
    position_tolerance: float = 0.5
    controller: str = 'waypoint'
    skip_gazebo: bool = False
    skip_sitl: bool = False
    headless: bool = True            # Run Gazebo in headless mode (no GUI)
    # Ship motion parameters
    ship_speed_kt: float = 3.0        # Ship speed in knots
    ship_heading_deg: float = 0.0     # Ship heading (0 = North/+X)


# Conversion constants
KT_TO_MS = 0.514444  # knots to m/s


@dataclass
class ShipState:
    """Ship state for tracking moving deck"""
    initial_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    heading: float = 0.0  # radians
    start_time: float = 0.0
    _current_position: np.ndarray = field(default=None)

    def __post_init__(self):
        """Initialize current position from initial position."""
        if self._current_position is None:
            self._current_position = self.initial_position.copy()

    def update(self, t: float):
        """Update ship position based on elapsed time from mission start."""
        elapsed = t - self.start_time
        self._current_position = self.initial_position + self.velocity * elapsed

    @property
    def position(self) -> np.ndarray:
        return self._current_position

    def get_deck_position(self, helipad_offset: np.ndarray) -> np.ndarray:
        """Get helipad position in world frame."""
        # Rotate helipad offset by ship heading
        c, s = np.cos(self.heading), np.sin(self.heading)
        rotated_offset = np.array([
            c * helipad_offset[0] - s * helipad_offset[1],
            s * helipad_offset[0] + c * helipad_offset[1],
            helipad_offset[2]
        ])
        return self._current_position + rotated_offset


@dataclass
class FlightMetrics:
    """Flight performance metrics"""
    controller_name: str = ""
    timestamp: str = ""
    success: bool = False

    # Timing
    total_flight_time: float = 0.0
    approach_time: float = 0.0
    descent_time: float = 0.0
    landing_time: float = 0.0

    # Position tracking
    mean_position_error: float = 0.0
    max_position_error: float = 0.0
    final_position_error: float = 0.0

    # Velocity
    mean_velocity: float = 0.0
    max_velocity: float = 0.0
    landing_velocity: float = 0.0

    # Energy (proxy: integrated acceleration)
    control_effort: float = 0.0

    # Smoothness (jerk)
    mean_jerk: float = 0.0
    max_jerk: float = 0.0

    # Touchdown accuracy
    touchdown_x_error: float = 0.0
    touchdown_y_error: float = 0.0

    error_message: str = ""


@dataclass
class VehicleState:
    """Vehicle state"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude: np.ndarray = field(default_factory=lambda: np.zeros(3))
    time: float = 0.0


# =============================================================================
# Guidance Controller Base Class
# =============================================================================
class GuidanceController(ABC):
    """Base class for guidance controllers"""

    def __init__(self, name: str):
        self.name = name
        self.target_position = np.zeros(3)
        self.target_velocity = np.zeros(3)

    @abstractmethod
    def compute_command(self, state: VehicleState, target_pos: np.ndarray,
                       target_vel: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute position and velocity commands.

        Args:
            state: Current vehicle state
            target_pos: Target position (NED)
            target_vel: Target velocity (NED)
            dt: Time step

        Returns:
            (position_cmd, velocity_cmd) in NED frame
        """
        pass

    def reset(self):
        """Reset controller state"""
        pass


class WaypointController(GuidanceController):
    """Simple waypoint following controller (baseline)"""

    def __init__(self, max_speed: float = 10.0, gain: float = 4.0):
        super().__init__("waypoint")
        self.max_speed = max_speed  # m/s - needs to be faster than ship + closure rate
        self.gain = gain  # Higher gain = more aggressive pursuit

    def compute_command(self, state: VehicleState, target_pos: np.ndarray,
                       target_vel: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        error = target_pos - state.position
        distance = np.linalg.norm(error)

        if distance < 0.01:
            return target_pos, target_vel

        direction = error / distance
        # Proportional control with minimum speed for pursuit
        pursuit_speed = max(2.0, min(self.max_speed, distance * self.gain))
        velocity_cmd = direction * pursuit_speed + target_vel

        return target_pos, velocity_cmd


class PMPGuidanceController(GuidanceController):
    """Pontryagin Maximum Principle optimal guidance"""

    def __init__(self):
        super().__init__("pmp")
        if HAS_PMP:
            self.params = QuadrotorParams()
            self.planner = LandingTrajectoryPlanner(self.params)
            self.controller = PMPController(self.params)
            self.trajectory = None
            self.trajectory_start_time = None

    def plan_trajectory(self, start_pos: np.ndarray, start_vel: np.ndarray,
                       target_pos: np.ndarray, target_vel: np.ndarray, tf: float):
        """Plan optimal trajectory"""
        if not HAS_PMP:
            return False

        result = self.planner.plan_landing(
            quad_pos=start_pos,
            quad_vel=start_vel,
            deck_pos=target_pos,
            deck_vel=target_vel,
            tf_desired=tf
        )

        if result.get('success'):
            self.trajectory = result['trajectory']
            self.trajectory_start_time = time.time()
            return True
        return False

    def compute_command(self, state: VehicleState, target_pos: np.ndarray,
                       target_vel: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        if not HAS_PMP or self.trajectory is None:
            # Fallback to waypoint
            error = target_pos - state.position
            distance = np.linalg.norm(error)
            if distance < 0.01:
                return target_pos, target_vel
            direction = error / distance
            speed = min(3.0, distance)
            return target_pos, direction * speed + target_vel

        # Sample trajectory at current time
        t = time.time() - self.trajectory_start_time
        sample = self.planner.sample_trajectory(self.trajectory, t)

        pos_cmd = np.array(sample['position'])
        vel_cmd = np.array(sample['velocity'])

        return pos_cmd, vel_cmd

    def reset(self):
        self.trajectory = None
        self.trajectory_start_time = None


class TauGuidanceController(GuidanceController):
    """Tau guidance (time-to-contact based)"""

    def __init__(self, tau_dot: float = -0.5, k_tau: float = 1.0):
        super().__init__("tau")
        self.tau_dot = tau_dot  # Desired rate of change of tau
        self.k_tau = k_tau      # Tau control gain

    def compute_command(self, state: VehicleState, target_pos: np.ndarray,
                       target_vel: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tau guidance computes velocity to maintain constant tau-dot.
        tau = range / range_rate
        tau_dot = d(tau)/dt
        """
        error = target_pos - state.position
        range_vec = error
        range_mag = np.linalg.norm(range_vec)

        if range_mag < 0.1:
            return target_pos, target_vel

        # Current range rate (closing velocity)
        rel_vel = target_vel - state.velocity
        range_rate = np.dot(rel_vel, range_vec) / range_mag

        # Current tau
        if abs(range_rate) < 0.01:
            tau = 100.0  # Large tau if not closing
        else:
            tau = -range_mag / range_rate

        # Desired range rate for constant tau_dot
        # tau_dot = (range_rate^2 - range * range_accel) / range_rate^2
        # For constant tau_dot, range_rate = -range / (tau_dot * t + tau_0)
        # Simplified: range_rate_desired = -range / tau_desired
        tau_desired = max(1.0, tau + self.tau_dot * dt)
        range_rate_desired = -range_mag / tau_desired

        # Velocity command
        direction = range_vec / range_mag
        speed = -range_rate_desired
        velocity_cmd = direction * speed + target_vel

        # Limit speed
        max_speed = 5.0
        if np.linalg.norm(velocity_cmd) > max_speed:
            velocity_cmd = velocity_cmd / np.linalg.norm(velocity_cmd) * max_speed

        return target_pos, velocity_cmd


class ZEMGuidanceController(GuidanceController):
    """Zero-Effort-Miss guidance"""

    def __init__(self, nav_gain: float = 3.0):
        super().__init__("zem")
        self.nav_gain = nav_gain  # Navigation constant (typically 3-5)

    def compute_command(self, state: VehicleState, target_pos: np.ndarray,
                       target_vel: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        ZEM guidance: a = N * V_c * omega
        where omega is the line-of-sight rate
        """
        # Range vector (target - current)
        r = target_pos - state.position
        range_mag = np.linalg.norm(r)

        if range_mag < 0.1:
            return target_pos, target_vel

        # Relative velocity
        v_rel = state.velocity - target_vel
        closing_speed = -np.dot(v_rel, r) / range_mag

        # Time to go
        if closing_speed < 0.1:
            t_go = range_mag / 1.0  # Assume 1 m/s if not closing
        else:
            t_go = range_mag / closing_speed

        # Zero-effort-miss
        zem = r + v_rel * t_go

        # Commanded acceleration
        if t_go > 0.1:
            a_cmd = self.nav_gain * zem / (t_go * t_go)
        else:
            a_cmd = np.zeros(3)

        # Integrate to get velocity command (simple Euler)
        velocity_cmd = state.velocity + a_cmd * dt

        # Limit velocity
        max_speed = 5.0
        if np.linalg.norm(velocity_cmd) > max_speed:
            velocity_cmd = velocity_cmd / np.linalg.norm(velocity_cmd) * max_speed

        return target_pos, velocity_cmd


class VariableHorizonMPCController(GuidanceController):
    """Variable Horizon MPC (simplified)"""

    def __init__(self, horizon: float = 5.0):
        super().__init__("mpc")
        self.horizon = horizon
        self.min_horizon = 1.0

    def compute_command(self, state: VehicleState, target_pos: np.ndarray,
                       target_vel: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simplified MPC: plan straight-line trajectory over adaptive horizon.
        """
        error = target_pos - state.position
        distance = np.linalg.norm(error)

        if distance < 0.1:
            return target_pos, target_vel

        # Adaptive horizon based on distance
        horizon = max(self.min_horizon, min(self.horizon, distance / 2.0))

        # Plan velocity to reach target in horizon time
        velocity_cmd = error / horizon + target_vel

        # Limit velocity
        max_speed = 5.0
        if np.linalg.norm(velocity_cmd) > max_speed:
            velocity_cmd = velocity_cmd / np.linalg.norm(velocity_cmd) * max_speed

        return target_pos, velocity_cmd


def create_controller(name: str) -> GuidanceController:
    """Factory function to create controllers"""
    controllers = {
        'waypoint': WaypointController,
        'pmp': PMPGuidanceController if HAS_PMP else WaypointController,
        'tau': TauGuidanceController,
        'zem': ZEMGuidanceController,
        'mpc': VariableHorizonMPCController,
    }

    if name not in controllers:
        print(f"Unknown controller '{name}', using waypoint")
        return WaypointController()

    return controllers[name]()


# =============================================================================
# Mission Controller
# =============================================================================
class ShipLandingMission:
    """Complete ship landing mission with pluggable guidance controllers"""

    def __init__(self, config: MissionConfig):
        self.config = config
        self.master: Optional[mavutil.mavlink_connection] = None
        self.gazebo_proc: Optional[subprocess.Popen] = None
        self.sitl_proc: Optional[subprocess.Popen] = None

        # State
        self.state = VehicleState()
        self.armed = False
        self.mode = ''
        self.gps_fix = 0
        self.ekf_flags = 0

        # Ship state (moving deck)
        ship_speed_ms = config.ship_speed_kt * KT_TO_MS
        ship_heading_rad = config.ship_heading_deg * DEG_TO_RAD
        self.ship = ShipState(
            initial_position=np.array([0.0, 0.0, DECK_HEIGHT]),  # Ship starts at origin
            velocity=np.array([
                ship_speed_ms * np.cos(ship_heading_rad),
                ship_speed_ms * np.sin(ship_heading_rad),
                0.0
            ]),
            heading=ship_heading_rad,
            start_time=0.0
        )
        self.helipad_offset = np.array([HELIPAD_X, HELIPAD_Y, 0.0])  # Offset from ship center

        # Controller
        self.controller = create_controller(config.controller)
        print(f"Using guidance controller: {self.controller.name}")
        print(f"Ship speed: {config.ship_speed_kt} kt ({ship_speed_ms:.2f} m/s)")
        print(f"Ship heading: {config.ship_heading_deg}° (0=North/+X)")

        # Metrics
        self.metrics = FlightMetrics(controller_name=self.controller.name)
        self.flight_path: List[Tuple[float, np.ndarray, np.ndarray]] = []

    def start_gazebo(self) -> bool:
        """Start Gazebo with ship simulation."""
        if self.config.skip_gazebo:
            print("Skipping Gazebo startup")
            return True

        mode_str = "headless" if self.config.headless else "GUI"
        print("\n" + "="*60)
        print(f"  Starting Gazebo Ship Simulation ({mode_str})")
        print("="*60)

        subprocess.run(['pkill', '-9', '-f', 'gz sim'], capture_output=True)
        time.sleep(2)

        env = os.environ.copy()
        env['GZ_SIM_PHYSICS_ENGINE_PATH'] = '/usr/lib/x86_64-linux-gnu/gz-physics-8/engine-plugins'

        world_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'gazebo', 'worlds', 'ship_simple.world'
        )

        if not os.path.exists(world_path):
            print(f"ERROR: World file not found: {world_path}")
            return False

        try:
            # Build command: -s for headless (server only), no flag for GUI
            cmd = ['gz', 'sim']
            if self.config.headless:
                cmd.append('-s')
            cmd.append(world_path)

            self.gazebo_proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            # GUI needs more time to start
            wait_time = 5 if self.config.headless else 10
            print(f"Waiting {wait_time}s for Gazebo...")
            time.sleep(wait_time)
            print("Gazebo started")
            return True
        except Exception as e:
            print(f"ERROR: {e}")
            return False

    def start_sitl(self) -> bool:
        """Start ArduPilot SITL."""
        if self.config.skip_sitl:
            print("Skipping SITL startup")
            return True

        print("\n" + "="*60)
        print("  Starting ArduPilot SITL")
        print("="*60)

        subprocess.run(['pkill', '-9', '-f', 'arducopter'], capture_output=True)
        time.sleep(2)

        ardupilot_dir = '/home/john/ardupilot'

        try:
            self.sitl_proc = subprocess.Popen(
                ['./Tools/autotest/sim_vehicle.py', '-v', 'ArduCopter', '-f', 'quad', '--no-mavproxy'],
                cwd=ardupilot_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("Waiting for SITL (30s)...")
            time.sleep(30)
            print("SITL started")
            return True
        except Exception as e:
            print(f"ERROR: {e}")
            return False

    def connect_mavlink(self, timeout: float = 30.0) -> bool:
        """Connect to ArduPilot via MAVLink."""
        print("\nConnecting to ArduPilot...")

        try:
            self.master = mavutil.mavlink_connection('tcp:127.0.0.1:5760')
            msg = self.master.wait_heartbeat(timeout=timeout)
            if msg is None:
                return False

            print(f"Connected! System {self.master.target_system}")

            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10, 1
            )
            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def update_state(self, timeout: float = 0.1):
        """Update vehicle state."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                time.sleep(0.01)
                continue

            msg_type = msg.get_type()

            if msg_type == 'LOCAL_POSITION_NED':
                old_vel = self.state.velocity.copy()
                self.state.position = np.array([msg.x, msg.y, msg.z])
                self.state.velocity = np.array([msg.vx, msg.vy, msg.vz])
                dt = 0.1
                self.state.acceleration = (self.state.velocity - old_vel) / dt
                self.state.time = time.time()

            elif msg_type == 'ATTITUDE':
                self.state.attitude = np.array([msg.roll, msg.pitch, msg.yaw])

            elif msg_type == 'HEARTBEAT':
                self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                try:
                    self.mode = str(mavutil.mode_mapping_acm.get(msg.custom_mode, 'UNKNOWN'))
                except:
                    self.mode = 'UNKNOWN'

            elif msg_type == 'GPS_RAW_INT':
                self.gps_fix = msg.fix_type

            elif msg_type == 'EKF_STATUS_REPORT':
                self.ekf_flags = msg.flags

    def wait_ready_to_arm(self, timeout: float = 120.0) -> bool:
        """Wait for ready to arm with robust checks."""
        print("\n--- Waiting for EKF/GPS ---")

        required = (EKF_ATTITUDE | EKF_VELOCITY_HORIZ | EKF_VELOCITY_VERT |
                   EKF_POS_HORIZ_REL | EKF_POS_HORIZ_ABS | EKF_POS_VERT_ABS)
        errors = EKF_CONST_POS_MODE | EKF_GPS_GLITCH | EKF_ACCEL_ERROR

        # Must stay ready for at least 2 seconds to avoid stale data
        ready_start = None
        ready_duration = 2.0

        start = time.time()
        last_print = 0

        while time.time() - start < timeout:
            self.update_state(timeout=0.5)  # Longer timeout to get fresh data

            elapsed = time.time() - start

            has_required = (self.ekf_flags & required) == required
            has_errors = (self.ekf_flags & errors) != 0
            has_gps = self.gps_fix >= 3

            if has_required and not has_errors and has_gps:
                if ready_start is None:
                    ready_start = time.time()
                elif time.time() - ready_start >= ready_duration:
                    print(f"Ready after {elapsed:.1f}s (EKF: 0x{self.ekf_flags:x}, GPS: {self.gps_fix})")
                    return True
            else:
                ready_start = None  # Reset if conditions not met

            if elapsed - last_print >= 5:
                print(f"  EKF: 0x{self.ekf_flags:x}, GPS: {self.gps_fix} [{elapsed:.0f}s]")
                last_print = elapsed

            time.sleep(0.1)

        print(f"Timeout! EKF: 0x{self.ekf_flags:x}, GPS: {self.gps_fix}")
        return False

    def set_mode(self, mode: str) -> bool:
        """Set flight mode."""
        mode_id = self.master.mode_mapping().get(mode)
        if mode_id is None:
            return False

        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )

        for _ in range(100):
            self.update_state()
            if self.mode == mode:
                return True
            time.sleep(0.1)
        return False

    def arm(self) -> bool:
        """Arm vehicle using pymavlink helper with force-arm fallback."""
        print("Arming (standard method)...")

        # Try standard arm first
        self.master.arducopter_arm()
        try:
            self.master.motors_armed_wait(timeout=10)
            print("  Armed!")
            self.armed = True
            return True
        except:
            pass

        # If that fails, try force arm
        print("Standard arm failed, trying force arm...")

        # Set ARMING_CHECK=0
        self.master.mav.param_set_send(
            self.master.target_system,
            self.master.target_component,
            b'ARMING_CHECK', 0,
            mavutil.mavlink.MAV_PARAM_TYPE_INT32
        )
        time.sleep(0.5)

        # Send force arm command (21196 = force arm magic number)
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 21196, 0, 0, 0, 0, 0
        )

        # Wait for armed
        for _ in range(30):
            self.update_state()
            if self.armed:
                print("  Force armed!")
                return True
            time.sleep(0.1)

        print("  Arm failed!")
        return False

    def disarm(self, force: bool = False) -> bool:
        """Disarm vehicle."""
        param = 21196 if force else 0
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, param, 0, 0, 0, 0, 0
        )

        for _ in range(50):
            self.update_state()
            if not self.armed:
                return True
            time.sleep(0.1)
        return False

    def takeoff(self, altitude: float) -> bool:
        """Take off with ACK check and progress monitoring."""
        print(f"Sending TAKEOFF command to {altitude:.1f}m...")

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude
        )

        # Wait for command ACK
        ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        if ack:
            if ack.result == 0:
                print(f"  Takeoff ACK: SUCCESS")
            else:
                print(f"  Takeoff ACK: result={ack.result} (0=success)")
        else:
            print("  No takeoff ACK received")

        # Monitor altitude with progress updates
        start_time = time.time()
        last_print = 0
        max_alt = 0

        for _ in range(300):
            self.update_state()
            current_alt = -self.state.position[2]
            max_alt = max(max_alt, current_alt)
            elapsed = time.time() - start_time

            # Progress update every 2 seconds
            if elapsed - last_print >= 2:
                vz = -self.state.velocity[2]
                print(f"  t={elapsed:.1f}s Alt={current_alt:.2f}m Vz={vz:.2f}m/s target={altitude:.1f}m")
                last_print = elapsed

            if current_alt >= altitude * 0.9:
                print(f"  Takeoff complete! Alt={current_alt:.2f}m")
                return True

            time.sleep(0.1)

        print(f"  Takeoff timeout. Max alt reached: {max_alt:.2f}m")
        return False

    def send_command(self, pos: np.ndarray, vel: np.ndarray):
        """Send position/velocity command."""
        type_mask = 0b0000_0000_0111_1000

        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            pos[0], pos[1], pos[2],
            vel[0], vel[1], vel[2],
            0, 0, 0, 0, 0
        )

    def fly_segment_ship_relative(self, ship_offset: np.ndarray, target_vel_offset: np.ndarray,
                                   tolerance: float, segment_name: str) -> bool:
        """
        Fly to ship-relative target using guidance controller.
        Target position is updated in real-time as ship moves.

        Args:
            ship_offset: Offset from helipad in ship frame (x=fwd, y=right, z=up)
            target_vel_offset: Velocity offset from ship velocity
            tolerance: Position tolerance in meters
            segment_name: Name of this flight segment
        """
        print(f"\n--- {segment_name} ---")
        print(f"Ship-relative offset: ({ship_offset[0]:.1f}, {ship_offset[1]:.1f}, {ship_offset[2]:.1f})")

        dt = 0.1
        max_time = 60.0
        start_time = time.time()

        while time.time() - start_time < max_time:
            self.update_state()
            current_time = time.time()

            # Update ship position
            self.ship.update(current_time)

            # Compute world-frame target from ship-relative offset
            # ship_offset is in ship frame: x=forward, y=right, z=up
            # Need to rotate by ship heading and add to helipad position
            c, s = np.cos(self.ship.heading), np.sin(self.ship.heading)
            rotated_offset = np.array([
                c * ship_offset[0] - s * ship_offset[1],
                s * ship_offset[0] + c * ship_offset[1],
                ship_offset[2]
            ])

            # Get helipad world position
            helipad_pos = self.ship.get_deck_position(self.helipad_offset)

            # Target in world frame (NED: z negative is up)
            target_pos = np.array([
                helipad_pos[0] + rotated_offset[0],
                helipad_pos[1] + rotated_offset[1],
                -(helipad_pos[2] + rotated_offset[2])  # Convert to NED
            ])

            # Target velocity includes ship motion
            target_vel = self.ship.velocity.copy()
            target_vel[2] = -target_vel[2]  # Convert to NED

            # Record flight path
            self.flight_path.append((
                self.state.time,
                self.state.position.copy(),
                self.state.velocity.copy()
            ))

            # Check if reached (relative to moving target)
            error = np.linalg.norm(target_pos - self.state.position)
            if error < tolerance:
                print(f"Reached target (error: {error:.2f}m, ship at ({helipad_pos[0]:.1f}, {helipad_pos[1]:.1f}))")
                return True

            # Get command from guidance controller
            pos_cmd, vel_cmd = self.controller.compute_command(
                self.state, target_pos, target_vel, dt
            )

            # Send command
            self.send_command(pos_cmd, vel_cmd)

            # Print progress every 2 seconds
            if int(time.time() - start_time) % 2 == 0 and (time.time() - start_time) % 2 < dt:
                quad_pos = self.state.position
                quad_alt = -quad_pos[2]
                print(f"  Quad: ({quad_pos[0]:.1f}, {quad_pos[1]:.1f}, alt={quad_alt:.1f}m) | Ship: ({helipad_pos[0]:.1f}, {helipad_pos[1]:.1f}) | Err: {error:.1f}m")

            time.sleep(dt)

        print(f"Timeout! Final error: {error:.2f}m")
        return False

    def fly_segment(self, target_pos: np.ndarray, target_vel: np.ndarray,
                   tolerance: float, segment_name: str) -> bool:
        """Fly to fixed target (legacy method)."""
        print(f"\n--- {segment_name} ---")
        print(f"Target: ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")

        dt = 0.1
        max_time = 60.0
        start_time = time.time()

        while time.time() - start_time < max_time:
            self.update_state()

            # Record flight path
            self.flight_path.append((
                self.state.time,
                self.state.position.copy(),
                self.state.velocity.copy()
            ))

            # Check if reached
            error = np.linalg.norm(target_pos - self.state.position)
            if error < tolerance:
                print(f"Reached target (error: {error:.2f}m)")
                return True

            # Get command from guidance controller
            pos_cmd, vel_cmd = self.controller.compute_command(
                self.state, target_pos, target_vel, dt
            )

            # Send command
            self.send_command(pos_cmd, vel_cmd)

            time.sleep(dt)

        print(f"Timeout! Final error: {np.linalg.norm(target_pos - self.state.position):.2f}m")
        return False

    def compute_ship_relative_waypoints(self) -> Tuple[List[Tuple[np.ndarray, np.ndarray, str]], float]:
        """
        Compute ship-relative flight waypoints.

        Returns offsets from helipad in ship frame:
        - x: forward (+) / aft (-)
        - y: right (+) / left (-)
        - z: up (+) from deck

        Returns:
            waypoints: List of (ship_offset, vel_offset, name)
            start_altitude: Altitude for takeoff (absolute NED)
        """
        cfg = self.config

        start_height = cfg.start_height_ft * FT_TO_M
        start_aft = cfg.start_aft_ft * FT_TO_M
        descent_angle = cfg.descent_angle_deg * DEG_TO_RAD
        hover_height = cfg.hover_height_m

        # All offsets are relative to helipad in ship frame
        # Start position: aft and above helipad
        start_offset = np.array([-start_aft, 0.0, start_height])

        # Descent intercept: where 60° descent starts
        # From start_height to hover_height, at descent_angle
        height_to_descend = start_height - hover_height
        horiz_dist = height_to_descend / np.tan(descent_angle)
        intercept_offset = np.array([-horiz_dist, 0.0, start_height])

        # Hover point: directly above helipad at hover_height
        hover_offset = np.array([0.0, 0.0, hover_height])

        # Landing point: on the deck
        land_offset = np.array([0.0, 0.0, 0.0])

        zero_vel = np.zeros(3)

        waypoints = [
            (intercept_offset, zero_vel, f"Level Flight to Descent Intercept ({cfg.descent_angle_deg}° start)"),
            (hover_offset, zero_vel, f"Descend at {cfg.descent_angle_deg}° to Hover ({hover_height}m)"),
            (land_offset, zero_vel, "Vertical Descent to Touchdown"),
        ]

        # Compute initial world position for takeoff
        # Get current helipad position
        helipad_pos = self.ship.get_deck_position(self.helipad_offset)

        # Rotate start offset by ship heading
        c, s = np.cos(self.ship.heading), np.sin(self.ship.heading)
        rotated_start = np.array([
            c * start_offset[0] - s * start_offset[1],
            s * start_offset[0] + c * start_offset[1],
            start_offset[2]
        ])

        start_pos_world = np.array([
            helipad_pos[0] + rotated_start[0],
            helipad_pos[1] + rotated_start[1],
            -(helipad_pos[2] + rotated_start[2])  # NED
        ])

        print("\n" + "="*60)
        print(f"  FLIGHT PLAN ({self.controller.name} controller)")
        print("="*60)
        print(f"  Ship speed: {cfg.ship_speed_kt} kt, heading: {cfg.ship_heading_deg}°")
        print(f"  Descent angle: {cfg.descent_angle_deg}°")
        print(f"  Start: {cfg.start_aft_ft}ft aft, {cfg.start_height_ft}ft above deck")
        print("\n  Ship-relative waypoints (from helipad):")
        for i, (offset, vel, name) in enumerate(waypoints):
            print(f"  WP{i+1}: {name}")
            print(f"        Offset: ({offset[0]:.1f}m fwd, {offset[1]:.1f}m right, {offset[2]:.1f}m up)")

        return waypoints, -start_pos_world[2]

    def compute_waypoints(self) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """Legacy: compute absolute waypoints (for stationary ship)."""
        cfg = self.config

        start_height = cfg.start_height_ft * FT_TO_M
        start_aft = cfg.start_aft_ft * FT_TO_M
        descent_angle = cfg.descent_angle_deg * DEG_TO_RAD
        hover_height = cfg.hover_height_m

        # Start position (NED)
        start_pos = np.array([
            HELIPAD_X - start_aft,
            HELIPAD_Y,
            -(DECK_HEIGHT + start_height)
        ])

        # Landing point
        land_pos = np.array([HELIPAD_X, HELIPAD_Y, -DECK_HEIGHT])

        # Hover point
        hover_pos = np.array([HELIPAD_X, HELIPAD_Y, -(DECK_HEIGHT + hover_height)])

        # Descent intercept
        height_diff = abs(start_pos[2] - hover_pos[2])
        horiz_dist = height_diff / np.tan(descent_angle)
        intercept_pos = np.array([
            HELIPAD_X - horiz_dist,
            HELIPAD_Y,
            start_pos[2]
        ])

        zero_vel = np.zeros(3)

        waypoints = [
            (intercept_pos, zero_vel, f"Level Flight to Intercept ({cfg.descent_angle_deg}° descent start)"),
            (hover_pos, zero_vel, f"Descent to Hover ({hover_height}m above deck)"),
            (land_pos, zero_vel, "Final Descent to Touchdown"),
        ]

        print("\n" + "="*60)
        print(f"  FLIGHT PLAN ({self.controller.name} controller)")
        print("="*60)
        print(f"  Descent angle: {cfg.descent_angle_deg}°")
        print(f"  Start: {cfg.start_aft_ft}ft aft, {cfg.start_height_ft}ft above")
        for i, (pos, vel, name) in enumerate(waypoints):
            alt = -pos[2] - DECK_HEIGHT
            print(f"  WP{i+1}: {name}")
            print(f"        Pos: ({pos[0]:.1f}, {pos[1]:.1f}) Alt: {alt:.1f}m above deck")

        return waypoints, -start_pos[2]

    def compute_metrics(self) -> FlightMetrics:
        """Compute flight metrics relative to moving ship."""
        if not self.flight_path:
            return self.metrics

        times = np.array([p[0] for p in self.flight_path])
        positions = np.array([p[1] for p in self.flight_path])
        velocities = np.array([p[2] for p in self.flight_path])

        self.metrics.total_flight_time = times[-1] - times[0]

        # Velocity metrics
        vel_mags = np.linalg.norm(velocities, axis=1)
        self.metrics.mean_velocity = float(np.mean(vel_mags))
        self.metrics.max_velocity = float(np.max(vel_mags))
        self.metrics.landing_velocity = float(vel_mags[-1])

        # Touchdown accuracy - relative to ship position at landing time
        final_pos = positions[-1]
        landing_time = times[-1]
        self.ship.update(landing_time)
        ship_helipad = self.ship.get_deck_position(self.helipad_offset)
        target_pos = np.array([ship_helipad[0], ship_helipad[1], -ship_helipad[2]])
        self.metrics.touchdown_x_error = float(abs(final_pos[0] - target_pos[0]))
        self.metrics.touchdown_y_error = float(abs(final_pos[1] - target_pos[1]))
        self.metrics.final_position_error = float(np.linalg.norm(final_pos - target_pos))

        # Control effort (integrated squared acceleration)
        if len(velocities) > 2:
            dt = np.diff(times)
            dv = np.diff(velocities, axis=0)
            accels = dv / dt[:, np.newaxis]
            # Use dt[:-1] which has same length as accels
            self.metrics.control_effort = float(np.sum(np.sum(accels**2, axis=1) * dt[:len(accels)]))

        return self.metrics

    def run_mission(self) -> FlightMetrics:
        """Execute landing mission."""
        self.metrics.timestamp = datetime.now().isoformat()

        print("\n" + "="*60)
        print(f"  SHIP LANDING MISSION ({self.controller.name})")
        print("="*60)

        try:
            if not self.start_gazebo():
                self.metrics.error_message = "Gazebo failed"
                return self.metrics

            if not self.start_sitl():
                self.metrics.error_message = "SITL failed"
                return self.metrics

            if not self.connect_mavlink():
                self.metrics.error_message = "MAVLink failed"
                return self.metrics

            if not self.wait_ready_to_arm():
                self.metrics.error_message = "Not ready to arm"
                return self.metrics

            # Initialize ship time reference
            self.ship.start_time = time.time()

            # Compute ship-relative waypoints
            waypoints, takeoff_alt = self.compute_ship_relative_waypoints()

            if not self.set_mode('GUIDED'):
                self.metrics.error_message = "Mode failed"
                return self.metrics

            if not self.arm():
                self.metrics.error_message = "Arm failed"
                return self.metrics

            print(f"\nTaking off to {takeoff_alt:.1f}m...")
            if not self.takeoff(takeoff_alt):
                self.metrics.error_message = "Takeoff failed"
                self.disarm(force=True)
                return self.metrics

            # Execute ship-relative flight segments
            print("\n--- Beginning Ship-Relative Approach ---")
            print(f"Ship moving at {self.config.ship_speed_kt} kt, tracking helipad...")

            for ship_offset, vel_offset, name in waypoints:
                # Use larger tolerance (2.0m) for moving ship tracking
                if not self.fly_segment_ship_relative(ship_offset, vel_offset, 2.0, name):
                    self.metrics.error_message = f"Failed: {name}"
                    self.disarm(force=True)
                    return self.metrics
                time.sleep(1)

            print("\n--- Landing Complete ---")
            time.sleep(2)
            self.disarm()

            self.metrics.success = True
            self.compute_metrics()

            print("\n" + "="*60)
            print("  MISSION SUCCESS!")
            print("="*60)

        except KeyboardInterrupt:
            print("\nAborted")
            self.disarm(force=True)
            self.metrics.error_message = "User abort"

        return self.metrics

    def cleanup(self):
        """
        Thoroughly cleanup all spawned processes and terminals.
        This ensures no orphaned processes remain after the script exits.
        """
        print("\n--- Cleaning up all processes ---")

        # Kill SITL processes
        sitl_patterns = [
            'arducopter',
            'sim_vehicle.py',
            'mavproxy',
            'ArduCopter',
        ]
        for pattern in sitl_patterns:
            subprocess.run(['pkill', '-9', '-f', pattern],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Terminate SITL subprocess if we have a handle
        if self.sitl_proc:
            try:
                self.sitl_proc.terminate()
                self.sitl_proc.wait(timeout=2)
            except:
                try:
                    self.sitl_proc.kill()
                except:
                    pass

        # Kill Gazebo processes
        gazebo_patterns = [
            'gz sim',
            'gz-sim',
            'gzserver',
            'gzclient',
            'ruby.*gz',  # Gazebo transport
        ]
        for pattern in gazebo_patterns:
            subprocess.run(['pkill', '-9', '-f', pattern],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Terminate Gazebo subprocess if we have a handle
        if self.gazebo_proc:
            try:
                self.gazebo_proc.terminate()
                self.gazebo_proc.wait(timeout=2)
            except:
                try:
                    self.gazebo_proc.kill()
                except:
                    pass

        # Kill any remaining related processes
        subprocess.run(['pkill', '-9', '-f', 'waf'],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Small delay to ensure processes are fully terminated
        time.sleep(1)

        # Verify cleanup
        remaining = subprocess.run(['pgrep', '-f', 'arducopter|gz sim'],
                                   capture_output=True, text=True)
        if remaining.stdout.strip():
            print("Warning: Some processes may still be running")
            print(f"  PIDs: {remaining.stdout.strip()}")
        else:
            print("All processes cleaned up successfully")


def compare_controllers(controllers: List[str], config: MissionConfig) -> Dict:
    """Compare multiple controllers."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': asdict(config),
        'controllers': {}
    }

    for ctrl_name in controllers:
        print(f"\n{'#'*60}")
        print(f"  Testing Controller: {ctrl_name}")
        print(f"{'#'*60}")

        config.controller = ctrl_name
        mission = ShipLandingMission(config)

        try:
            metrics = mission.run_mission()
            results['controllers'][ctrl_name] = asdict(metrics)
        finally:
            mission.cleanup()
            time.sleep(5)

    # Print comparison
    print("\n" + "="*60)
    print("  CONTROLLER COMPARISON")
    print("="*60)
    print(f"{'Controller':<12} {'Success':<8} {'Time (s)':<10} {'Error (m)':<10} {'Effort':<10}")
    print("-"*60)

    for name, m in results['controllers'].items():
        print(f"{name:<12} {str(m['success']):<8} {m['total_flight_time']:<10.1f} "
              f"{m['final_position_error']:<10.3f} {m['control_effort']:<10.1f}")

    # Save results
    output_file = f"controller_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


def cleanup_on_exit():
    """Cleanup handler called on exit."""
    global _active_mission
    if _active_mission is not None:
        print("\n--- Exit cleanup triggered ---")
        _active_mission.cleanup()
        _active_mission = None


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    global _active_mission
    print(f"\n\nReceived signal {signum}, cleaning up...")
    if _active_mission is not None:
        _active_mission.cleanup()
        _active_mission = None
    sys.exit(1)


def main():
    global _active_mission

    parser = argparse.ArgumentParser(description='Ship Landing Mission')
    parser.add_argument('--controller', type=str, default='waypoint',
                       choices=['waypoint', 'pmp', 'tau', 'zem', 'mpc'],
                       help='Guidance controller to use')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple controllers')
    parser.add_argument('--descent-angle', type=float, default=60.0,
                       help='Descent angle in degrees (default: 60)')
    parser.add_argument('--hover-height', type=float, default=1.0,
                       help='Hover height above deck in meters (default: 1.0)')
    parser.add_argument('--start-height-ft', type=float, default=20.0,
                       help='Start height above ship in feet (default: 20)')
    parser.add_argument('--start-aft-ft', type=float, default=20.0,
                       help='Start distance aft of helipad in feet (default: 20)')
    parser.add_argument('--ship-speed', type=float, default=3.0,
                       help='Ship speed in knots (default: 3.0)')
    parser.add_argument('--ship-heading', type=float, default=0.0,
                       help='Ship heading in degrees (0=North/+X, default: 0)')
    parser.add_argument('--skip-gazebo', action='store_true',
                       help='Skip Gazebo startup')
    parser.add_argument('--skip-sitl', action='store_true',
                       help='Skip SITL startup')
    parser.add_argument('--gui', action='store_true',
                       help='Run Gazebo with GUI (default: headless)')
    args = parser.parse_args()

    config = MissionConfig(
        descent_angle_deg=args.descent_angle,
        hover_height_m=args.hover_height,
        start_height_ft=args.start_height_ft,
        start_aft_ft=args.start_aft_ft,
        controller=args.controller,
        skip_gazebo=args.skip_gazebo,
        skip_sitl=args.skip_sitl,
        headless=not args.gui,
        ship_speed_kt=args.ship_speed,
        ship_heading_deg=args.ship_heading,
    )

    # Register cleanup handlers
    atexit.register(cleanup_on_exit)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if args.compare:
            compare_controllers(args.compare, config)
        else:
            mission = ShipLandingMission(config)
            _active_mission = mission
            try:
                metrics = mission.run_mission()
                print(f"\n--- Results ---")
                print(json.dumps(asdict(metrics), indent=2))
            finally:
                mission.cleanup()
                _active_mission = None
    except Exception as e:
        print(f"\nFatal error: {e}")
        if _active_mission is not None:
            _active_mission.cleanup()
            _active_mission = None
        raise


if __name__ == "__main__":
    main()
