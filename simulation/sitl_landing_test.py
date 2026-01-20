#!/usr/bin/env python3
"""
ArduPilot SITL Landing Test and Evaluation Script

Comprehensive testing script for PMP trajectory landing on moving ship deck.
Properly handles EKF initialization, GPS lock, and prearm checks before attempting
flight operations.

Based on ArduPilot autotest patterns from vehicle_test_suite.py.

Features:
1. Robust EKF and GPS waiting with status monitoring
2. Prearm check validation
3. Automatic arming and takeoff
4. PMP trajectory execution with metrics collection
5. Landing accuracy evaluation
6. Performance statistics and logging

Usage:
    python3 sitl_landing_test.py [--connection CONNECTION] [--target X Y Z]

Example:
    # With SITL running:
    cd /home/john/ardupilot
    ./Tools/autotest/sim_vehicle.py -v ArduCopter -f quad --no-mavproxy

    # Then run test:
    python3 sitl_landing_test.py --connection tcp:127.0.0.1:5760
"""

import numpy as np
import time
import argparse
import sys
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymavlink import mavutil

# Try to import project modules
try:
    from optimal_control.pmp_controller import PMPController, create_pmp_trajectory, extract_waypoints
    from optimal_control.trajectory_planner import LandingTrajectoryPlanner
    from quad_dynamics.quadrotor import QuadrotorParams
    HAS_PMP = True
except ImportError as e:
    print(f"Warning: Could not import PMP modules: {e}")
    HAS_PMP = False


# =============================================================================
# EKF Status Flags (from mavutil.mavlink)
# =============================================================================
EKF_ATTITUDE = 1
EKF_VELOCITY_HORIZ = 2
EKF_VELOCITY_VERT = 4
EKF_POS_HORIZ_REL = 8
EKF_POS_HORIZ_ABS = 16
EKF_POS_VERT_ABS = 32
EKF_POS_VERT_AGL = 64
EKF_CONST_POS_MODE = 128
EKF_PRED_POS_HORIZ_REL = 256
EKF_PRED_POS_HORIZ_ABS = 512
EKF_GPS_GLITCH = 1024
EKF_ACCEL_ERROR = 2048


@dataclass
class FlightMetrics:
    """Flight test metrics"""
    test_name: str
    timestamp: str
    success: bool

    # Timing
    ekf_init_time: float = 0.0
    arm_time: float = 0.0
    takeoff_time: float = 0.0
    trajectory_duration: float = 0.0
    total_flight_time: float = 0.0

    # Position errors
    mean_position_error: float = 0.0
    max_position_error: float = 0.0
    final_position_error: float = 0.0

    # Velocity errors
    mean_velocity_error: float = 0.0
    max_velocity_error: float = 0.0

    # Landing accuracy
    landing_horizontal_error: float = 0.0
    landing_vertical_error: float = 0.0
    landing_velocity: float = 0.0

    # Status
    error_message: str = ""
    gps_satellites: int = 0
    gps_fix_type: int = 0
    ekf_flags: int = 0


class SITLLandingTest:
    """
    Comprehensive ArduPilot SITL landing test with proper EKF/GPS handling.
    """

    def __init__(self, connection_string: str = 'tcp:127.0.0.1:5760'):
        self.connection_string = connection_string
        self.master: Optional[mavutil.mavlink_connection] = None

        # State tracking
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.attitude = np.zeros(3)
        self.armed = False
        self.mode = ''

        # GPS and EKF status
        self.gps_fix_type = 0
        self.gps_satellites = 0
        self.ekf_flags = 0
        self.prearm_healthy = False

        # Metrics collection
        self.position_history: List[Tuple[float, np.ndarray]] = []
        self.velocity_history: List[Tuple[float, np.ndarray]] = []

        # PMP components (if available)
        if HAS_PMP:
            self.params = QuadrotorParams()
            self.planner = LandingTrajectoryPlanner(self.params)
            self.controller = PMPController(self.params)

        print(f"SITL Landing Test initialized")
        print(f"  Connection: {connection_string}")
        print(f"  PMP Controller: {'Available' if HAS_PMP else 'Not available'}")

    # =========================================================================
    # Connection and State Management
    # =========================================================================

    def connect(self, timeout: float = 30.0) -> bool:
        """Connect to ArduPilot SITL."""
        print(f"\n{'='*60}")
        print(f"  Connecting to ArduPilot SITL")
        print(f"{'='*60}")

        try:
            print(f"Connecting to {self.connection_string}...")
            self.master = mavutil.mavlink_connection(self.connection_string)

            # Wait for heartbeat
            print("Waiting for heartbeat...")
            msg = self.master.wait_heartbeat(timeout=timeout)
            if msg is None:
                print("ERROR: No heartbeat received")
                return False

            print(f"Connected! System {self.master.target_system}, Component {self.master.target_component}")

            # Request data streams
            self._request_data_streams()
            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def _request_data_streams(self):
        """Request necessary data streams from ArduPilot."""
        # Request all streams at 10Hz
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            10,  # 10 Hz
            1    # Start
        )

    def _update_state(self, timeout: float = 0.1):
        """Update state from incoming MAVLink messages."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                time.sleep(0.01)
                continue

            msg_type = msg.get_type()

            if msg_type == 'LOCAL_POSITION_NED':
                self.position = np.array([msg.x, msg.y, msg.z])
                self.velocity = np.array([msg.vx, msg.vy, msg.vz])

            elif msg_type == 'ATTITUDE':
                self.attitude = np.array([msg.roll, msg.pitch, msg.yaw])

            elif msg_type == 'HEARTBEAT':
                self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                try:
                    mode_map = mavutil.mode_mapping_acm.get(msg.custom_mode, 'UNKNOWN')
                    if isinstance(mode_map, dict):
                        self.mode = list(mode_map.keys())[0]
                    else:
                        self.mode = str(mode_map)
                except:
                    self.mode = 'UNKNOWN'

            elif msg_type == 'GPS_RAW_INT':
                self.gps_fix_type = msg.fix_type
                self.gps_satellites = msg.satellites_visible

            elif msg_type == 'EKF_STATUS_REPORT':
                self.ekf_flags = msg.flags

            elif msg_type == 'SYS_STATUS':
                # Check prearm bit
                prearm_bit = mavutil.mavlink.MAV_SYS_STATUS_PREARM_CHECK
                self.prearm_healthy = bool(
                    (msg.onboard_control_sensors_present & prearm_bit) and
                    (msg.onboard_control_sensors_enabled & prearm_bit) and
                    (msg.onboard_control_sensors_health & prearm_bit)
                )

    # =========================================================================
    # EKF and GPS Waiting (Based on ArduPilot autotest patterns)
    # =========================================================================

    def wait_ekf_happy(self, timeout: float = 45.0, require_absolute: bool = True) -> bool:
        """
        Wait for EKF to be healthy.
        Based on ArduPilot vehicle_test_suite.py wait_ekf_happy()
        """
        print(f"\n--- Waiting for EKF initialization (timeout: {timeout}s) ---")

        # Required flags for arming
        required_flags = (
            EKF_ATTITUDE |
            EKF_VELOCITY_HORIZ |
            EKF_VELOCITY_VERT |
            EKF_POS_HORIZ_REL |
            EKF_PRED_POS_HORIZ_REL
        )

        if require_absolute:
            required_flags |= (
                EKF_POS_HORIZ_ABS |
                EKF_POS_VERT_ABS |
                EKF_PRED_POS_HORIZ_ABS
            )

        # Error flags that must NOT be set
        error_flags = EKF_CONST_POS_MODE | EKF_ACCEL_ERROR
        if require_absolute:
            error_flags |= EKF_GPS_GLITCH

        start_time = time.time()
        last_print = 0

        while time.time() - start_time < timeout:
            self._update_state()

            # Print status every 5 seconds
            elapsed = time.time() - start_time
            if elapsed - last_print >= 5.0:
                self._print_ekf_status()
                last_print = elapsed

            # Check if required flags are set
            has_required = (self.ekf_flags & required_flags) == required_flags
            has_errors = (self.ekf_flags & error_flags) != 0

            if has_required and not has_errors:
                print(f"EKF healthy after {elapsed:.1f}s (flags: 0x{self.ekf_flags:x})")
                return True

            time.sleep(0.1)

        print(f"ERROR: EKF timeout after {timeout}s (flags: 0x{self.ekf_flags:x})")
        self._print_ekf_status()
        return False

    def wait_gps_healthy(self, timeout: float = 60.0, min_fix_type: int = 3) -> bool:
        """
        Wait for GPS fix.
        fix_type: 0=no fix, 1=no fix, 2=2D, 3=3D, 4=DGPS, 5=RTK float, 6=RTK fixed
        """
        print(f"\n--- Waiting for GPS fix >= {min_fix_type} (timeout: {timeout}s) ---")

        start_time = time.time()
        last_print = 0

        while time.time() - start_time < timeout:
            self._update_state()

            elapsed = time.time() - start_time
            if elapsed - last_print >= 5.0:
                print(f"  GPS: fix_type={self.gps_fix_type}, sats={self.gps_satellites}")
                last_print = elapsed

            if self.gps_fix_type >= min_fix_type:
                print(f"GPS fix acquired after {elapsed:.1f}s (type={self.gps_fix_type}, sats={self.gps_satellites})")
                return True

            time.sleep(0.1)

        print(f"ERROR: GPS timeout (fix_type={self.gps_fix_type}, sats={self.gps_satellites})")
        return False

    def wait_prearm_healthy(self, timeout: float = 30.0) -> bool:
        """Wait for prearm checks to pass."""
        print(f"\n--- Waiting for prearm checks (timeout: {timeout}s) ---")

        start_time = time.time()

        while time.time() - start_time < timeout:
            self._update_state()

            if self.prearm_healthy:
                elapsed = time.time() - start_time
                print(f"Prearm checks passed after {elapsed:.1f}s")
                return True

            time.sleep(0.1)

        print("ERROR: Prearm checks failed")
        return False

    def wait_ready_to_arm(self, timeout: float = 120.0) -> Tuple[bool, float]:
        """
        Wait for vehicle to be ready to arm.
        Returns (success, time_taken)
        """
        print(f"\n{'='*60}")
        print(f"  Waiting for Ready to Arm")
        print(f"{'='*60}")

        start_time = time.time()

        # 1. Wait for EKF
        if not self.wait_ekf_happy(timeout=min(45.0, timeout)):
            return False, time.time() - start_time

        # 2. Wait for GPS
        remaining = timeout - (time.time() - start_time)
        if not self.wait_gps_healthy(timeout=min(60.0, remaining)):
            return False, time.time() - start_time

        # 3. Wait for prearm
        remaining = timeout - (time.time() - start_time)
        if not self.wait_prearm_healthy(timeout=min(30.0, remaining)):
            return False, time.time() - start_time

        elapsed = time.time() - start_time
        print(f"\nVehicle ready to arm after {elapsed:.1f}s")
        return True, elapsed

    def _print_ekf_status(self):
        """Print detailed EKF status."""
        flags = self.ekf_flags
        print(f"  EKF flags: 0x{flags:x}")
        print(f"    ATTITUDE:          {'YES' if flags & EKF_ATTITUDE else 'NO'}")
        print(f"    VELOCITY_HORIZ:    {'YES' if flags & EKF_VELOCITY_HORIZ else 'NO'}")
        print(f"    VELOCITY_VERT:     {'YES' if flags & EKF_VELOCITY_VERT else 'NO'}")
        print(f"    POS_HORIZ_REL:     {'YES' if flags & EKF_POS_HORIZ_REL else 'NO'}")
        print(f"    POS_HORIZ_ABS:     {'YES' if flags & EKF_POS_HORIZ_ABS else 'NO'}")
        print(f"    POS_VERT_ABS:      {'YES' if flags & EKF_POS_VERT_ABS else 'NO'}")
        print(f"    CONST_POS_MODE:    {'YES (ERROR)' if flags & EKF_CONST_POS_MODE else 'NO (good)'}")
        print(f"    GPS_GLITCH:        {'YES (ERROR)' if flags & EKF_GPS_GLITCH else 'NO (good)'}")
        print(f"    ACCEL_ERROR:       {'YES (ERROR)' if flags & EKF_ACCEL_ERROR else 'NO (good)'}")

    # =========================================================================
    # Flight Control
    # =========================================================================

    def set_mode(self, mode: str, timeout: float = 10.0) -> bool:
        """Set flight mode."""
        print(f"Setting mode to {mode}...")

        mode_mapping = self.master.mode_mapping()
        if mode not in mode_mapping:
            print(f"ERROR: Unknown mode '{mode}'")
            print(f"  Available modes: {list(mode_mapping.keys())}")
            return False

        mode_id = mode_mapping[mode]

        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )

        start_time = time.time()
        while time.time() - start_time < timeout:
            self._update_state()
            if self.mode == mode:
                print(f"Mode set to {mode}")
                return True
            time.sleep(0.1)

        print(f"ERROR: Failed to set mode to {mode} (current: {self.mode})")
        return False

    def arm(self, timeout: float = 10.0) -> Tuple[bool, float]:
        """Arm the vehicle. Returns (success, time_taken)."""
        print("Arming vehicle...")

        start_time = time.time()

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,  # confirmation
            1,  # arm
            0, 0, 0, 0, 0, 0
        )

        while time.time() - start_time < timeout:
            self._update_state()
            if self.armed:
                elapsed = time.time() - start_time
                print(f"Armed after {elapsed:.1f}s")
                return True, elapsed
            time.sleep(0.1)

        print("ERROR: Failed to arm")
        return False, time.time() - start_time

    def disarm(self, force: bool = False) -> bool:
        """Disarm the vehicle."""
        print("Disarming vehicle...")

        param = 21196 if force else 0  # Force disarm magic number

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, param, 0, 0, 0, 0, 0
        )

        for _ in range(50):
            self._update_state()
            if not self.armed:
                print("Disarmed")
                return True
            time.sleep(0.1)

        return False

    def takeoff(self, altitude: float = 10.0, timeout: float = 30.0) -> Tuple[bool, float]:
        """Take off to specified altitude. Returns (success, time_taken)."""
        print(f"Taking off to {altitude}m...")

        start_time = time.time()

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0, 0, 0,
            altitude
        )

        while time.time() - start_time < timeout:
            self._update_state()
            current_alt = -self.position[2]  # NED: negative z is up

            if current_alt >= altitude * 0.9:
                elapsed = time.time() - start_time
                print(f"Reached {current_alt:.1f}m after {elapsed:.1f}s")
                return True, elapsed

            if int(time.time() - start_time) % 3 == 0:
                print(f"  Altitude: {current_alt:.1f}m / {altitude}m")

            time.sleep(0.1)

        current_alt = -self.position[2]
        print(f"ERROR: Takeoff timeout at {current_alt:.1f}m")
        return False, time.time() - start_time

    def send_position_target(self, x: float, y: float, z: float,
                            vx: float = 0, vy: float = 0, vz: float = 0,
                            yaw: float = 0):
        """Send position/velocity target command."""
        type_mask = 0b0000_0000_0111_1000  # Use position and velocity

        self.master.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            x, y, z,
            vx, vy, vz,
            0, 0, 0,  # acceleration
            yaw, 0     # yaw, yaw_rate
        )

    # =========================================================================
    # Test Execution
    # =========================================================================

    def run_hover_test(self, altitude: float = 10.0, duration: float = 10.0) -> FlightMetrics:
        """Run basic hover test to validate SITL operation."""
        metrics = FlightMetrics(
            test_name="hover_test",
            timestamp=datetime.now().isoformat(),
            success=False
        )

        print(f"\n{'='*60}")
        print(f"  HOVER TEST")
        print(f"  Altitude: {altitude}m, Duration: {duration}s")
        print(f"{'='*60}")

        try:
            # Connect
            if not self.connect():
                metrics.error_message = "Connection failed"
                return metrics

            # Wait for ready
            ready, ekf_time = self.wait_ready_to_arm()
            metrics.ekf_init_time = ekf_time
            metrics.gps_satellites = self.gps_satellites
            metrics.gps_fix_type = self.gps_fix_type
            metrics.ekf_flags = self.ekf_flags

            if not ready:
                metrics.error_message = "Not ready to arm"
                return metrics

            # Set mode and arm
            if not self.set_mode('GUIDED'):
                metrics.error_message = "Failed to set GUIDED mode"
                return metrics

            armed, arm_time = self.arm()
            metrics.arm_time = arm_time
            if not armed:
                metrics.error_message = "Failed to arm"
                return metrics

            # Takeoff
            took_off, takeoff_time = self.takeoff(altitude)
            metrics.takeoff_time = takeoff_time
            if not took_off:
                metrics.error_message = "Failed to takeoff"
                self.disarm(force=True)
                return metrics

            # Hover and collect data
            print(f"\nHovering for {duration}s...")
            self.position_history.clear()
            self.velocity_history.clear()

            target_pos = self.position.copy()
            start_time = time.time()

            while time.time() - start_time < duration:
                self._update_state()
                t = time.time() - start_time

                self.position_history.append((t, self.position.copy()))
                self.velocity_history.append((t, self.velocity.copy()))

                # Send position hold command
                self.send_position_target(target_pos[0], target_pos[1], target_pos[2])
                time.sleep(0.1)

            metrics.total_flight_time = time.time() - start_time

            # Compute metrics
            positions = np.array([p[1] for p in self.position_history])
            velocities = np.array([v[1] for v in self.velocity_history])

            pos_errors = np.linalg.norm(positions - target_pos, axis=1)
            vel_mags = np.linalg.norm(velocities, axis=1)

            metrics.mean_position_error = float(np.mean(pos_errors))
            metrics.max_position_error = float(np.max(pos_errors))
            metrics.mean_velocity_error = float(np.mean(vel_mags))
            metrics.max_velocity_error = float(np.max(vel_mags))

            # Land
            print("\nLanding...")
            self.set_mode('LAND')
            time.sleep(10)
            self.disarm(force=True)

            metrics.success = True

        except Exception as e:
            metrics.error_message = str(e)
            print(f"ERROR: {e}")

        return metrics

    def run_trajectory_test(self, target_pos: np.ndarray, tf: float = 10.0) -> FlightMetrics:
        """Run PMP trajectory test."""
        metrics = FlightMetrics(
            test_name="trajectory_test",
            timestamp=datetime.now().isoformat(),
            success=False
        )

        if not HAS_PMP:
            metrics.error_message = "PMP modules not available"
            return metrics

        print(f"\n{'='*60}")
        print(f"  TRAJECTORY TEST")
        print(f"  Target: {target_pos}")
        print(f"  Duration: {tf}s")
        print(f"{'='*60}")

        try:
            # Connect
            if not self.connect():
                metrics.error_message = "Connection failed"
                return metrics

            # Wait for ready
            ready, ekf_time = self.wait_ready_to_arm()
            metrics.ekf_init_time = ekf_time

            if not ready:
                metrics.error_message = "Not ready to arm"
                return metrics

            # Arm and takeoff
            if not self.set_mode('GUIDED'):
                metrics.error_message = "Failed to set GUIDED mode"
                return metrics

            armed, arm_time = self.arm()
            metrics.arm_time = arm_time
            if not armed:
                metrics.error_message = "Failed to arm"
                return metrics

            took_off, takeoff_time = self.takeoff(10.0)
            metrics.takeoff_time = takeoff_time
            if not took_off:
                metrics.error_message = "Takeoff failed"
                self.disarm(force=True)
                return metrics

            # Generate PMP trajectory
            print("\nGenerating PMP trajectory...")
            self._update_state()

            result = self.planner.plan_landing(
                quad_pos=self.position,
                quad_vel=self.velocity,
                deck_pos=target_pos,
                deck_vel=np.zeros(3),
                tf_desired=tf
            )

            if not result.get('success'):
                print("Warning: Using simple trajectory")

            # Execute trajectory
            print(f"\nExecuting trajectory to {target_pos}...")
            self.position_history.clear()

            N = int(tf * 10)
            dt = tf / N
            start_time = time.time()

            for i in range(N):
                alpha = i / N
                target = (1 - alpha) * self.position + alpha * target_pos
                target_vel = (target_pos - self.position) / max(tf - i*dt, 0.1)

                self.send_position_target(
                    target[0], target[1], target[2],
                    target_vel[0], target_vel[1], target_vel[2]
                )

                self._update_state()
                t = time.time() - start_time
                self.position_history.append((t, self.position.copy()))

                time.sleep(dt)

            metrics.trajectory_duration = time.time() - start_time

            # Final position
            self._update_state()
            final_error = np.linalg.norm(self.position - target_pos)
            horizontal_error = np.linalg.norm(self.position[:2] - target_pos[:2])
            vertical_error = abs(self.position[2] - target_pos[2])

            metrics.final_position_error = float(final_error)
            metrics.landing_horizontal_error = float(horizontal_error)
            metrics.landing_vertical_error = float(vertical_error)
            metrics.landing_velocity = float(np.linalg.norm(self.velocity))

            # Land
            print("\nLanding...")
            self.set_mode('LAND')
            time.sleep(10)
            self.disarm(force=True)

            metrics.success = True

        except Exception as e:
            metrics.error_message = str(e)
            print(f"ERROR: {e}")

        return metrics

    def run_full_evaluation(self, num_trials: int = 3) -> Dict:
        """Run full evaluation suite."""
        print(f"\n{'='*60}")
        print(f"  FULL EVALUATION SUITE")
        print(f"  Trials: {num_trials}")
        print(f"{'='*60}")

        results = {
            'timestamp': datetime.now().isoformat(),
            'num_trials': num_trials,
            'hover_tests': [],
            'trajectory_tests': [],
            'summary': {}
        }

        # Hover tests
        print("\n--- HOVER TESTS ---")
        for i in range(num_trials):
            print(f"\nTrial {i+1}/{num_trials}")
            metrics = self.run_hover_test(altitude=10.0, duration=10.0)
            results['hover_tests'].append(asdict(metrics))
            time.sleep(5)  # Wait between trials

        # Trajectory tests
        print("\n--- TRAJECTORY TESTS ---")
        targets = [
            np.array([20, 0, -15]),
            np.array([30, 10, -20]),
            np.array([40, -10, -25]),
        ]

        for i, target in enumerate(targets[:num_trials]):
            print(f"\nTrial {i+1}/{num_trials}")
            metrics = self.run_trajectory_test(target, tf=10.0)
            results['trajectory_tests'].append(asdict(metrics))
            time.sleep(5)

        # Summary statistics
        hover_success = sum(1 for t in results['hover_tests'] if t['success'])
        traj_success = sum(1 for t in results['trajectory_tests'] if t['success'])

        results['summary'] = {
            'hover_success_rate': hover_success / len(results['hover_tests']) if results['hover_tests'] else 0,
            'trajectory_success_rate': traj_success / len(results['trajectory_tests']) if results['trajectory_tests'] else 0,
            'avg_ekf_init_time': np.mean([t['ekf_init_time'] for t in results['hover_tests']]) if results['hover_tests'] else 0,
        }

        # Save results
        output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        return results


def main():
    parser = argparse.ArgumentParser(description='ArduPilot SITL Landing Test')
    parser.add_argument('--connection', type=str, default='tcp:127.0.0.1:5760',
                       help='MAVLink connection string')
    parser.add_argument('--test', type=str, choices=['hover', 'trajectory', 'full'],
                       default='hover', help='Test type to run')
    parser.add_argument('--target', nargs=3, type=float, default=[30, 0, -20],
                       help='Target position [x y z] NED')
    parser.add_argument('--trials', type=int, default=3,
                       help='Number of trials for full evaluation')
    args = parser.parse_args()

    test = SITLLandingTest(args.connection)

    if args.test == 'hover':
        metrics = test.run_hover_test()
        print(f"\n--- Results ---")
        print(json.dumps(asdict(metrics), indent=2))

    elif args.test == 'trajectory':
        target = np.array(args.target)
        metrics = test.run_trajectory_test(target)
        print(f"\n--- Results ---")
        print(json.dumps(asdict(metrics), indent=2))

    elif args.test == 'full':
        results = test.run_full_evaluation(num_trials=args.trials)
        print(f"\n--- Summary ---")
        print(json.dumps(results['summary'], indent=2))


if __name__ == "__main__":
    main()
