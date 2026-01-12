#!/usr/bin/env python3
"""
ArduPilot + Gazebo PMP Trajectory Executor

Executes Pontryagin Maximum Principle optimal trajectories on ArduCopter SITL
with Gazebo visualization.

Features:
1. Generates optimal trajectory from current state to target
2. Converts to MAVLink SET_POSITION_TARGET_LOCAL_NED commands
3. Sends trajectory to ArduCopter in Guided mode
4. Visualizes flight in Gazebo

Prerequisites:
- ArduPilot SITL running: sim_vehicle.py -v ArduCopter --no-mavproxy
- Gazebo with ArduPilot plugin (optional for visualization)
- pymavlink installed: pip install pymavlink

Usage:
  python3 ardupilot_pmp_gazebo.py [--target X Y Z] [--tf TIME]

Example:
  python3 ardupilot_pmp_gazebo.py --target 50 10 -20 --tf 10
"""

import numpy as np
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymavlink import mavutil

from optimal_control.pmp_controller import (
    PMPController, PMPTrajectory, create_pmp_trajectory,
    extract_waypoints, trajectory_to_mavlink_messages
)
from optimal_control.trajectory_planner import LandingTrajectoryPlanner
from quad_dynamics.quadrotor import QuadrotorParams


class ArduPilotPMPExecutor:
    """
    Execute PMP trajectories on ArduPilot via MAVLink.
    """

    def __init__(self, connection_string: str = 'udp:127.0.0.1:14550'):
        """
        Initialize MAVLink connection.

        Args:
            connection_string: MAVLink connection (e.g., 'udp:127.0.0.1:14550')
        """
        self.connection_string = connection_string
        self.master = None
        self.params = QuadrotorParams()
        self.planner = LandingTrajectoryPlanner(self.params)
        self.controller = PMPController(self.params)

        # Current state
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.attitude = np.zeros(3)  # roll, pitch, yaw
        self.armed = False
        self.mode = ''

    def connect(self) -> bool:
        """Connect to ArduPilot."""
        print(f"Connecting to {self.connection_string}...")
        try:
            self.master = mavutil.mavlink_connection(self.connection_string)
            self.master.wait_heartbeat(timeout=10)
            print(f"Connected! System {self.master.target_system}, Component {self.master.target_component}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def wait_for_position(self, timeout: float = 10.0) -> bool:
        """Wait for valid position data."""
        print("Waiting for position data...")
        start = time.time()
        while time.time() - start < timeout:
            self._update_state()
            if np.linalg.norm(self.position) > 0.1:
                print(f"Position acquired: {self.position}")
                return True
            time.sleep(0.1)
        print("Timeout waiting for position")
        return False

    def _update_state(self):
        """Update state from MAVLink messages."""
        while True:
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                break

            msg_type = msg.get_type()

            if msg_type == 'LOCAL_POSITION_NED':
                self.position = np.array([msg.x, msg.y, msg.z])
                self.velocity = np.array([msg.vx, msg.vy, msg.vz])

            elif msg_type == 'ATTITUDE':
                self.attitude = np.array([msg.roll, msg.pitch, msg.yaw])

            elif msg_type == 'HEARTBEAT':
                self.armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                mode_map = mavutil.mode_mapping_acm.get(msg.custom_mode, 'UNKNOWN')
                if isinstance(mode_map, dict):
                    self.mode = list(mode_map.keys())[0]
                else:
                    self.mode = str(mode_map)

    def set_mode(self, mode: str) -> bool:
        """Set flight mode."""
        print(f"Setting mode to {mode}...")
        mode_id = self.master.mode_mapping().get(mode)
        if mode_id is None:
            print(f"Unknown mode: {mode}")
            return False

        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )

        # Wait for mode change
        for _ in range(50):
            self._update_state()
            if self.mode == mode:
                print(f"Mode set to {mode}")
                return True
            time.sleep(0.1)

        print(f"Failed to set mode to {mode}")
        return False

    def arm(self) -> bool:
        """Arm the vehicle."""
        print("Arming...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )

        for _ in range(50):
            self._update_state()
            if self.armed:
                print("Armed!")
                return True
            time.sleep(0.1)

        print("Failed to arm")
        return False

    def takeoff(self, altitude: float = 10.0) -> bool:
        """Take off to specified altitude."""
        print(f"Taking off to {altitude}m...")

        # Send takeoff command
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude
        )

        # Wait for altitude
        for _ in range(100):
            self._update_state()
            if abs(self.position[2]) > altitude * 0.9:  # NED: negative z is up
                print(f"Reached altitude: {-self.position[2]:.1f}m")
                return True
            time.sleep(0.1)

        print(f"Takeoff timeout at {-self.position[2]:.1f}m")
        return False

    def generate_pmp_trajectory(self, target_pos: np.ndarray,
                                target_vel: np.ndarray = None,
                                tf: float = None) -> PMPTrajectory:
        """
        Generate optimal PMP trajectory from current state to target.

        Args:
            target_pos: Target position [x, y, z] NED
            target_vel: Target velocity [vx, vy, vz] NED (default: hover)
            tf: Trajectory duration (estimated if not provided)

        Returns:
            PMPTrajectory object
        """
        self._update_state()

        if target_vel is None:
            target_vel = np.zeros(3)

        # Estimate tf if not provided
        if tf is None:
            dist = np.linalg.norm(target_pos - self.position)
            tf = max(5.0, dist / 5.0)  # ~5 m/s average

        print(f"\nGenerating PMP trajectory:")
        print(f"  From: {self.position}")
        print(f"  To: {target_pos}")
        print(f"  Duration: {tf:.1f}s")

        # Plan trajectory using min-snap planner
        result = self.planner.plan_landing(
            quad_pos=self.position,
            quad_vel=self.velocity,
            deck_pos=target_pos,
            deck_vel=target_vel,
            tf_desired=tf
        )

        if not result.get('success'):
            print("Warning: trajectory planning had issues, using simple trajectory")
            # Create simple straight-line trajectory
            N = int(tf * 10)
            t = np.linspace(0, tf, N)
            x = np.zeros((N, 12))
            for i, ti in enumerate(t):
                alpha = ti / tf
                x[i, 0:3] = (1 - alpha) * self.position + alpha * target_pos
                x[i, 3:6] = (target_pos - self.position) / tf  # constant velocity
            u = np.ones((N, 4)) * self.params.mass * 9.81 / 4
            lam = np.zeros((N, 12))
        else:
            # Convert trajectory result to state arrays
            traj_result = result['trajectory']
            N = 50
            t = np.linspace(0, tf, N)
            x = np.zeros((N, 12))
            u = np.zeros((N, 4))
            lam = np.zeros((N, 12))

            for i, ti in enumerate(t):
                sample = self.planner.sample_trajectory(traj_result, ti)
                x[i, 0:3] = sample['position']
                x[i, 3:6] = sample['velocity']
                u[i, 0] = sample['thrust']

        # Create PMP trajectory
        pmp_traj = create_pmp_trajectory(
            x_traj=x, u_traj=u, t_traj=t, tf=tf,
            deck_pos=target_pos,
            deck_vel=target_vel,
            deck_att=np.zeros(3),
            params=self.params
        )

        print(f"  Generated {len(t)} waypoints")
        return pmp_traj

    def send_position_target(self, x: float, y: float, z: float,
                             vx: float = 0, vy: float = 0, vz: float = 0,
                             yaw: float = 0):
        """Send SET_POSITION_TARGET_LOCAL_NED command."""
        # Type mask: use position and velocity
        type_mask = (
            0b0000_0000_0111_1000  # Ignore acceleration, yaw_rate
        )

        self.master.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms (ignored)
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            x, y, z,
            vx, vy, vz,
            0, 0, 0,  # acceleration (ignored)
            yaw, 0    # yaw, yaw_rate
        )

    def execute_trajectory(self, trajectory: PMPTrajectory,
                          update_rate: float = 10.0) -> dict:
        """
        Execute PMP trajectory on ArduCopter.

        Args:
            trajectory: PMPTrajectory to execute
            update_rate: Position command update rate (Hz)

        Returns:
            Execution statistics
        """
        print(f"\nExecuting trajectory (tf={trajectory.tf:.1f}s)...")

        # Ensure we're in GUIDED mode
        if not self.set_mode('GUIDED'):
            return {'success': False, 'error': 'Failed to set GUIDED mode'}

        # Extract waypoints at update rate
        interval = 1.0 / update_rate
        waypoints = extract_waypoints(trajectory, interval=interval, frame='NED')

        print(f"  Sending {len(waypoints)} position commands at {update_rate}Hz")

        # Track execution
        start_time = time.time()
        position_errors = []
        velocity_errors = []

        for i, wp in enumerate(waypoints):
            # Send position command
            pos = wp['position']
            vel = wp['velocity']
            yaw = wp['yaw']

            self.send_position_target(
                pos[0], pos[1], pos[2],
                vel[0], vel[1], vel[2],
                yaw
            )

            # Wait for next update
            target_time = start_time + wp['time']
            while time.time() < target_time:
                self._update_state()
                time.sleep(0.01)

            # Record errors
            pos_error = np.linalg.norm(np.array(pos) - self.position)
            vel_error = np.linalg.norm(np.array(vel) - self.velocity)
            position_errors.append(pos_error)
            velocity_errors.append(vel_error)

            # Progress
            if i % 10 == 0:
                print(f"    t={wp['time']:.1f}s: pos_err={pos_error:.2f}m, vel_err={vel_error:.2f}m/s")

        # Final position
        final_target = waypoints[-1]['position']
        final_error = np.linalg.norm(np.array(final_target) - self.position)

        stats = {
            'success': True,
            'duration': time.time() - start_time,
            'waypoints': len(waypoints),
            'mean_pos_error': np.mean(position_errors),
            'max_pos_error': np.max(position_errors),
            'final_pos_error': final_error,
            'mean_vel_error': np.mean(velocity_errors)
        }

        print(f"\nTrajectory execution complete:")
        print(f"  Duration: {stats['duration']:.1f}s")
        print(f"  Mean position error: {stats['mean_pos_error']:.2f}m")
        print(f"  Final position error: {stats['final_pos_error']:.2f}m")

        return stats

    def run_demo(self, target_pos: np.ndarray, tf: float = None):
        """
        Run full PMP trajectory demo.

        1. Connect to ArduPilot
        2. Arm and takeoff
        3. Generate PMP trajectory
        4. Execute trajectory
        5. Hold final position
        """
        print("\n" + "=" * 60)
        print("  ArduPilot PMP Trajectory Demo")
        print("=" * 60)

        # Connect
        if not self.connect():
            return

        # Wait for position
        if not self.wait_for_position():
            return

        # Set to GUIDED and arm
        if not self.set_mode('GUIDED'):
            return

        if not self.armed:
            if not self.arm():
                return

        # Takeoff if on ground
        if abs(self.position[2]) < 2.0:
            if not self.takeoff(10.0):
                return

        # Generate trajectory
        trajectory = self.generate_pmp_trajectory(target_pos, tf=tf)

        # Execute
        stats = self.execute_trajectory(trajectory)

        # Hold position for a few seconds
        print("\nHolding final position...")
        time.sleep(5.0)

        print("\nDemo complete!")
        return stats


def main():
    parser = argparse.ArgumentParser(description='ArduPilot PMP Trajectory Demo')
    parser.add_argument('--target', nargs=3, type=float, default=[50, 0, -20],
                       help='Target position [x y z] in NED (default: 50 0 -20)')
    parser.add_argument('--tf', type=float, default=None,
                       help='Trajectory duration in seconds (auto if not set)')
    parser.add_argument('--connection', type=str, default='udp:127.0.0.1:14550',
                       help='MAVLink connection string')
    args = parser.parse_args()

    target = np.array(args.target)
    print(f"Target position: {target} (NED)")

    executor = ArduPilotPMPExecutor(args.connection)
    executor.run_demo(target, tf=args.tf)


if __name__ == "__main__":
    main()
