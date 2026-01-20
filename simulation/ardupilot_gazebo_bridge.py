#!/usr/bin/env python3
"""
ArduPilot-Gazebo Bridge

Bridges ArduPilot SITL state to Gazebo for visualization.
Reads position/attitude from ArduPilot via MAVLink and publishes
to Gazebo to update the quadcopter model pose.

Usage:
    1. Start Gazebo with ship world:
       gz sim worlds/ship_simple.world

    2. Start ArduPilot SITL:
       cd /home/john/ardupilot
       ./Tools/autotest/sim_vehicle.py -v ArduCopter -f quad --no-mavproxy

    3. Run this bridge:
       python3 ardupilot_gazebo_bridge.py

    4. Run your flight script:
       python3 sitl_landing_test.py --test trajectory
"""

import numpy as np
import time
import subprocess
import json
import argparse
import threading
from dataclasses import dataclass
from typing import Optional
from pymavlink import mavutil


@dataclass
class VehicleState:
    """Vehicle state from ArduPilot"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    timestamp: float = 0.0


class ArduPilotGazeboBridge:
    """
    Bridge between ArduPilot SITL and Gazebo.

    Reads vehicle state from ArduPilot and publishes pose updates to Gazebo.
    """

    def __init__(self,
                 mavlink_connection: str = 'tcp:127.0.0.1:5760',
                 gazebo_model: str = 'quadcopter',
                 gazebo_world: str = 'ship_simple',
                 update_rate: float = 50.0,
                 offset: tuple = (0, 0, 0)):
        """
        Initialize bridge.

        Args:
            mavlink_connection: MAVLink connection string
            gazebo_model: Name of the quadcopter model in Gazebo
            gazebo_world: Name of the Gazebo world
            update_rate: Pose update rate in Hz
            offset: Position offset (x, y, z) to align with ship helipad
        """
        self.mavlink_connection = mavlink_connection
        self.gazebo_model = gazebo_model
        self.gazebo_world = gazebo_world
        self.update_rate = update_rate
        self.offset = np.array(offset)

        self.master: Optional[mavutil.mavlink_connection] = None
        self.state = VehicleState()
        self.running = False

        # Ship helipad position (from ship_simple.world)
        # The helipad is at approximately x=15, y=0, z=1.5 on the ship
        self.helipad_offset = np.array([15.0, 0.0, 1.5])

    def connect_mavlink(self) -> bool:
        """Connect to ArduPilot via MAVLink."""
        print(f"Connecting to ArduPilot at {self.mavlink_connection}...")
        try:
            self.master = mavutil.mavlink_connection(self.mavlink_connection)
            self.master.wait_heartbeat(timeout=10)
            print(f"Connected! System {self.master.target_system}")

            # Request data streams
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                50,  # 50 Hz
                1
            )
            return True
        except Exception as e:
            print(f"MAVLink connection failed: {e}")
            return False

    def check_gazebo_running(self) -> bool:
        """Check if Gazebo is running."""
        try:
            result = subprocess.run(
                ['gz', 'topic', '-l'],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Gazebo check failed: {e}")
            return False

    def update_state_from_mavlink(self):
        """Read state from ArduPilot MAVLink messages."""
        while self.running:
            try:
                msg = self.master.recv_match(blocking=True, timeout=0.1)
                if msg is None:
                    continue

                msg_type = msg.get_type()

                if msg_type == 'LOCAL_POSITION_NED':
                    # ArduPilot NED: x=North, y=East, z=Down
                    # Gazebo ENU: x=East, y=North, z=Up
                    self.state.x = msg.y   # ArduPilot East -> Gazebo X
                    self.state.y = msg.x   # ArduPilot North -> Gazebo Y
                    self.state.z = -msg.z  # ArduPilot Down -> Gazebo Up
                    self.state.vx = msg.vy
                    self.state.vy = msg.vx
                    self.state.vz = -msg.vz
                    self.state.timestamp = time.time()

                elif msg_type == 'ATTITUDE':
                    # Convert NED to ENU attitude
                    self.state.roll = msg.roll
                    self.state.pitch = -msg.pitch
                    self.state.yaw = -msg.yaw + np.pi/2  # Rotate 90 degrees

            except Exception as e:
                if self.running:
                    print(f"MAVLink read error: {e}")

    def publish_to_gazebo(self):
        """Publish pose updates to Gazebo."""
        interval = 1.0 / self.update_rate

        while self.running:
            try:
                # Apply offsets for ship-relative positioning
                x = self.state.x + self.offset[0]
                y = self.state.y + self.offset[1]
                z = self.state.z + self.offset[2]

                # Convert roll/pitch/yaw to quaternion
                qw, qx, qy, qz = self._euler_to_quaternion(
                    self.state.roll, self.state.pitch, self.state.yaw
                )

                # Create pose message for gz service
                pose_msg = {
                    "name": self.gazebo_model,
                    "position": {"x": x, "y": y, "z": z},
                    "orientation": {"w": qw, "x": qx, "y": qy, "z": qz}
                }

                # Use gz service to set model pose
                # This is more reliable than gz topic for pose updates
                cmd = [
                    'gz', 'service', '-s', '/world/ship_simple/set_pose',
                    '--reqtype', 'gz.msgs.Pose',
                    '--reptype', 'gz.msgs.Boolean',
                    '--timeout', '100',
                    '--req', f'name: "{self.gazebo_model}" position: {{x: {x}, y: {y}, z: {z}}} orientation: {{w: {qw}, x: {qx}, y: {qy}, z: {qz}}}'
                ]

                subprocess.run(cmd, capture_output=True, timeout=0.5)

            except subprocess.TimeoutExpired:
                pass
            except Exception as e:
                if self.running:
                    print(f"Gazebo publish error: {e}")

            time.sleep(interval)

    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float):
        """Convert Euler angles to quaternion."""
        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return qw, qx, qy, qz

    def run(self):
        """Run the bridge."""
        print("\n" + "="*60)
        print("  ArduPilot-Gazebo Bridge")
        print("="*60)

        # Connect to ArduPilot
        if not self.connect_mavlink():
            print("Failed to connect to ArduPilot")
            return False

        # Check Gazebo
        if not self.check_gazebo_running():
            print("Warning: Gazebo may not be running")
            print("Start Gazebo with: gz sim worlds/ship_simple.world")

        print(f"\nBridge running:")
        print(f"  MAVLink: {self.mavlink_connection}")
        print(f"  Gazebo model: {self.gazebo_model}")
        print(f"  Update rate: {self.update_rate} Hz")
        print(f"  Position offset: {self.offset}")
        print("\nPress Ctrl+C to stop\n")

        self.running = True

        # Start threads
        mavlink_thread = threading.Thread(target=self.update_state_from_mavlink)
        gazebo_thread = threading.Thread(target=self.publish_to_gazebo)

        mavlink_thread.start()
        gazebo_thread.start()

        # Print status periodically
        try:
            while self.running:
                print(f"\rPos: ({self.state.x:6.2f}, {self.state.y:6.2f}, {self.state.z:6.2f}) "
                      f"Att: ({np.degrees(self.state.roll):5.1f}, {np.degrees(self.state.pitch):5.1f}, {np.degrees(self.state.yaw):5.1f})",
                      end='', flush=True)
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n\nStopping bridge...")

        self.running = False
        mavlink_thread.join(timeout=2)
        gazebo_thread.join(timeout=2)

        print("Bridge stopped")
        return True


class SimpleGazeboBridge:
    """
    Simplified bridge using gz topic pub for pose updates.
    This is faster than the service-based approach.
    """

    def __init__(self,
                 mavlink_connection: str = 'tcp:127.0.0.1:5760',
                 model_name: str = 'quadcopter',
                 world_name: str = 'ship_simple'):
        self.mavlink_connection = mavlink_connection
        self.model_name = model_name
        self.world_name = world_name
        self.master = None
        self.running = False

    def run(self):
        """Run simplified bridge using pose_v topic."""
        print("Connecting to ArduPilot...")
        self.master = mavutil.mavlink_connection(self.mavlink_connection)
        self.master.wait_heartbeat(timeout=10)
        print(f"Connected to system {self.master.target_system}")

        # Request streams
        self.master.mav.request_data_stream_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 50, 1
        )

        self.running = True
        print("Bridge running. Press Ctrl+C to stop.")

        x, y, z = 0, 0, 5
        roll, pitch, yaw = 0, 0, 0

        try:
            while self.running:
                msg = self.master.recv_match(blocking=True, timeout=0.02)
                if msg is None:
                    continue

                msg_type = msg.get_type()

                if msg_type == 'LOCAL_POSITION_NED':
                    # NED to ENU conversion
                    x = msg.y + 15  # Offset to helipad position
                    y = msg.x
                    z = -msg.z + 1.5  # Offset for ship deck height

                elif msg_type == 'ATTITUDE':
                    roll = msg.roll
                    pitch = -msg.pitch
                    yaw = -msg.yaw + np.pi/2

                    # Convert to quaternion
                    cr, sr = np.cos(roll/2), np.sin(roll/2)
                    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
                    cy, sy = np.cos(yaw/2), np.sin(yaw/2)

                    qw = cr*cp*cy + sr*sp*sy
                    qx = sr*cp*cy - cr*sp*sy
                    qy = cr*sp*cy + sr*cp*sy
                    qz = cr*cp*sy - sr*sp*cy

                    # Publish to Gazebo
                    # Use pose_v topic with model ID
                    pose_str = f'pose {{ name: "{self.model_name}" position {{ x: {x} y: {y} z: {z} }} orientation {{ w: {qw} x: {qx} y: {qy} z: {qz} }} }}'

                    subprocess.run(
                        ['gz', 'topic', '-t', f'/world/{self.world_name}/pose/info',
                         '-m', 'gz.msgs.Pose_V', '-p', pose_str],
                        capture_output=True, timeout=0.1
                    )

                    print(f"\rPos: ({x:6.2f}, {y:6.2f}, {z:6.2f})", end='', flush=True)

        except KeyboardInterrupt:
            print("\nStopping...")

        self.running = False


def main():
    parser = argparse.ArgumentParser(description='ArduPilot-Gazebo Bridge')
    parser.add_argument('--connection', type=str, default='tcp:127.0.0.1:5760',
                       help='MAVLink connection string')
    parser.add_argument('--model', type=str, default='quadcopter',
                       help='Gazebo model name')
    parser.add_argument('--world', type=str, default='ship_simple',
                       help='Gazebo world name')
    parser.add_argument('--rate', type=float, default=30.0,
                       help='Update rate in Hz')
    parser.add_argument('--offset', nargs=3, type=float, default=[15, 0, 1.5],
                       help='Position offset (x y z) for ship-relative positioning')
    parser.add_argument('--simple', action='store_true',
                       help='Use simple topic-based bridge')
    args = parser.parse_args()

    if args.simple:
        bridge = SimpleGazeboBridge(
            mavlink_connection=args.connection,
            model_name=args.model,
            world_name=args.world
        )
    else:
        bridge = ArduPilotGazeboBridge(
            mavlink_connection=args.connection,
            gazebo_model=args.model,
            gazebo_world=args.world,
            update_rate=args.rate,
            offset=tuple(args.offset)
        )

    bridge.run()


if __name__ == "__main__":
    main()
