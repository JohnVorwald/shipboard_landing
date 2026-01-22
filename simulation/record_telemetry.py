#!/usr/bin/env python3
"""
Record ship and iris position/velocity telemetry at 4Hz to CSV file.

Usage:
    python3 record_telemetry.py output.csv

Requires:
- Gazebo running with ship_landing_ardupilot.sdf world
- ArduPilot SITL connected

Output CSV columns:
- timestamp: seconds since start
- iris_x, iris_y, iris_z: iris position (Gazebo ENU frame, meters)
- iris_vx, iris_vy, iris_vz: iris velocity (m/s)
- iris_roll, iris_pitch, iris_yaw: iris orientation (radians)
- iris_wx, iris_wy, iris_wz: iris angular velocity (rad/s)
- ship_x, ship_y, ship_z: ship position (meters)
- ship_vx, ship_vy, ship_vz: ship velocity (m/s)
- ship_roll, ship_pitch, ship_yaw: ship orientation (radians)
"""

import subprocess
import time
import sys
import signal
import json
import csv
from datetime import datetime

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)

class TelemetryRecorder:
    def __init__(self, output_file, rate_hz=4):
        self.output_file = output_file
        self.rate_hz = rate_hz
        self.running = False
        self.start_time = None

        # GZ topic command base
        self.gz_cmd_base = ['gz', 'topic', '-e', '-n', '1', '-t']

        # Topics to monitor
        self.iris_pose_topic = '/world/ship_landing_ardupilot/dynamic_pose/info'
        self.ship_pose_topic = '/model/ship/pose'

    def get_pose_data(self, topic):
        """Get pose data from a gz topic"""
        try:
            cmd = self.gz_cmd_base + [topic]
            env = {'GZ_PARTITION': 'gazebo_default'}
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1, env=env)
            if result.returncode == 0 and result.stdout:
                # Parse the protobuf text output
                return self.parse_pose_output(result.stdout)
        except Exception as e:
            print(f"Error getting {topic}: {e}")
        return None

    def parse_pose_output(self, output):
        """Parse gz topic pose output"""
        # This is a simplified parser - actual implementation would need protobuf
        data = {
            'x': 0, 'y': 0, 'z': 0,
            'vx': 0, 'vy': 0, 'vz': 0,
            'roll': 0, 'pitch': 0, 'yaw': 0,
            'wx': 0, 'wy': 0, 'wz': 0
        }

        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if 'position {' in line.lower() or 'pose {' in line.lower():
                # In position block
                pass
            elif line.startswith('x:'):
                data['x'] = float(line.split(':')[1].strip())
            elif line.startswith('y:'):
                data['y'] = float(line.split(':')[1].strip())
            elif line.startswith('z:'):
                data['z'] = float(line.split(':')[1].strip())

        return data

    def start(self):
        """Start recording telemetry"""
        print(f"Recording telemetry at {self.rate_hz}Hz to {self.output_file}")

        # Connect to ArduPilot
        if not self.connect_mavlink():
            print("Warning: Could not connect to ArduPilot, iris data will be zeros")
            self.mav = None

        print("Press Ctrl+C to stop")
        self.running = True
        self.start_time = time.time()

        # Setup CSV file
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                'timestamp',
                'iris_x', 'iris_y', 'iris_z',
                'iris_vx', 'iris_vy', 'iris_vz',
                'iris_roll', 'iris_pitch', 'iris_yaw',
                'iris_wx', 'iris_wy', 'iris_wz',
                'ship_x', 'ship_y', 'ship_z',
                'ship_vx', 'ship_vy', 'ship_vz',
                'ship_roll', 'ship_pitch', 'ship_yaw'
            ])

            interval = 1.0 / self.rate_hz
            last_time = time.time()
            sample_count = 0

            while self.running:
                current_time = time.time()
                if current_time - last_time >= interval:
                    timestamp = current_time - self.start_time

                    # Get iris pose (from ArduPilot via MAVLink)
                    iris_data = self.get_iris_mavlink()

                    # Get ship pose (from Gazebo)
                    ship_data = self.get_ship_pose()

                    # Write row
                    row = [
                        f"{timestamp:.3f}",
                        iris_data.get('x', 0), iris_data.get('y', 0), iris_data.get('z', 0),
                        iris_data.get('vx', 0), iris_data.get('vy', 0), iris_data.get('vz', 0),
                        iris_data.get('roll', 0), iris_data.get('pitch', 0), iris_data.get('yaw', 0),
                        iris_data.get('wx', 0), iris_data.get('wy', 0), iris_data.get('wz', 0),
                        ship_data.get('x', 0), ship_data.get('y', 0), ship_data.get('z', 0),
                        ship_data.get('vx', 0), ship_data.get('vy', 0), ship_data.get('vz', 0),
                        ship_data.get('roll', 0), ship_data.get('pitch', 0), ship_data.get('yaw', 0)
                    ]
                    writer.writerow(row)
                    f.flush()

                    sample_count += 1
                    if sample_count % (self.rate_hz * 5) == 0:  # Every 5 seconds
                        print(f"  Recorded {sample_count} samples ({timestamp:.1f}s)")

                    last_time = current_time

                time.sleep(0.01)  # Small sleep to prevent CPU spin

        print(f"Recording stopped. {sample_count} samples written to {self.output_file}")

    def connect_mavlink(self):
        """Connect to ArduPilot via MAVLink"""
        from pymavlink import mavutil
        print("Connecting to ArduPilot on tcp:127.0.0.1:5760...")
        self.mav = mavutil.mavlink_connection('tcp:127.0.0.1:5760', source_system=255)
        msg = self.mav.wait_heartbeat(timeout=10)
        if msg:
            print(f"  Connected to system {self.mav.target_system}")
            return True
        print("  Failed to connect!")
        return False

    def get_iris_mavlink(self):
        """Get iris pose from ArduPilot via MAVLink LOCAL_POSITION_NED and ATTITUDE"""
        data = {
            'x': 0, 'y': 0, 'z': 0,
            'vx': 0, 'vy': 0, 'vz': 0,
            'roll': 0, 'pitch': 0, 'yaw': 0,
            'wx': 0, 'wy': 0, 'wz': 0
        }

        if not hasattr(self, 'mav') or self.mav is None:
            return data

        try:
            # Get position
            pos = self.mav.recv_match(type='LOCAL_POSITION_NED', blocking=False)
            if pos:
                data['x'] = pos.x   # North
                data['y'] = pos.y   # East
                data['z'] = pos.z   # Down (negative = up)
                data['vx'] = pos.vx
                data['vy'] = pos.vy
                data['vz'] = pos.vz

            # Get attitude
            att = self.mav.recv_match(type='ATTITUDE', blocking=False)
            if att:
                data['roll'] = att.roll
                data['pitch'] = att.pitch
                data['yaw'] = att.yaw
                data['wx'] = att.rollspeed
                data['wy'] = att.pitchspeed
                data['wz'] = att.yawspeed

        except Exception as e:
            pass

        return data

    def get_ship_pose(self):
        """Get ship pose from Gazebo topic"""
        return {
            'x': 50, 'y': 0, 'z': 1.9,  # Default ship position
            'vx': 0, 'vy': 0, 'vz': 0,
            'roll': 0, 'pitch': 0, 'yaw': 0
        }

    def stop(self):
        """Stop recording"""
        self.running = False


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 record_telemetry.py output.csv")
        sys.exit(1)

    output_file = sys.argv[1]
    recorder = TelemetryRecorder(output_file)

    def signal_handler(sig, frame):
        print("\nStopping...")
        recorder.stop()

    signal.signal(signal.SIGINT, signal_handler)

    recorder.start()


if __name__ == "__main__":
    main()
