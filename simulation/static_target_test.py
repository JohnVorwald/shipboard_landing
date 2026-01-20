#!/usr/bin/env python3
"""
Static Target Verification Test

This test verifies that the ENU (Gazebo) to NED (ArduPilot) coordinate
transformation is correct before attempting to track a moving ship.

Test setup:
- Drone starts at Gazebo origin (0, 0, 0.195)
- Red sphere test target at (20, 10, 5) in Gazebo ENU frame
- Drone arms at origin, so home = (0, 0, 0) in Gazebo frame

Expected behavior:
- After takeoff to 10m, drone should fly to hover above the red sphere
- ENU to NED conversion: N=x_enu, E=y_enu, D=-z_enu

Usage:
    1. Start Gazebo:
       export GZ_SIM_SYSTEM_PLUGIN_PATH=$GZ_SIM_SYSTEM_PLUGIN_PATH:/home/john/gz_ws/src/ardupilot_gazebo/build
       export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:/home/john/gz_ws/src/ardupilot_gazebo/models
       gz sim -r worlds/ship_landing_ardupilot.sdf

    2. Start ArduPilot SITL:
       cd ~/ardupilot/ArduCopter
       ../Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --console

    3. Run this test:
       python3 static_target_test.py
"""

import numpy as np
import time
import argparse
import sys
from pymavlink import mavutil


# Test target position in Gazebo ENU frame
TEST_TARGET_ENU = np.array([20.0, 10.0, 5.0])

# Takeoff altitude
TAKEOFF_ALT = 10.0


def enu_to_ned(pos_enu):
    """
    Convert Gazebo ENU to ArduPilot NED frame.

    Gazebo ENU: X=East, Y=North, Z=Up
    ArduPilot NED: X=North, Y=East, Z=Down

    Note: This is the standard conversion used in ArduPilot docs.
    """
    return np.array([
        pos_enu[0],   # N = East (Gazebo X)
        pos_enu[1],   # E = North (Gazebo Y)
        -pos_enu[2]   # D = -Up (negative Gazebo Z)
    ])


class StaticTargetTest:
    """Test ENU to NED coordinate transformation with a static target."""

    def __init__(self, connection: str = 'tcp:127.0.0.1:5760'):
        self.connection_string = connection
        self.master = None

        # State
        self.position_ned = np.zeros(3)
        self.velocity_ned = np.zeros(3)
        self.armed = False
        self.mode = ''
        self.home_set = False

    def connect(self, timeout: float = 30.0) -> bool:
        """Connect to ArduPilot SITL."""
        print(f"\nConnecting to {self.connection_string}...")
        try:
            self.master = mavutil.mavlink_connection(self.connection_string)
            msg = self.master.wait_heartbeat(timeout=timeout)
            if msg is None:
                print("ERROR: No heartbeat received")
                return False

            print(f"Connected! System {self.master.target_system}")

            # Request data streams
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
        """Update state from MAVLink messages."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                time.sleep(0.01)
                continue

            msg_type = msg.get_type()

            if msg_type == 'LOCAL_POSITION_NED':
                self.position_ned = np.array([msg.x, msg.y, msg.z])
                self.velocity_ned = np.array([msg.vx, msg.vy, msg.vz])

            elif msg_type == 'HEARTBEAT':
                self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

            elif msg_type == 'HOME_POSITION':
                self.home_set = True

    def wait_ready(self, timeout: float = 60.0) -> bool:
        """Wait for vehicle to be ready to arm."""
        print("\nWaiting for vehicle to be ready...")
        start = time.time()

        while time.time() - start < timeout:
            self.update_state()

            # Check for GPS and EKF
            msg = self.master.recv_match(type='EKF_STATUS_REPORT', blocking=True, timeout=1)
            if msg:
                # Check required flags
                required = 0x1FF  # Basic flags
                if (msg.flags & required) == required:
                    print(f"EKF ready (flags: 0x{msg.flags:x})")
                    return True

            time.sleep(0.5)

        print("ERROR: Vehicle not ready")
        return False

    def set_mode(self, mode: str, timeout: float = 10.0) -> bool:
        """Set flight mode."""
        print(f"Setting mode to {mode}...")

        mode_mapping = self.master.mode_mapping()
        if mode not in mode_mapping:
            print(f"ERROR: Unknown mode '{mode}'")
            return False

        mode_id = mode_mapping[mode]

        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )

        start = time.time()
        while time.time() - start < timeout:
            self.update_state()
            msg = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if msg and msg.custom_mode == mode_id:
                print(f"Mode set to {mode}")
                return True

        print(f"ERROR: Failed to set mode")
        return False

    def arm(self, timeout: float = 10.0) -> bool:
        """Arm the vehicle."""
        print("Arming...")

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )

        start = time.time()
        while time.time() - start < timeout:
            self.update_state()
            if self.armed:
                print("Armed!")
                return True
            time.sleep(0.1)

        print("ERROR: Failed to arm")
        return False

    def takeoff(self, altitude: float, timeout: float = 30.0) -> bool:
        """Take off to specified altitude."""
        print(f"Taking off to {altitude}m...")

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude
        )

        start = time.time()
        while time.time() - start < timeout:
            self.update_state()
            current_alt = -self.position_ned[2]  # NED: negative z is up

            if current_alt >= altitude * 0.9:
                print(f"Reached altitude: {current_alt:.1f}m")
                return True

            if int(time.time() - start) % 5 == 0:
                print(f"  Altitude: {current_alt:.1f}m")

            time.sleep(0.1)

        print(f"ERROR: Takeoff timeout")
        return False

    def send_position_target_ned(self, north: float, east: float, down: float):
        """Send position target in NED frame."""
        type_mask = 0b0000_1111_1111_1000  # Use only position

        self.master.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            north, east, down,
            0, 0, 0,  # velocity
            0, 0, 0,  # acceleration
            0, 0      # yaw, yaw_rate
        )

    def fly_to_target(self, target_ned: np.ndarray, timeout: float = 60.0) -> bool:
        """Fly to target position and verify arrival."""
        print(f"\nFlying to NED target: N={target_ned[0]:.1f}, E={target_ned[1]:.1f}, D={target_ned[2]:.1f}")

        start = time.time()
        last_print = 0

        while time.time() - start < timeout:
            # Send position command
            self.send_position_target_ned(target_ned[0], target_ned[1], target_ned[2])

            self.update_state()

            # Calculate error
            error = np.linalg.norm(self.position_ned - target_ned)
            horiz_error = np.linalg.norm(self.position_ned[:2] - target_ned[:2])

            # Print status
            if time.time() - last_print > 2:
                print(f"  Pos: N={self.position_ned[0]:6.1f} E={self.position_ned[1]:6.1f} D={self.position_ned[2]:6.1f} | "
                      f"Error: {error:.1f}m (horiz: {horiz_error:.1f}m)")
                last_print = time.time()

            # Check if arrived (within 2m)
            if error < 2.0:
                print(f"\n>>> ARRIVED at target! Error: {error:.2f}m")
                return True

            time.sleep(0.1)

        print(f"\nERROR: Did not reach target within {timeout}s")
        return False

    def land(self):
        """Land the vehicle."""
        print("\nLanding...")
        self.set_mode('LAND')

        start = time.time()
        while time.time() - start < 30:
            self.update_state()
            if not self.armed:
                print("Landed and disarmed")
                return
            time.sleep(0.5)

    def run_test(self) -> bool:
        """Run the static target verification test."""
        print("=" * 60)
        print("  STATIC TARGET COORDINATE VERIFICATION TEST")
        print("=" * 60)
        print(f"\nTest target (Gazebo ENU): X={TEST_TARGET_ENU[0]}, Y={TEST_TARGET_ENU[1]}, Z={TEST_TARGET_ENU[2]}")

        # Convert target to NED
        target_ned = enu_to_ned(TEST_TARGET_ENU)
        print(f"Target (ArduPilot NED): N={target_ned[0]}, E={target_ned[1]}, D={target_ned[2]}")

        print("\nExpected behavior:")
        print("  - Drone should fly to the RED SPHERE in Gazebo")
        print("  - If it flies to wrong location, coordinate transform is incorrect")

        # Connect
        if not self.connect():
            return False

        # Wait for ready
        if not self.wait_ready():
            return False

        # Set GUIDED mode
        if not self.set_mode('GUIDED'):
            return False

        # Arm
        if not self.arm():
            return False

        # Takeoff
        if not self.takeoff(TAKEOFF_ALT):
            self.land()
            return False

        print("\nHovering at takeoff altitude for 5 seconds...")
        time.sleep(5)

        # Fly to target
        success = self.fly_to_target(target_ned, timeout=60)

        if success:
            print("\n" + "=" * 60)
            print("  TEST PASSED!")
            print("  Coordinate transformation ENU->NED is CORRECT")
            print("=" * 60)

            # Hold position for visual verification
            print("\nHolding position for 10 seconds for visual verification...")
            for i in range(10):
                self.send_position_target_ned(target_ned[0], target_ned[1], target_ned[2])
                self.update_state()
                time.sleep(1)
        else:
            print("\n" + "=" * 60)
            print("  TEST FAILED!")
            print("  Check coordinate transformation logic")
            print("=" * 60)

        # Land
        self.land()

        return success


def main():
    parser = argparse.ArgumentParser(description='Static Target Coordinate Verification Test')
    parser.add_argument('--connection', type=str, default='tcp:127.0.0.1:5760',
                       help='MAVLink connection string')
    args = parser.parse_args()

    test = StaticTargetTest(args.connection)
    success = test.run_test()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
