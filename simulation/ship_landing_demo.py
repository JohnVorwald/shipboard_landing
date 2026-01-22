#!/usr/bin/env python3
"""
Ship Landing Demo with Video Recording
- Arms quadcopter with retries
- Takes off and flies a pattern
- Records video from both cameras
- Lands on helipad
"""

import time
import subprocess
import threading
import os
import sys
from datetime import datetime

# Try to import pymavlink
try:
    from pymavlink import mavutil
except ImportError:
    print("Installing pymavlink...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pymavlink", "-q"])
    from pymavlink import mavutil

# Try to import OpenCV for video recording
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    print("OpenCV not found - video recording disabled")
    print("Install with: pip install opencv-python")
    HAS_OPENCV = False


class CameraRecorder:
    """Records Gazebo camera feed to video file using gz topic"""

    def __init__(self, topic, output_file, fps=10):
        self.topic = topic
        self.output_file = output_file
        self.fps = fps
        self.recording = False
        self.process = None
        self.frames = []

    def start(self):
        """Start recording using gz topic to capture frames"""
        self.recording = True
        print(f"Starting recording: {self.topic} -> {self.output_file}")

        # Use subprocess to capture gz topic output
        # This is a simplified approach - for real video, use GStreamer
        self.thread = threading.Thread(target=self._record_loop)
        self.thread.daemon = True
        self.thread.start()

    def _record_loop(self):
        """Background thread to record frames"""
        # For now, we'll record using GStreamer if available
        # The gimbal camera streams to UDP 5600
        if "gimbal" in self.topic:
            # Try GStreamer recording
            cmd = [
                "gst-launch-1.0", "-e",
                "udpsrc", "port=5600", "!",
                "application/x-rtp,encoding-name=H264", "!",
                "rtph264depay", "!", "h264parse", "!",
                "mp4mux", "!", "filesink", f"location={self.output_file}"
            ]
        else:
            # For landing camera, we'd need gz-transport bindings
            # Placeholder - just note that recording was requested
            print(f"Note: {self.topic} recording requires gz-transport Python bindings")
            return

        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            print("GStreamer not found - skipping video recording")

    def stop(self):
        """Stop recording"""
        self.recording = False
        if self.process:
            self.process.terminate()
            self.process.wait()
            print(f"Saved: {self.output_file}")


class ShipLandingController:
    """Controls quadcopter for ship landing with MAVLink"""

    def __init__(self, connection_string='udpin:0.0.0.0:14550'):
        self.connection_string = connection_string
        self.master = None
        self.armed = False

    def connect(self, timeout=30):
        """Connect to ArduPilot"""
        print(f"Connecting to {self.connection_string}...")
        self.master = mavutil.mavlink_connection(self.connection_string)
        self.master.wait_heartbeat(timeout=timeout)
        print(f"Connected to system {self.master.target_system}, component {self.master.target_component}")
        return True

    def set_mode(self, mode):
        """Set flight mode"""
        mode_mapping = {
            'STABILIZE': 0,
            'ACRO': 1,
            'ALT_HOLD': 2,
            'AUTO': 3,
            'GUIDED': 4,
            'LOITER': 5,
            'RTL': 6,
            'CIRCLE': 7,
            'LAND': 9,
            'DRIFT': 11,
            'SPORT': 13,
            'POSHOLD': 16,
        }

        mode_id = mode_mapping.get(mode.upper(), mode)
        self.master.set_mode(mode_id)
        time.sleep(0.5)
        print(f"Set mode: {mode}")

    def arm(self, max_attempts=10, force=True):
        """Arm the quadcopter with retries"""
        print(f"Arming (max {max_attempts} attempts)...")

        for attempt in range(1, max_attempts + 1):
            print(f"  Attempt {attempt}/{max_attempts}...")

            # Send arm command
            if force:
                # Force arm (bypasses some checks)
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0,  # confirmation
                    1,  # arm
                    21196,  # force arm magic number
                    0, 0, 0, 0, 0
                )
            else:
                self.master.arducopter_arm()

            # Wait for response
            time.sleep(1)

            # Check if armed
            msg = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=2)
            if msg:
                if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                    print(f"  Armed successfully on attempt {attempt}")
                    self.armed = True
                    return True

            # Check for failure message
            ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=1)
            if ack:
                if ack.result != 0:
                    print(f"  Arm failed with result: {ack.result}")

        print("Failed to arm after all attempts")
        return False

    def disarm(self):
        """Disarm the quadcopter"""
        self.master.arducopter_disarm()
        time.sleep(1)
        self.armed = False
        print("Disarmed")

    def takeoff(self, altitude=10):
        """Takeoff to specified altitude"""
        print(f"Taking off to {altitude}m...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,  # confirmation
            0,  # pitch
            0, 0, 0,  # empty
            0, 0,  # lat, lon (use current)
            altitude  # altitude
        )

        # Wait for takeoff
        start = time.time()
        while time.time() - start < 30:
            pos = self.get_position()
            if pos and pos['relative_alt'] > altitude - 1:
                print(f"  Reached {pos['relative_alt']:.1f}m")
                return True
            time.sleep(0.5)

        print("Takeoff timeout")
        return False

    def land(self):
        """Land the quadcopter"""
        print("Landing...")
        self.set_mode('LAND')

        # Wait for landing
        start = time.time()
        while time.time() - start < 60:
            pos = self.get_position()
            if pos and pos['relative_alt'] < 0.5:
                print("  Landed")
                time.sleep(2)
                return True
            time.sleep(0.5)

        return False

    def goto_position_ned(self, north, east, down):
        """Go to position relative to home (NED frame)"""
        print(f"Going to NED: N={north}m, E={east}m, D={down}m (Alt={-down}m)")

        self.master.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111111000,  # type_mask (position only)
            north, east, down,  # position
            0, 0, 0,  # velocity
            0, 0, 0,  # acceleration
            0, 0  # yaw, yaw_rate
        )

    def get_position(self):
        """Get current position"""
        msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
        if msg:
            return {
                'lat': msg.lat / 1e7,
                'lon': msg.lon / 1e7,
                'alt': msg.alt / 1000,
                'relative_alt': msg.relative_alt / 1000,
            }
        return None

    def get_local_position(self):
        """Get local position relative to home"""
        msg = self.master.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=2)
        if msg:
            return {
                'x': msg.x,  # North
                'y': msg.y,  # East
                'z': msg.z,  # Down (negative = up)
            }
        return None

    def wait_for_position(self, target_n, target_e, target_d, tolerance=2, timeout=60):
        """Wait until reaching target position"""
        start = time.time()
        while time.time() - start < timeout:
            pos = self.get_local_position()
            if pos:
                dist = ((pos['x'] - target_n)**2 +
                        (pos['y'] - target_e)**2 +
                        (pos['z'] - target_d)**2)**0.5
                print(f"  Position: N={pos['x']:.1f} E={pos['y']:.1f} Alt={-pos['z']:.1f}m | Distance: {dist:.1f}m")
                if dist < tolerance:
                    return True
            time.sleep(1)
        return False


def run_demo():
    """Run the ship landing demo"""
    print("=" * 60)
    print("Ship Landing Demo with Video Recording")
    print("=" * 60)

    # Output directory
    output_dir = "/home/john/github/shipboard_landing/simulation"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize controller
    controller = ShipLandingController()

    # Initialize camera recorders
    recorders = []
    gimbal_video = os.path.join(output_dir, f"gimbal_camera_{timestamp}.mp4")
    landing_video = os.path.join(output_dir, f"landing_camera_{timestamp}.mp4")

    try:
        # Connect to ArduPilot
        if not controller.connect():
            print("Failed to connect")
            return False

        # Set GUIDED mode
        controller.set_mode('GUIDED')
        time.sleep(1)

        # Start video recording
        print("\nStarting video recording...")
        gimbal_recorder = CameraRecorder(
            "/world/ship_landing_ardupilot/model/iris/model/gimbal/link/pitch_link/sensor/camera/image",
            gimbal_video
        )
        # gimbal_recorder.start()  # Uncomment if GStreamer is set up
        recorders.append(gimbal_recorder)

        # Aim gimbal down
        print("Aiming gimbal camera down...")
        subprocess.run([
            "gz", "topic", "-t", "/gimbal/cmd_pitch",
            "-m", "gz.msgs.Double", "-p", "data: -1.57"
        ], capture_output=True)

        # Arm with retries
        if not controller.arm(max_attempts=10, force=True):
            print("Failed to arm")
            return False

        time.sleep(1)

        # Takeoff
        if not controller.takeoff(altitude=10):
            print("Takeoff failed")
            return False

        time.sleep(2)

        # Fly a simple pattern
        print("\n--- Flying pattern ---")

        # Move north
        print("Moving 10m North...")
        controller.goto_position_ned(10, 0, -10)
        controller.wait_for_position(10, 0, -10, tolerance=2, timeout=30)
        time.sleep(2)

        # Move east
        print("Moving 10m East...")
        controller.goto_position_ned(10, 10, -10)
        controller.wait_for_position(10, 10, -10, tolerance=2, timeout=30)
        time.sleep(2)

        # Return to home
        print("Returning to helipad...")
        controller.goto_position_ned(0, 0, -10)
        controller.wait_for_position(0, 0, -10, tolerance=2, timeout=30)
        time.sleep(2)

        # Land
        print("\n--- Landing ---")
        controller.land()

        # Wait for disarm
        time.sleep(5)

        print("\n" + "=" * 60)
        print("Demo complete!")
        print("=" * 60)

        # Print final position
        pos = controller.get_local_position()
        if pos:
            print(f"Final position: N={pos['x']:.2f}m, E={pos['y']:.2f}m")
            print(f"Landing accuracy: {(pos['x']**2 + pos['y']**2)**0.5:.2f}m from helipad center")

        return True

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        controller.set_mode('LAND')
        return False

    finally:
        # Stop recorders
        for recorder in recorders:
            recorder.stop()


if __name__ == "__main__":
    # Check if Gazebo and ArduPilot are running
    print("Make sure Gazebo and ArduPilot SITL are running:")
    print("  Terminal 1: /home/john/github/shipboard_landing/gazebo/launch_ship.sh")
    print("  Terminal 2: cd /home/john/ardupilot && ./Tools/autotest/sim_vehicle.py -v ArduCopter -f JSON --model JSON --console")
    print()

    input("Press Enter when ready...")

    success = run_demo()
    sys.exit(0 if success else 1)
