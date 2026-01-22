#!/usr/bin/env python3
"""
Headless landing test - monitors quad and ship positions,
evaluates landing success automatically.
"""

import subprocess
import time
import sys
import os
import signal
import json
import re

# Gazebo environment setup
ENV = os.environ.copy()
ENV.update({
    'GZ_IP': '127.0.0.1',
    'GZ_PARTITION': 'gazebo_default',
    'GZ_SIM_PHYSICS_ENGINE_PATH': '/usr/lib/x86_64-linux-gnu/gz-physics-8/engine-plugins',
})

WORLD_FILE = os.path.join(os.path.dirname(__file__), 'worlds/ship_simple.world')

# Landing criteria
HELIPAD_OFFSET = (15.0, 0.0, 2.2)  # Helipad position relative to ship
LANDING_TOLERANCE_XY = 3.0  # meters - within helipad radius
LANDING_TOLERANCE_Z = 0.5   # meters - on the deck
MAX_SIM_TIME = 30.0  # seconds


def get_model_pose(model_name):
    """Get model pose using gz model command."""
    try:
        result = subprocess.run(
            ['gz', 'model', '-m', model_name, '-p'],
            capture_output=True, text=True, timeout=2, env=ENV
        )
        if result.returncode == 0:
            # Parse pose from output (format varies by gz version)
            output = result.stdout.strip()
            # Try to extract x, y, z from output
            # Format: "x y z roll pitch yaw" or similar
            numbers = re.findall(r'[-+]?\d*\.?\d+', output)
            if len(numbers) >= 3:
                return float(numbers[0]), float(numbers[1]), float(numbers[2])
    except (subprocess.TimeoutExpired, Exception) as e:
        pass
    return None


def get_poses_from_topic():
    """Get poses by echoing the pose topic."""
    try:
        result = subprocess.run(
            ['gz', 'topic', '-e', '-t', '/world/ship_simple_world/pose/info', '-n', '1'],
            capture_output=True, text=True, timeout=3, env=ENV
        )
        if result.returncode == 0:
            output = result.stdout
            poses = {}

            # Parse protobuf text format for poses
            # Look for name and position blocks
            current_name = None
            for line in output.split('\n'):
                if 'name:' in line:
                    match = re.search(r'name:\s*"([^"]+)"', line)
                    if match:
                        current_name = match.group(1)
                elif current_name and 'position' in line.lower():
                    # Next few lines have x, y, z
                    pass
                elif current_name:
                    for coord in ['x:', 'y:', 'z:']:
                        if coord in line:
                            val = re.search(r'[-+]?\d*\.?\d+', line)
                            if val:
                                if current_name not in poses:
                                    poses[current_name] = {}
                                poses[current_name][coord[0]] = float(val.group())

            return poses
    except Exception as e:
        print(f"Topic error: {e}")
    return {}


def check_landing(quad_pos, ship_pos):
    """Check if quad has landed on ship helipad."""
    if quad_pos is None or ship_pos is None:
        return False, "No position data"

    # Calculate helipad world position
    helipad_x = ship_pos[0] + HELIPAD_OFFSET[0]
    helipad_y = ship_pos[1] + HELIPAD_OFFSET[1]
    helipad_z = ship_pos[2] + HELIPAD_OFFSET[2]

    # Check distance from helipad
    dx = quad_pos[0] - helipad_x
    dy = quad_pos[1] - helipad_y
    dz = quad_pos[2] - helipad_z

    dist_xy = (dx**2 + dy**2)**0.5

    status = f"Quad: ({quad_pos[0]:.2f}, {quad_pos[1]:.2f}, {quad_pos[2]:.2f}) | "
    status += f"Helipad: ({helipad_x:.2f}, {helipad_y:.2f}, {helipad_z:.2f}) | "
    status += f"Offset: XY={dist_xy:.2f}m Z={dz:.2f}m"

    if dist_xy <= LANDING_TOLERANCE_XY and abs(dz) <= LANDING_TOLERANCE_Z:
        return True, status

    return False, status


def run_test():
    """Run headless landing test."""
    print("=" * 60)
    print("Shipboard Landing Test (Headless)")
    print("=" * 60)
    print(f"World: {WORLD_FILE}")
    print(f"Max time: {MAX_SIM_TIME}s")
    print(f"Landing tolerance: XY={LANDING_TOLERANCE_XY}m, Z={LANDING_TOLERANCE_Z}m")
    print()

    # Start Gazebo server headless
    print("Starting Gazebo server...")
    gz_proc = subprocess.Popen(
        ['gz', 'sim', '-s', '-r', WORLD_FILE],  # -r to run immediately
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=ENV
    )

    time.sleep(3)  # Wait for sim to initialize

    if gz_proc.poll() is not None:
        print("ERROR: Gazebo failed to start")
        stderr = gz_proc.stderr.read().decode()
        print(stderr)
        return False

    print("Gazebo running. Monitoring positions...")
    print("-" * 60)

    start_time = time.time()
    landed = False
    last_status = ""

    try:
        while time.time() - start_time < MAX_SIM_TIME:
            # Get poses
            poses = get_poses_from_topic()

            quad_pos = None
            ship_pos = None

            if 'quadcopter' in poses:
                p = poses['quadcopter']
                if 'x' in p and 'y' in p and 'z' in p:
                    quad_pos = (p['x'], p['y'], p['z'])

            if 'ship' in poses:
                p = poses['ship']
                if 'x' in p and 'y' in p and 'z' in p:
                    ship_pos = (p['x'], p['y'], p['z'])

            landed, status = check_landing(quad_pos, ship_pos)

            if status != last_status:
                elapsed = time.time() - start_time
                print(f"[{elapsed:5.1f}s] {status}")
                last_status = status

            if landed:
                print("-" * 60)
                print("SUCCESS: Quad landed on helipad!")
                break

            time.sleep(0.5)

        if not landed:
            print("-" * 60)
            print("TIMEOUT: Landing not achieved")

    except KeyboardInterrupt:
        print("\nTest interrupted")

    finally:
        print("Stopping Gazebo...")
        gz_proc.terminate()
        try:
            gz_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gz_proc.kill()

    return landed


if __name__ == '__main__':
    success = run_test()
    sys.exit(0 if success else 1)
