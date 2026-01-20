#!/usr/bin/env python3
"""
Gazebo Landing Controller

Connects to Gazebo Ionic via subprocess commands, reads poses,
and applies velocity commands to land the quad on the ship.
"""

import subprocess
import time
import re
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guidance.zem_guidance import ZEMGuidance, ZEMGuidanceConfig

# Gazebo environment
GZ_ENV = os.environ.copy()
GZ_ENV.update({
    'GZ_IP': '127.0.0.1',
    'GZ_PARTITION': 'gazebo_default',
})


def parse_pose_msg(output: str) -> dict:
    """Parse gz topic pose output into dictionary."""
    poses = {}
    current_name = None
    current_pose = {}

    for line in output.split('\n'):
        line = line.strip()

        if 'name:' in line:
            if current_name and current_pose:
                poses[current_name] = current_pose
            match = re.search(r'name:\s*"([^"]+)"', line)
            if match:
                current_name = match.group(1)
                current_pose = {'pos': [0, 0, 0], 'quat': [1, 0, 0, 0]}
        elif current_name:
            if 'x:' in line and 'position' not in line.lower():
                val = re.search(r'[-+]?\d*\.?\d+', line.split(':')[-1])
                if val:
                    if 'orientation' in str(current_pose.get('_section', '')):
                        pass  # Skip orientation x
                    else:
                        current_pose['pos'][0] = float(val.group())
            elif 'y:' in line:
                val = re.search(r'[-+]?\d*\.?\d+', line.split(':')[-1])
                if val:
                    current_pose['pos'][1] = float(val.group())
            elif 'z:' in line and 'position' not in line.lower():
                val = re.search(r'[-+]?\d*\.?\d+', line.split(':')[-1])
                if val:
                    current_pose['pos'][2] = float(val.group())

    if current_name and current_pose:
        poses[current_name] = current_pose

    return poses


def get_poses() -> dict:
    """Get current poses from Gazebo."""
    try:
        result = subprocess.run(
            ['gz', 'topic', '-e', '-t', '/world/ship_simple_world/pose/info', '-n', '1'],
            capture_output=True, text=True, timeout=3, env=GZ_ENV
        )
        if result.returncode == 0:
            return parse_pose_msg(result.stdout)
    except Exception as e:
        print(f"Error getting poses: {e}")
    return {}


def apply_wrench(model: str, link: str, force: list, torque: list = None):
    """Apply wrench to a model link using gz service."""
    if torque is None:
        torque = [0, 0, 0]

    # Use gz service to apply wrench
    # Service: /world/ship_simple_world/wrench
    try:
        # Format the request
        req = f'entity: {{name: "{model}", type: MODEL}}, wrench: {{force: {{x: {force[0]}, y: {force[1]}, z: {force[2]}}}, torque: {{x: {torque[0]}, y: {torque[1]}, z: {torque[2]}}}}}'

        result = subprocess.run(
            ['gz', 'service', '-s', '/world/ship_simple_world/wrench',
             '--reqtype', 'gz.msgs.EntityWrench', '--reptype', 'gz.msgs.Boolean',
             '--timeout', '100', '--req', req],
            capture_output=True, text=True, timeout=2, env=GZ_ENV
        )
        return result.returncode == 0
    except Exception as e:
        return False


def set_velocity(model: str, linear: list, angular: list = None):
    """Set model velocity using gz service."""
    if angular is None:
        angular = [0, 0, 0]

    try:
        # Use world control to set velocity
        # This requires the ApplyLinkWrench system plugin
        req = f'name: "{model}", linear: {{x: {linear[0]}, y: {linear[1]}, z: {linear[2]}}}, angular: {{x: {angular[0]}, y: {angular[1]}, z: {angular[2]}}}'

        result = subprocess.run(
            ['gz', 'service', '-s', f'/world/ship_simple_world/set_physics',
             '--reqtype', 'gz.msgs.Physics', '--reptype', 'gz.msgs.Boolean',
             '--timeout', '100', '--req', req],
            capture_output=True, text=True, timeout=2, env=GZ_ENV
        )
        return result.returncode == 0
    except Exception as e:
        return False


def run_landing():
    """Run the landing controller."""
    print("=" * 60)
    print("Gazebo Shipboard Landing Controller")
    print("=" * 60)

    # Ship parameters
    ship_speed = 0.0  # Start with stationary ship for testing
    helipad_offset = np.array([15.0, 0.0, 2.2])  # Helipad offset from ship origin

    # Landing parameters
    landing_tolerance = 0.5
    max_time = 30.0
    dt = 0.1

    # Quad parameters
    mass = 2.0
    g = 9.81

    print("\nWaiting for Gazebo...")
    time.sleep(1)

    # Get initial poses
    poses = get_poses()
    if 'quadcopter' not in poses:
        print("ERROR: Quadcopter not found in simulation!")
        print(f"Available models: {list(poses.keys())}")
        return False

    print(f"\nInitial poses:")
    for name, pose in poses.items():
        if name in ['quadcopter', 'ship']:
            print(f"  {name}: {pose['pos']}")

    # Initialize guidance
    guidance = ZEMGuidance(ZEMGuidanceConfig(
        N_position=4.0,
        N_velocity=2.0,
        max_accel=6.0,
        max_descent=4.0
    ))

    print("\nStarting landing sequence...")
    print("-" * 60)

    t = 0.0
    landed = False

    while t < max_time and not landed:
        # Get current poses
        poses = get_poses()
        if 'quadcopter' not in poses or 'ship' not in poses:
            print("Lost tracking!")
            time.sleep(dt)
            t += dt
            continue

        quad_pos = np.array(poses['quadcopter']['pos'])
        ship_pos = np.array(poses['ship']['pos'])

        # Helipad position
        deck_pos = ship_pos + helipad_offset
        deck_vel = np.array([ship_speed, 0, 0])

        # Height above deck
        height = quad_pos[2] - deck_pos[2]  # In Gazebo, +z is up
        horiz_dist = np.sqrt((quad_pos[0] - deck_pos[0])**2 + (quad_pos[1] - deck_pos[1])**2)

        # Check landing
        if height < landing_tolerance and horiz_dist < 3.0:
            print(f"\n[{t:.1f}s] LANDED! Height: {height:.2f}m, Lateral: {horiz_dist:.2f}m")
            landed = True
            break

        # Status update
        if int(t * 10) % 10 == 0:
            print(f"[{t:.1f}s] Quad: ({quad_pos[0]:.1f}, {quad_pos[1]:.1f}, {quad_pos[2]:.1f}) | "
                  f"Height: {height:.1f}m | Dist: {horiz_dist:.1f}m")

        # Compute guidance (convert to Gazebo frame: +z up)
        # Our guidance uses NED (+z down), Gazebo uses ENU (+z up)
        quad_pos_ned = np.array([quad_pos[0], quad_pos[1], -quad_pos[2]])
        deck_pos_ned = np.array([deck_pos[0], deck_pos[1], -deck_pos[2]])
        quad_vel_ned = np.array([0, 0, 0])  # Estimated
        deck_vel_ned = np.array([deck_vel[0], deck_vel[1], -deck_vel[2]])

        acc_cmd_ned, info = guidance.compute_control(
            quad_pos_ned, quad_vel_ned, deck_pos_ned, deck_vel_ned
        )

        # Convert back to Gazebo frame
        acc_cmd = np.array([acc_cmd_ned[0], acc_cmd_ned[1], -acc_cmd_ned[2]])

        # Apply force (F = ma)
        force = mass * acc_cmd
        # Add gravity compensation
        force[2] += mass * g

        # Limit forces
        force = np.clip(force, -mass * g * 2, mass * g * 2)

        # Apply wrench to quad
        success = apply_wrench('quadcopter', 'base_link', force.tolist())
        if not success and t < 1.0:
            print("Note: Wrench service not available - quad will fall naturally")

        time.sleep(dt)
        t += dt

    print("-" * 60)
    if landed:
        print("SUCCESS: Quadcopter landed on ship!")
    else:
        print("TIMEOUT: Landing not completed")

    return landed


def main():
    """Main entry point."""
    # Check if Gazebo is running
    poses = get_poses()
    if not poses:
        print("Gazebo not running or not responding.")
        print("\nStart Gazebo with:")
        print("  cd gazebo")
        print("  export GZ_IP=127.0.0.1 GZ_PARTITION=gazebo_default")
        print("  gz sim -s -r worlds/ship_simple.world")
        return 1

    print(f"Found {len(poses)} models in Gazebo")

    success = run_landing()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
