#!/usr/bin/env python3
"""
Autonomous Landing Controller v3 for Gazebo

Simplified pose reading with proper parsing.
"""

import subprocess
import time
import sys
import os
import re
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guidance.zem_guidance import ZEMGuidance, ZEMGuidanceConfig
from ship_wave_motion import ShipWaveMotion, SEA_STATE_3, SEA_STATE_4

# Gazebo environment
GZ_ENV = os.environ.copy()
GZ_ENV.update({
    'GZ_IP': '127.0.0.1',
    'GZ_PARTITION': 'gazebo_default',
})

WORLD = "ship_landing_simple"

# Quad parameters
MASS = 2.0
G = 9.81
HOVER_THRUST = MASS * G

# Helipad offset from ship origin
HELIPAD_OFFSET = np.array([15.0, 0.0, 4.0])

# Ship motion
SHIP_SPEED = 2.5  # m/s (~5 knots)
SEA_STATE = SEA_STATE_4


def get_poses():
    """Get poses from Gazebo dynamic_pose topic."""
    try:
        result = subprocess.run(
            ['gz', 'topic', '-e', '-t', f'/world/{WORLD}/dynamic_pose/info', '-n', '1'],
            capture_output=True, text=True, timeout=1.5, env=GZ_ENV
        )
        if result.returncode == 0:
            return parse_poses(result.stdout)
    except:
        pass
    return {}


def parse_poses(output):
    """Parse dynamic pose message - only contains moving models."""
    poses = {}
    lines = output.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for model name
        if line.startswith('name:') and '"' in line:
            match = re.search(r'name:\s*"([^"]+)"', line)
            if match:
                name = match.group(1)
                # Skip if it's a link (links come after models with same name pattern)
                if name in ['base_link', 'hull', 'link']:
                    i += 1
                    continue
                pos = [0.0, 0.0, 0.0]
                # Read until we find position block
                j = i + 1
                in_position = False
                while j < len(lines) and j < i + 15:
                    pline = lines[j].strip()
                    if pline == 'position {':
                        in_position = True
                    elif pline == '}' and in_position:
                        break
                    elif in_position:
                        if pline.startswith('x:'):
                            val = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', pline)
                            if val:
                                pos[0] = float(val.group())
                        elif pline.startswith('y:'):
                            val = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', pline)
                            if val:
                                pos[1] = float(val.group())
                        elif pline.startswith('z:'):
                            val = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', pline)
                            if val:
                                pos[2] = float(val.group())
                    j += 1
                poses[name] = np.array(pos)
        i += 1
    return poses


def get_model_pose(model_name):
    """Get a specific model's pose."""
    poses = get_poses()
    return poses.get(model_name)


def apply_wrench(link_name, force):
    """Apply wrench to link."""
    msg = f'entity: {{name: "{link_name}", type: LINK}}, wrench: {{force: {{x: {force[0]}, y: {force[1]}, z: {force[2]}}}}}'
    try:
        subprocess.Popen(
            ['gz', 'topic', '-t', f'/world/{WORLD}/wrench/persistent',
             '-m', 'gz.msgs.EntityWrench', '-p', msg],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=GZ_ENV
        )
    except:
        pass


def clear_wrench(link_name):
    """Clear persistent wrench."""
    msg = f'entity: {{name: "{link_name}", type: LINK}}'
    try:
        subprocess.run(
            ['gz', 'topic', '-t', f'/world/{WORLD}/wrench/clear',
             '-m', 'gz.msgs.Entity', '-p', msg],
            capture_output=True, timeout=0.5, env=GZ_ENV
        )
    except:
        pass


def set_pose(model_name, x, y, z, roll=0, pitch=0, yaw=0):
    """Set model pose."""
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    msg = f'name: "{model_name}", position: {{x: {x}, y: {y}, z: {z}}}, orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {qw}}}'
    subprocess.Popen(
        ['gz', 'service', '-s', f'/world/{WORLD}/set_pose',
         '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
         '--timeout', '100', '--req', msg],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=GZ_ENV
    )


class LandingController:
    """Autonomous landing controller."""

    def __init__(self):
        config = ZEMGuidanceConfig(
            N_position=3.0,
            N_velocity=2.0,
            max_accel=5.0,
            max_descent=3.0,
            max_climb=5.0,
            terminal_height=2.0
        )
        self.guidance = ZEMGuidance(config)
        self.prev_pos = None
        self.prev_time = None
        self.vel = np.zeros(3)
        self.landed = False

    def estimate_velocity(self, pos):
        """Estimate velocity from position changes with rate limiting."""
        now = time.time()
        if self.prev_pos is not None and self.prev_time is not None:
            dt = now - self.prev_time
            if 0.01 < dt < 0.3:
                new_vel = (pos - self.prev_pos) / dt
                # Clamp unreasonable velocities (max 20 m/s)
                new_vel = np.clip(new_vel, -20, 20)
                alpha = min(0.3, dt * 5)
                self.vel = alpha * new_vel + (1 - alpha) * self.vel
        self.prev_pos = pos.copy()
        self.prev_time = now

    def compute_control(self, quad_pos, helipad_pos, helipad_vel):
        """Compute thrust force."""
        rel_pos = quad_pos - helipad_pos
        rel_vel = self.vel - helipad_vel
        height = rel_pos[2]
        horiz_dist = np.linalg.norm(rel_pos[:2])

        # Landing check
        if height < 0.5 and horiz_dist < 2.0 and np.linalg.norm(rel_vel) < 1.5:
            self.landed = True
            return np.array([0, 0, HOVER_THRUST * 0.3])

        # Convert to NED
        quad_ned = np.array([quad_pos[0], quad_pos[1], -quad_pos[2]])
        vel_ned = np.array([self.vel[0], self.vel[1], -self.vel[2]])
        target_ned = np.array([helipad_pos[0], helipad_pos[1], -helipad_pos[2]])
        target_vel_ned = np.array([helipad_vel[0], helipad_vel[1], -helipad_vel[2]])

        # Get acceleration command
        acc_ned, _ = self.guidance.compute_control(quad_ned, vel_ned, target_ned, target_vel_ned)

        # Convert back
        acc = np.array([acc_ned[0], acc_ned[1], -acc_ned[2]])

        # Compute force
        force = MASS * acc
        force[2] += HOVER_THRUST

        # Clamp forces more aggressively
        max_vertical = 1.5 * HOVER_THRUST  # Max 1.5x hover
        min_vertical = 0.5 * HOVER_THRUST  # Min 0.5x hover
        force[2] = np.clip(force[2], min_vertical, max_vertical)

        max_horiz = MASS * 3.0  # Max 3 m/s² horizontal
        horiz = np.sqrt(force[0]**2 + force[1]**2)
        if horiz > max_horiz:
            force[0] *= max_horiz / horiz
            force[1] *= max_horiz / horiz

        return force

    def run(self, max_time=45.0, dt=0.05):
        """Run controller."""
        print("=" * 60)
        print("Autonomous Landing Controller v3")
        print("=" * 60)
        print(f"Ship speed: {SHIP_SPEED:.1f} m/s (~{SHIP_SPEED * 1.944:.1f} knots)")
        print(f"Sea State: Hs={SEA_STATE.significant_wave_height:.1f}m")
        print()

        # Reset quad to starting position
        QUAD_START = np.array([-20.0, 0.0, 15.0])
        print("Resetting quad to starting position...")
        set_pose('quadcopter', QUAD_START[0], QUAD_START[1], QUAD_START[2])
        clear_wrench('quadcopter::base_link')
        time.sleep(0.2)

        # Get initial positions
        print("Getting initial positions...")
        quad_pos = get_model_pose('quadcopter')
        ship_pos = get_model_pose('ship')

        if quad_pos is None:
            quad_pos = QUAD_START.copy()
            print(f"Using default quad position: ({quad_pos[0]:.1f}, {quad_pos[1]:.1f}, {quad_pos[2]:.1f})")
        else:
            print(f"Quad starting at: ({quad_pos[0]:.1f}, {quad_pos[1]:.1f}, {quad_pos[2]:.1f})")

        if ship_pos is None:
            ship_pos = np.array([0.0, 0.0, 0.0])
        print(f"Ship starting at: ({ship_pos[0]:.1f}, {ship_pos[1]:.1f}, {ship_pos[2]:.1f})")

        # Initialize ship motion
        ship_initial = ship_pos.copy()
        wave_model = ShipWaveMotion(sea_state=SEA_STATE, forward_speed=SHIP_SPEED)
        start_time = time.time()

        # Apply hover immediately
        print("\nApplying initial hover thrust...")
        apply_wrench('quadcopter::base_link', [0, 0, HOVER_THRUST])

        print("Starting approach...")
        print("-" * 60)

        t = 0
        last_print = 0
        pose_failures = 0

        while t < max_time and not self.landed:
            loop_start = time.time()
            sim_time = time.time() - start_time

            # Get quad pose
            new_pos = get_model_pose('quadcopter')
            if new_pos is not None:
                quad_pos = new_pos
                pose_failures = 0
            else:
                pose_failures += 1
                if pose_failures > 20:
                    print("ERROR: Lost quad position")
                    break

            # Get helipad state from wave model
            helipad_pos, helipad_vel = wave_model.get_helipad_state(sim_time, ship_initial, HELIPAD_OFFSET)

            # Update ship pose in Gazebo
            ship_state = wave_model.get_motion(sim_time, ship_initial)
            ship_pos = ship_state['position']
            roll, pitch, _ = ship_state['orientation']
            set_pose('ship', ship_pos[0], ship_pos[1], ship_pos[2], roll, pitch, 0)

            # Compute control
            self.estimate_velocity(quad_pos)
            force = self.compute_control(quad_pos, helipad_pos, helipad_vel)
            apply_wrench('quadcopter::base_link', force.tolist())

            # Status
            if time.time() - last_print > 0.5:
                rel_pos = quad_pos - helipad_pos
                height = rel_pos[2]
                horiz_dist = np.linalg.norm(rel_pos[:2])
                print(f"[{t:5.1f}s] Q:({quad_pos[0]:6.1f},{quad_pos[1]:5.1f},{quad_pos[2]:5.1f}) "
                      f"V:({self.vel[0]:+5.1f},{self.vel[1]:+4.1f},{self.vel[2]:+4.1f}) "
                      f"H:{height:5.1f}m D:{horiz_dist:5.1f}m")
                last_print = time.time()

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            t += dt

        # Cleanup
        clear_wrench('quadcopter::base_link')

        print("-" * 60)
        if self.landed:
            print("SUCCESS: Landed on moving helipad!")
            final_pos = get_model_pose('quadcopter')
            if final_pos is not None:
                print(f"Final position: ({final_pos[0]:.1f}, {final_pos[1]:.1f}, {final_pos[2]:.1f})")
        else:
            print("TIMEOUT or MISSED")

        return self.landed


def main():
    print("Checking Gazebo...")

    # Test pose reading
    quad_pos = get_model_pose('quadcopter')
    if quad_pos is None:
        print("ERROR: Cannot read quadcopter pose")
        print("Make sure Gazebo is running with: gz sim -s -r worlds/ship_landing_simple.world")
        return 1

    print(f"Found quadcopter at: ({quad_pos[0]:.1f}, {quad_pos[1]:.1f}, {quad_pos[2]:.1f})")

    controller = LandingController()
    success = controller.run()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
