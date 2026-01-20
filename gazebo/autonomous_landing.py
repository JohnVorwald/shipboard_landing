#!/usr/bin/env python3
"""
Autonomous Landing Controller for Gazebo

Flies the quad from a starting position to land on the ship helipad.
Uses ZEM/ZEV guidance for the approach and landing.
"""

import subprocess
import time
import sys
import os
import re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guidance.zem_guidance import ZEMGuidance, ZEMGuidanceConfig

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

# Helipad offset from ship origin (Gazebo: +z up)
HELIPAD_OFFSET = np.array([15.0, 0.0, 4.0])


def get_poses():
    """Get poses from Gazebo."""
    try:
        result = subprocess.run(
            ['gz', 'topic', '-e', '-t', f'/world/{WORLD}/pose/info', '-n', '1'],
            capture_output=True, text=True, timeout=2, env=GZ_ENV
        )
        if result.returncode == 0:
            return parse_poses(result.stdout)
    except:
        pass
    return {}


def parse_poses(output):
    """Parse pose message."""
    poses = {}
    current_name = None
    pos = [0, 0, 0]

    for line in output.split('\n'):
        line = line.strip()
        if 'name:' in line and '"' in line:
            if current_name:
                poses[current_name] = {'pos': pos.copy()}
            match = re.search(r'name:\s*"([^"]+)"', line)
            if match:
                current_name = match.group(1)
                pos = [0, 0, 0]
        elif current_name:
            for i, coord in enumerate(['x:', 'y:', 'z:']):
                if line.startswith(coord):
                    val = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line.split(':')[-1])
                    if val:
                        pos[i] = float(val.group())

    if current_name:
        poses[current_name] = {'pos': pos}
    return poses


def apply_wrench(link_name, force, torque=None, persistent=True):
    """Apply wrench to link using persistent wrench."""
    if torque is None:
        torque = [0, 0, 0]

    msg = f'entity: {{name: "{link_name}", type: LINK}}, wrench: {{force: {{x: {force[0]}, y: {force[1]}, z: {force[2]}}}, torque: {{x: {torque[0]}, y: {torque[1]}, z: {torque[2]}}}}}'

    topic = f'/world/{WORLD}/wrench/persistent' if persistent else f'/world/{WORLD}/wrench'

    try:
        subprocess.run(
            ['gz', 'topic', '-t', topic, '-m', 'gz.msgs.EntityWrench', '-p', msg],
            capture_output=True, timeout=0.3, env=GZ_ENV
        )
    except:
        pass


def clear_wrench(link_name):
    """Clear persistent wrench from link."""
    msg = f'entity: {{name: "{link_name}", type: LINK}}'
    try:
        subprocess.run(
            ['gz', 'topic', '-t', f'/world/{WORLD}/wrench/clear',
             '-m', 'gz.msgs.Entity', '-p', msg],
            capture_output=True, timeout=0.3, env=GZ_ENV
        )
    except:
        pass


class LandingController:
    """Autonomous landing controller."""

    def __init__(self):
        config = ZEMGuidanceConfig(
            N_position=3.0,
            N_velocity=2.0,
            max_accel=4.0,
            max_descent=2.0,
            max_climb=4.0,
            terminal_height=2.0
        )
        self.guidance = ZEMGuidance(config)
        self.prev_pos = None
        self.prev_time = None
        self.vel = np.zeros(3)
        self.landed = False

    def estimate_velocity(self, pos):
        """Estimate velocity from position."""
        now = time.time()
        if self.prev_pos is not None and self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 0.01:
                new_vel = (pos - self.prev_pos) / dt
                # Low-pass filter
                self.vel = 0.3 * new_vel + 0.7 * self.vel
        self.prev_pos = pos.copy()
        self.prev_time = now

    def compute_control(self, quad_pos, ship_pos):
        """Compute thrust force."""
        deck_pos = ship_pos + HELIPAD_OFFSET
        deck_vel = np.zeros(3)

        # Relative position
        rel_pos = quad_pos - deck_pos
        height = rel_pos[2]  # Gazebo: +z up
        horiz_dist = np.linalg.norm(rel_pos[:2])

        # Check landing
        if height < 0.5 and horiz_dist < 2.0 and np.linalg.norm(self.vel) < 1.0:
            self.landed = True
            return np.array([0, 0, HOVER_THRUST * 0.5])

        # Convert to NED for guidance (NED: +z down)
        quad_ned = np.array([quad_pos[0], quad_pos[1], -quad_pos[2]])
        vel_ned = np.array([self.vel[0], self.vel[1], -self.vel[2]])
        deck_ned = np.array([deck_pos[0], deck_pos[1], -deck_pos[2]])

        # Get acceleration command
        acc_ned, _ = self.guidance.compute_control(quad_ned, vel_ned, deck_ned, np.zeros(3))

        # Convert back to Gazebo (+z up)
        acc = np.array([acc_ned[0], acc_ned[1], -acc_ned[2]])

        # Force = mass * acceleration
        force = MASS * acc

        # Add gravity compensation
        force[2] += HOVER_THRUST

        # Clamp forces
        force[2] = np.clip(force[2], 0.3 * HOVER_THRUST, 1.8 * HOVER_THRUST)
        max_horiz = MASS * 4.0
        horiz_mag = np.sqrt(force[0]**2 + force[1]**2)
        if horiz_mag > max_horiz:
            force[0] *= max_horiz / horiz_mag
            force[1] *= max_horiz / horiz_mag

        return force

    def run(self, max_time=60.0, dt=0.02):
        """Run controller."""
        print("=" * 60)
        print("Autonomous Landing Controller")
        print("=" * 60)

        t = 0
        while t < max_time and not self.landed:
            loop_start = time.time()

            poses = get_poses()
            if 'quadcopter' not in poses:
                time.sleep(dt)
                t += dt
                continue

            quad_pos = np.array(poses['quadcopter']['pos'])
            ship_pos = np.array(poses.get('ship', {'pos': [0, 0, 0]})['pos'])

            self.estimate_velocity(quad_pos)
            force = self.compute_control(quad_pos, ship_pos)

            # Apply force
            apply_wrench('quadcopter::base_link', force.tolist())

            # Status
            deck_pos = ship_pos + HELIPAD_OFFSET
            height = quad_pos[2] - deck_pos[2]
            horiz_dist = np.linalg.norm(quad_pos[:2] - deck_pos[:2])

            if int(t * 2) % 2 == 0:
                print(f"[{t:5.1f}s] Pos:({quad_pos[0]:6.1f},{quad_pos[1]:5.1f},{quad_pos[2]:5.1f}) "
                      f"H:{height:5.1f}m D:{horiz_dist:5.1f}m F:({force[0]:5.1f},{force[1]:5.1f},{force[2]:5.1f})")

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            t += dt

        # Clear wrench when done
        clear_wrench('quadcopter::base_link')

        print("-" * 60)
        if self.landed:
            print("SUCCESS: Landed on helipad!")
        else:
            print("TIMEOUT")
        return self.landed


def main():
    print("Checking Gazebo...")
    poses = get_poses()
    if not poses:
        print("ERROR: Gazebo not running")
        print("Start with: gz sim -s -r worlds/ship_landing_simple.world")
        return 1

    if 'quadcopter' not in poses:
        print("ERROR: quadcopter not found")
        return 1

    print(f"Quad at: {poses['quadcopter']['pos']}")
    print("Starting controller...")

    controller = LandingController()
    success = controller.run()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
