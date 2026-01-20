#!/usr/bin/env python3
"""
Autonomous Landing Controller with Ship Motion for Gazebo

Flies the quad from a starting position to land on a moving ship helipad.
Ship moves forward at ~5 knots with wave-induced motion.
"""

import subprocess
import time
import sys
import os
import re
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guidance.zem_guidance import ZEMGuidance, ZEMGuidanceConfig
from ship_wave_motion import ShipWaveMotion, SEA_STATE_3, SEA_STATE_4, SEA_STATE_5

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

# Ship motion parameters - using realistic wave model
SHIP_SPEED = 2.5  # m/s (~5 knots)
SEA_STATE = SEA_STATE_4  # Moderate seas


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


def set_pose(model_name, x, y, z, roll=0, pitch=0, yaw=0):
    """Set model pose using gz service."""
    # Convert euler to quaternion
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    msg = f'name: "{model_name}", position: {{x: {x}, y: {y}, z: {z}}}, orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {qw}}}'

    try:
        subprocess.run(
            ['gz', 'service', '-s', f'/world/{WORLD}/set_pose',
             '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
             '--timeout', '100', '--req', msg],
            capture_output=True, timeout=0.5, env=GZ_ENV
        )
    except:
        pass


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


class ShipMotionController:
    """Controls ship motion using realistic wave model."""

    def __init__(self, initial_pos):
        self.initial_pos = np.array(initial_pos)
        self.start_time = time.time()
        self.wave_model = ShipWaveMotion(
            sea_state=SEA_STATE,
            forward_speed=SHIP_SPEED
        )

    def get_state(self):
        """Get current ship position and velocity with realistic wave motion."""
        t = time.time() - self.start_time
        motion = self.wave_model.get_motion(t, self.initial_pos)

        pos = motion['position']
        vel = motion['linear_velocity']
        roll = motion['orientation'][0]
        pitch = motion['orientation'][1]

        return pos, vel, roll, pitch

    def get_helipad_state(self):
        """Get helipad position and velocity."""
        t = time.time() - self.start_time
        return self.wave_model.get_helipad_state(t, self.initial_pos, HELIPAD_OFFSET)


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

    def compute_control(self, quad_pos, ship_pos, ship_vel):
        """Compute thrust force for landing on moving ship."""
        # Helipad position accounts for ship motion
        deck_pos = ship_pos + HELIPAD_OFFSET
        deck_vel = ship_vel

        # Relative position and velocity
        rel_pos = quad_pos - deck_pos
        rel_vel = self.vel - deck_vel
        height = rel_pos[2]
        horiz_dist = np.linalg.norm(rel_pos[:2])

        # Check landing
        if height < 0.5 and horiz_dist < 2.0 and np.linalg.norm(rel_vel) < 1.5:
            self.landed = True
            return np.array([0, 0, HOVER_THRUST * 0.5])

        # Convert to NED for guidance (NED: +z down)
        quad_ned = np.array([quad_pos[0], quad_pos[1], -quad_pos[2]])
        vel_ned = np.array([self.vel[0], self.vel[1], -self.vel[2]])
        deck_ned = np.array([deck_pos[0], deck_pos[1], -deck_pos[2]])
        deck_vel_ned = np.array([deck_vel[0], deck_vel[1], -deck_vel[2]])

        # Get acceleration command
        acc_ned, _ = self.guidance.compute_control(quad_ned, vel_ned, deck_ned, deck_vel_ned)

        # Convert back to Gazebo (+z up)
        acc = np.array([acc_ned[0], acc_ned[1], -acc_ned[2]])

        # Force = mass * acceleration
        force = MASS * acc

        # Add gravity compensation
        force[2] += HOVER_THRUST

        # Clamp forces
        force[2] = np.clip(force[2], 0.3 * HOVER_THRUST, 2.0 * HOVER_THRUST)
        max_horiz = MASS * 5.0
        horiz_mag = np.sqrt(force[0]**2 + force[1]**2)
        if horiz_mag > max_horiz:
            force[0] *= max_horiz / horiz_mag
            force[1] *= max_horiz / horiz_mag

        return force

    def run(self, max_time=60.0, dt=0.02):
        """Run controller with moving ship."""
        print("=" * 60)
        print("Autonomous Landing Controller with Ship Motion")
        print("=" * 60)
        print(f"Ship speed: {SHIP_SPEED:.1f} m/s (~{SHIP_SPEED * 1.944:.1f} knots)")
        print(f"Sea State: Hs={SEA_STATE.significant_wave_height:.1f}m, Tp={SEA_STATE.modal_period:.1f}s")
        print()

        # Get initial ship position
        poses = get_poses()
        if 'ship' in poses:
            ship_initial = poses['ship']['pos']
        else:
            ship_initial = [0, 0, 0]

        ship_motion = ShipMotionController(ship_initial)

        t = 0
        update_count = 0

        while t < max_time and not self.landed:
            loop_start = time.time()

            poses = get_poses()
            if 'quadcopter' not in poses:
                time.sleep(dt)
                t += dt
                continue

            quad_pos = np.array(poses['quadcopter']['pos'])

            # Get ship state with motion
            ship_pos, ship_vel, roll, pitch = ship_motion.get_state()

            # Update ship pose in Gazebo
            if update_count % 2 == 0:  # Update every other cycle to reduce load
                set_pose('ship', ship_pos[0], ship_pos[1], ship_pos[2], roll, pitch, 0)

            self.estimate_velocity(quad_pos)
            force = self.compute_control(quad_pos, ship_pos, ship_vel)

            # Apply force
            apply_wrench('quadcopter::base_link', force.tolist())

            # Status
            deck_pos = ship_pos + HELIPAD_OFFSET
            rel_pos = quad_pos - deck_pos
            height = rel_pos[2]
            horiz_dist = np.linalg.norm(rel_pos[:2])

            if int(t * 2) % 2 == 0:
                print(f"[{t:5.1f}s] Q:({quad_pos[0]:6.1f},{quad_pos[1]:5.1f},{quad_pos[2]:5.1f}) "
                      f"S:({ship_pos[0]:5.1f},{ship_pos[2]:4.1f}) "
                      f"H:{height:5.1f}m D:{horiz_dist:5.1f}m")

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            t += dt
            update_count += 1

        # Clear wrench when done
        clear_wrench('quadcopter::base_link')

        print("-" * 60)
        if self.landed:
            print("SUCCESS: Landed on moving helipad!")
            final_poses = get_poses()
            if 'quadcopter' in final_poses:
                final_pos = final_poses['quadcopter']['pos']
                print(f"Final quad position: ({final_pos[0]:.1f}, {final_pos[1]:.1f}, {final_pos[2]:.1f})")
        else:
            print("TIMEOUT or MISSED")
        return self.landed


def main():
    print("Checking Gazebo...")
    poses = get_poses()
    if not poses:
        print("ERROR: Gazebo not running")
        print(f"Start with: gz sim -s -r worlds/ship_landing_simple.world")
        return 1

    if 'quadcopter' not in poses:
        print("ERROR: quadcopter not found")
        return 1

    print(f"Quad at: {poses['quadcopter']['pos']}")
    if 'ship' in poses:
        print(f"Ship at: {poses['ship']['pos']}")
    print("Starting controller...")

    controller = LandingController()
    success = controller.run()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
