#!/usr/bin/env python3
"""
Autonomous Landing Controller v2 for Gazebo

Key improvements:
- Applies hover thrust immediately to prevent falling
- Uses realistic ship wave motion
- Better state estimation
"""

import subprocess
import time
import sys
import os
import re
import math
import numpy as np
from threading import Thread
from queue import Queue, Empty

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
SEA_STATE = SEA_STATE_4  # Moderate seas


def apply_wrench_fast(link_name, force):
    """Apply wrench without waiting for response."""
    msg = f'entity: {{name: "{link_name}", type: LINK}}, wrench: {{force: {{x: {force[0]}, y: {force[1]}, z: {force[2]}}}}}'
    subprocess.Popen(
        ['gz', 'topic', '-t', f'/world/{WORLD}/wrench/persistent', '-m', 'gz.msgs.EntityWrench', '-p', msg],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=GZ_ENV
    )


def clear_wrench(link_name):
    """Clear persistent wrench."""
    msg = f'entity: {{name: "{link_name}", type: LINK}}'
    try:
        subprocess.run(
            ['gz', 'topic', '-t', f'/world/{WORLD}/wrench/clear', '-m', 'gz.msgs.Entity', '-p', msg],
            capture_output=True, timeout=0.5, env=GZ_ENV
        )
    except:
        pass


def set_pose_fast(model_name, x, y, z, roll=0, pitch=0, yaw=0):
    """Set pose without waiting."""
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


class PoseReader:
    """Background pose reader for non-blocking pose updates."""

    def __init__(self):
        self.quad_pos = np.array([-20.0, 0.0, 15.0])  # Default starting position
        self.ship_pos = np.array([0.0, 0.0, 0.0])
        self.running = False
        self.last_update = 0

    def parse_poses(self, output):
        """Parse pose message."""
        poses = {}
        current_name = None
        pos = [0, 0, 0]

        for line in output.split('\n'):
            line = line.strip()
            if 'name:' in line and '"' in line:
                if current_name and current_name in ['quadcopter', 'ship']:
                    poses[current_name] = pos.copy()
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

        if current_name and current_name in ['quadcopter', 'ship']:
            poses[current_name] = pos
        return poses

    def read_loop(self):
        """Continuously read poses."""
        while self.running:
            try:
                result = subprocess.run(
                    ['gz', 'topic', '-e', '-t', f'/world/{WORLD}/pose/info', '-n', '1'],
                    capture_output=True, text=True, timeout=1.5, env=GZ_ENV
                )
                if result.returncode == 0:
                    poses = self.parse_poses(result.stdout)
                    if 'quadcopter' in poses:
                        self.quad_pos = np.array(poses['quadcopter'])
                    if 'ship' in poses:
                        self.ship_pos = np.array(poses['ship'])
                    self.last_update = time.time()
            except:
                pass

    def start(self):
        """Start background reading."""
        self.running = True
        self.thread = Thread(target=self.read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop background reading."""
        self.running = False

    def get_poses(self):
        """Get latest poses."""
        return self.quad_pos.copy(), self.ship_pos.copy()


class LandingController:
    """Autonomous landing controller with immediate hover."""

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
            if dt > 0.005:
                new_vel = (pos - self.prev_pos) / dt
                alpha = min(0.5, dt * 10)  # Adaptive filter
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

        # Convert to NED (guidance uses NED)
        quad_ned = np.array([quad_pos[0], quad_pos[1], -quad_pos[2]])
        vel_ned = np.array([self.vel[0], self.vel[1], -self.vel[2]])
        target_ned = np.array([helipad_pos[0], helipad_pos[1], -helipad_pos[2]])
        target_vel_ned = np.array([helipad_vel[0], helipad_vel[1], -helipad_vel[2]])

        # Get acceleration command
        acc_ned, _ = self.guidance.compute_control(quad_ned, vel_ned, target_ned, target_vel_ned)

        # Convert back to Gazebo frame
        acc = np.array([acc_ned[0], acc_ned[1], -acc_ned[2]])

        # Compute force
        force = MASS * acc
        force[2] += HOVER_THRUST  # Gravity compensation

        # Clamp
        force[2] = np.clip(force[2], 0.3 * HOVER_THRUST, 2.0 * HOVER_THRUST)
        max_horiz = MASS * 5.0
        horiz = np.sqrt(force[0]**2 + force[1]**2)
        if horiz > max_horiz:
            force[0] *= max_horiz / horiz
            force[1] *= max_horiz / horiz

        return force

    def run(self, max_time=45.0, dt=0.02):
        """Run controller."""
        print("=" * 60)
        print("Autonomous Landing Controller v2")
        print("=" * 60)
        print(f"Ship speed: {SHIP_SPEED:.1f} m/s (~{SHIP_SPEED * 1.944:.1f} knots)")
        print(f"Sea State: Hs={SEA_STATE.significant_wave_height:.1f}m")
        print()

        # Apply hover immediately!
        print("Applying initial hover thrust...")
        apply_wrench_fast('quadcopter::base_link', [0, 0, HOVER_THRUST])
        time.sleep(0.1)

        # Start pose reader
        pose_reader = PoseReader()
        pose_reader.start()

        # Wait for first pose update
        print("Waiting for pose data...")
        start_wait = time.time()
        while time.time() - start_wait < 5.0:
            if pose_reader.last_update > 0:
                break
            apply_wrench_fast('quadcopter::base_link', [0, 0, HOVER_THRUST])
            time.sleep(0.1)

        if pose_reader.last_update == 0:
            print("ERROR: Could not get pose data")
            pose_reader.stop()
            return False

        # Initialize ship motion
        _, ship_initial = pose_reader.get_poses()
        wave_model = ShipWaveMotion(sea_state=SEA_STATE, forward_speed=SHIP_SPEED)
        start_time = time.time()

        print("Starting approach...")
        print("-" * 60)

        t = 0
        last_print = 0

        while t < max_time and not self.landed:
            loop_start = time.time()
            sim_time = time.time() - start_time

            # Get poses
            quad_pos, _ = pose_reader.get_poses()

            # Get ship/helipad state from wave model
            helipad_pos, helipad_vel = wave_model.get_helipad_state(sim_time, ship_initial, HELIPAD_OFFSET)

            # Update ship pose in Gazebo
            ship_state = wave_model.get_motion(sim_time, ship_initial)
            ship_pos = ship_state['position']
            roll, pitch, _ = ship_state['orientation']
            set_pose_fast('ship', ship_pos[0], ship_pos[1], ship_pos[2], roll, pitch, 0)

            # Compute and apply control
            self.estimate_velocity(quad_pos)
            force = self.compute_control(quad_pos, helipad_pos, helipad_vel)
            apply_wrench_fast('quadcopter::base_link', force.tolist())

            # Status output
            if time.time() - last_print > 0.5:
                rel_pos = quad_pos - helipad_pos
                height = rel_pos[2]
                horiz_dist = np.linalg.norm(rel_pos[:2])
                print(f"[{t:5.1f}s] Q:({quad_pos[0]:6.1f},{quad_pos[1]:5.1f},{quad_pos[2]:5.1f}) "
                      f"H:{height:5.1f}m D:{horiz_dist:5.1f}m Roll:{math.degrees(roll):+4.1f}°")
                last_print = time.time()

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            t += dt

        # Cleanup
        pose_reader.stop()
        clear_wrench('quadcopter::base_link')

        print("-" * 60)
        if self.landed:
            print("SUCCESS: Landed on moving helipad!")
        else:
            quad_pos, _ = pose_reader.get_poses()
            print(f"TIMEOUT - Final position: ({quad_pos[0]:.1f}, {quad_pos[1]:.1f}, {quad_pos[2]:.1f})")

        return self.landed


def main():
    print("Checking Gazebo...")

    # Quick check - look for pose topic
    try:
        result = subprocess.run(
            ['gz', 'topic', '-l'],
            capture_output=True, text=True, timeout=3, env=GZ_ENV
        )
        if f'/world/{WORLD}/pose/info' not in result.stdout:
            print(f"ERROR: World '{WORLD}' not running")
            print("Start Gazebo with: gz sim -s -r worlds/ship_landing_simple.world")
            print(f"Available topics: {result.stdout[:200]}")
            return 1
        print("Gazebo connected!")
    except Exception as e:
        print(f"ERROR: Cannot connect to Gazebo: {e}")
        return 1

    controller = LandingController()
    success = controller.run()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
