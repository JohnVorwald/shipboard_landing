#!/usr/bin/env python3
"""
Simple Autonomous Landing Controller for Gazebo

Minimal version that works reliably:
- No ship pose updates (ship stays static visually)
- Helipad position calculated from wave model
- Simple force control
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
from ship_wave_motion import ShipWaveMotion, SEA_STATE_4

# Gazebo
GZ_ENV = os.environ.copy()
GZ_ENV['GZ_IP'] = '127.0.0.1'
GZ_ENV['GZ_PARTITION'] = 'gazebo_default'

WORLD = "ship_landing_simple"
MASS = 2.0
G = 9.81
HOVER_THRUST = MASS * G
HELIPAD_OFFSET = np.array([15.0, 0.0, 4.0])


def get_quad_pose():
    """Get quadcopter pose from Gazebo."""
    try:
        result = subprocess.run(
            ['gz', 'topic', '-e', '-t', f'/world/{WORLD}/dynamic_pose/info', '-n', '1'],
            capture_output=True, text=True, timeout=1.5, env=GZ_ENV
        )
        if result.returncode == 0:
            # Find quadcopter section
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'name: "quadcopter"' in line:
                    pos = [0, 0, 0]
                    for j in range(i, min(i+15, len(lines))):
                        pline = lines[j].strip()
                        if pline.startswith('x:'):
                            val = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', pline)
                            if val: pos[0] = float(val.group())
                        elif pline.startswith('y:'):
                            val = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', pline)
                            if val: pos[1] = float(val.group())
                        elif pline.startswith('z:') and 'position' not in pline:
                            val = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', pline)
                            if val: pos[2] = float(val.group())
                    return np.array(pos)
    except:
        pass
    return None


def apply_force(force):
    """Apply force to quadcopter."""
    msg = f'entity: {{name: "quadcopter::base_link", type: LINK}}, wrench: {{force: {{x: {force[0]}, y: {force[1]}, z: {force[2]}}}}}'
    subprocess.Popen(
        ['gz', 'topic', '-t', f'/world/{WORLD}/wrench/persistent',
         '-m', 'gz.msgs.EntityWrench', '-p', msg],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=GZ_ENV
    )


def clear_force():
    """Clear persistent force."""
    msg = 'entity: {name: "quadcopter::base_link", type: LINK}'
    try:
        subprocess.run(
            ['gz', 'topic', '-t', f'/world/{WORLD}/wrench/clear',
             '-m', 'gz.msgs.Entity', '-p', msg],
            capture_output=True, timeout=0.5, env=GZ_ENV
        )
    except:
        pass


def main():
    print("=" * 60)
    print("Simple Autonomous Landing")
    print("=" * 60)

    # Check Gazebo
    quad_pos = get_quad_pose()
    if quad_pos is None:
        print("ERROR: Cannot read quad pose. Is Gazebo running?")
        print("Start with: gz sim -s -r worlds/ship_landing_simple.world")
        return 1

    print(f"Quad at: ({quad_pos[0]:.1f}, {quad_pos[1]:.1f}, {quad_pos[2]:.1f})")

    # Setup guidance
    config = ZEMGuidanceConfig(
        N_position=3.0,
        N_velocity=2.0,
        max_accel=3.0,  # Reduced
        max_descent=2.0,
        max_climb=3.0,
        terminal_height=2.0
    )
    guidance = ZEMGuidance(config)

    # Ship motion model (ship at origin)
    wave_model = ShipWaveMotion(sea_state=SEA_STATE_4, forward_speed=2.5)
    ship_initial = np.array([0.0, 0.0, 0.0])

    # State
    prev_pos = None
    prev_time = None
    vel = np.zeros(3)
    landed = False
    start_time = time.time()

    print("\nStarting landing approach...")
    print("-" * 60)

    t = 0
    dt = 0.05
    max_time = 45.0

    while t < max_time and not landed:
        loop_start = time.time()
        sim_time = time.time() - start_time

        # Get quad position
        new_pos = get_quad_pose()
        if new_pos is not None:
            quad_pos = new_pos

        # Estimate velocity
        now = time.time()
        if prev_pos is not None and prev_time is not None:
            dt_vel = now - prev_time
            if 0.01 < dt_vel < 0.3:
                new_vel = (quad_pos - prev_pos) / dt_vel
                new_vel = np.clip(new_vel, -15, 15)  # Clamp
                vel = 0.3 * new_vel + 0.7 * vel
        prev_pos = quad_pos.copy()
        prev_time = now

        # Get helipad position from wave model
        helipad_pos, helipad_vel = wave_model.get_helipad_state(sim_time, ship_initial, HELIPAD_OFFSET)

        # Relative state
        rel_pos = quad_pos - helipad_pos
        height = rel_pos[2]
        horiz_dist = np.linalg.norm(rel_pos[:2])

        # Check landing
        if height < 0.5 and horiz_dist < 2.5 and np.linalg.norm(vel) < 2.0:
            landed = True
            force = np.array([0, 0, HOVER_THRUST * 0.3])
        else:
            # Guidance in NED
            quad_ned = np.array([quad_pos[0], quad_pos[1], -quad_pos[2]])
            vel_ned = np.array([vel[0], vel[1], -vel[2]])
            target_ned = np.array([helipad_pos[0], helipad_pos[1], -helipad_pos[2]])
            target_vel_ned = np.array([helipad_vel[0], helipad_vel[1], -helipad_vel[2]])

            acc_ned, _ = guidance.compute_control(quad_ned, vel_ned, target_ned, target_vel_ned)

            # Convert to Gazebo frame
            acc = np.array([acc_ned[0], acc_ned[1], -acc_ned[2]])

            # Force
            force = MASS * acc
            force[2] += HOVER_THRUST

            # Clamp aggressively
            force[2] = np.clip(force[2], 0.6 * HOVER_THRUST, 1.4 * HOVER_THRUST)
            max_horiz = MASS * 2.5
            horiz = np.sqrt(force[0]**2 + force[1]**2)
            if horiz > max_horiz:
                force[:2] *= max_horiz / horiz

        apply_force(force.tolist())

        # Print status
        if int(t * 2) % 2 == 0:
            print(f"[{t:5.1f}s] Q:({quad_pos[0]:6.1f},{quad_pos[1]:5.1f},{quad_pos[2]:5.1f}) "
                  f"V:({vel[0]:+5.1f},{vel[1]:+4.1f},{vel[2]:+4.1f}) "
                  f"H:{height:5.1f}m D:{horiz_dist:5.1f}m F:({force[0]:+4.0f},{force[1]:+4.0f},{force[2]:4.0f})")

        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)
        t += dt

    clear_force()

    print("-" * 60)
    if landed:
        print("SUCCESS: Landed!")
        final = get_quad_pose()
        if final is not None:
            print(f"Final position: ({final[0]:.1f}, {final[1]:.1f}, {final[2]:.1f})")
    else:
        print("TIMEOUT")

    return 0 if landed else 1


if __name__ == '__main__':
    sys.exit(main())
