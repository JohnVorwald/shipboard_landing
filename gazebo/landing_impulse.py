#!/usr/bin/env python3
"""
Landing Controller using Impulse-based Wrench (Non-Persistent)

The key insight: /wrench/persistent applies a CONTINUOUS force that persists
until cleared. Each call ADDS to the existing force, causing runaway.

This version uses /wrench (non-persistent) which applies a ONE-TIME impulse
per message. This is more like how real motor controllers work.
"""

import subprocess
import time
import sys
import os
import re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ship_wave_motion import ShipWaveMotion, SEA_STATE_4

# Gazebo environment
GZ_ENV = os.environ.copy()
GZ_ENV['GZ_IP'] = '127.0.0.1'
GZ_ENV['GZ_PARTITION'] = 'gazebo_default'

WORLD = "ship_landing_simple"
MASS = 2.0
G = 9.81
HOVER_THRUST = MASS * G
HELIPAD_OFFSET = np.array([15.0, 0.0, 4.0])


def get_quad_pose():
    """Get quad pose from dynamic_pose topic."""
    try:
        result = subprocess.run(
            ['gz', 'topic', '-e', '-t', f'/world/{WORLD}/dynamic_pose/info', '-n', '1'],
            capture_output=True, text=True, timeout=1.0, env=GZ_ENV
        )
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'name: "quadcopter"' in line:
                    pos = [0.0, 0.0, 0.0]
                    in_pos = False
                    for j in range(i, min(i+15, len(lines))):
                        pline = lines[j].strip()
                        if pline == 'position {':
                            in_pos = True
                        elif pline.startswith('}') and in_pos:
                            break
                        elif in_pos:
                            for idx, c in enumerate(['x:', 'y:', 'z:']):
                                if pline.startswith(c):
                                    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', pline)
                                    if m:
                                        pos[idx] = float(m.group())
                    return np.array(pos)
    except:
        pass
    return None


def apply_impulse(force, duration=0.001):
    """
    Apply a one-time force impulse (non-persistent).

    The /wrench topic applies force for ONE physics step only.
    This is the correct approach for external force control.
    """
    msg = f'entity: {{name: "quadcopter::base_link", type: LINK}}, wrench: {{force: {{x: {force[0]}, y: {force[1]}, z: {force[2]}}}}}'
    subprocess.Popen(
        ['gz', 'topic', '-t', f'/world/{WORLD}/wrench',
         '-m', 'gz.msgs.EntityWrench', '-p', msg],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=GZ_ENV
    )


def clear_persistent():
    """Clear any persistent wrenches that might be lingering."""
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
    print("IMPULSE-BASED LANDING CONTROLLER")
    print("=" * 60)
    print("Using non-persistent wrench (impulse per timestep)")
    print()

    # Clear any persistent wrenches first
    clear_persistent()
    time.sleep(0.5)

    # Get initial pose
    pos = get_quad_pose()
    if pos is None:
        print("ERROR: Cannot read quad pose")
        print(f"Start Gazebo: gz sim -s -r worlds/{WORLD}.world")
        return 1

    print(f"Quad at: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

    # Setup
    wave_model = ShipWaveMotion(sea_state=SEA_STATE_4, forward_speed=2.5)
    ship_initial = np.array([0.0, 0.0, 0.0])

    prev_pos = None
    prev_time = None
    vel = np.zeros(3)
    start_time = time.time()

    print("\nStarting control loop...")
    print("-" * 60)

    # Simple PD gains for position control
    Kp = np.array([1.0, 1.0, 2.0])  # Position gains
    Kd = np.array([2.0, 2.0, 3.0])  # Velocity damping

    dt = 0.02  # 50 Hz
    t = 0
    max_time = 45

    while t < max_time:
        loop_start = time.time()
        sim_time = time.time() - start_time

        # Get pose
        new_pos = get_quad_pose()
        if new_pos is not None:
            pos = new_pos

        # Velocity estimation
        now = time.time()
        if prev_pos is not None and prev_time is not None:
            dt_vel = now - prev_time
            if 0.01 < dt_vel < 0.2:
                raw_vel = (pos - prev_pos) / dt_vel
                raw_vel = np.clip(raw_vel, -20, 20)
                vel = 0.3 * raw_vel + 0.7 * vel
        prev_pos = pos.copy()
        prev_time = now

        # Target: helipad position from wave model
        target_pos, target_vel = wave_model.get_helipad_state(sim_time, ship_initial, HELIPAD_OFFSET)

        # Relative state
        error = pos - target_pos
        error_vel = vel - target_vel
        height = error[2]
        horiz = np.linalg.norm(error[:2])

        # Check landing
        if height < 0.5 and horiz < 2.5 and np.linalg.norm(error_vel) < 2.0:
            print(f"\n>>> LANDED at t={t:.1f}s!")
            break

        # Simple PD control
        # Acceleration command = -Kp * error - Kd * error_vel
        acc_cmd = -Kp * error - Kd * error_vel

        # Clamp accelerations
        acc_cmd = np.clip(acc_cmd, -3, 3)

        # Force = mass * acceleration
        force = MASS * acc_cmd

        # Add gravity compensation
        force[2] += HOVER_THRUST

        # Clamp total force
        force[2] = np.clip(force[2], 0.5 * HOVER_THRUST, 1.5 * HOVER_THRUST)
        horiz_f = np.linalg.norm(force[:2])
        if horiz_f > MASS * 3:
            force[:2] *= (MASS * 3) / horiz_f

        # Apply impulse
        apply_impulse(force.tolist())

        # Print status
        if int(t * 2) % 2 == 0:
            print(f"[{t:5.1f}s] P:({pos[0]:6.1f},{pos[1]:5.1f},{pos[2]:5.1f}) "
                  f"V:({vel[0]:+5.1f},{vel[1]:+4.1f},{vel[2]:+4.1f}) "
                  f"H:{height:5.1f}m D:{horiz:5.1f}m "
                  f"F:({force[0]:+5.1f},{force[1]:+4.1f},{force[2]:5.1f})")

        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)
        t += dt

    print("-" * 60)
    final = get_quad_pose()
    if final is not None:
        print(f"Final position: ({final[0]:.1f}, {final[1]:.1f}, {final[2]:.1f})")

    return 0


if __name__ == '__main__':
    sys.exit(main())
