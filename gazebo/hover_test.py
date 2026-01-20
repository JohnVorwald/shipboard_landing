#!/usr/bin/env python3
"""
Simple Hover Test using Non-Persistent Wrench

Based on Gemini's guidance: use /wrench (not /wrench/persistent)
to apply force for single physics steps.
"""

import subprocess
import time
import sys
import os
import re
import numpy as np

GZ_ENV = os.environ.copy()
GZ_ENV['GZ_IP'] = '127.0.0.1'
GZ_ENV['GZ_PARTITION'] = 'gazebo_default'

WORLD = "ship_landing_simple"
MASS = 2.0
GRAVITY = 9.81
HOVER_THRUST = MASS * GRAVITY


def get_pose():
    """Get quad pose."""
    try:
        result = subprocess.run(
            ['gz', 'topic', '-e', '-t', f'/world/{WORLD}/dynamic_pose/info', '-n', '1'],
            capture_output=True, text=True, timeout=0.5, env=GZ_ENV
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


def send_wrench(fx, fy, fz):
    """Send single-step wrench (non-persistent)."""
    msg = f'entity: {{name: "quadcopter::base_link", type: LINK}}, wrench: {{force: {{x: {fx}, y: {fy}, z: {fz}}}}}'
    subprocess.Popen(
        ['gz', 'topic', '-t', f'/world/{WORLD}/wrench',
         '-m', 'gz.msgs.EntityWrench', '-p', msg],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=GZ_ENV
    )


def clear_persistent():
    """Clear any persistent wrenches."""
    msg = 'entity: {name: "quadcopter::base_link", type: LINK}'
    subprocess.run(
        ['gz', 'topic', '-t', f'/world/{WORLD}/wrench/clear',
         '-m', 'gz.msgs.Entity', '-p', msg],
        capture_output=True, timeout=0.5, env=GZ_ENV
    )


class SimplePID:
    """Simple PID controller."""
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0
        self.last_error = 0

    def compute(self, current, dt):
        error = self.setpoint - current
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10, 10)  # Anti-windup
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


def main():
    print("=" * 60)
    print("HOVER TEST - Using Single-Step Wrench")
    print("=" * 60)

    # Clear any lingering persistent wrenches
    print("Clearing persistent wrenches...")
    clear_persistent()
    time.sleep(0.2)

    # Get initial position
    pos = get_pose()
    if pos is None:
        print("ERROR: Cannot get quad pose")
        print(f"Start Gazebo: gz sim -s -r worlds/{WORLD}.world")
        return 1

    print(f"Initial position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

    # Target altitude - hover at current height
    target_z = pos[2]
    print(f"Target altitude: {target_z:.1f}m")

    # PID controller for altitude (tuned based on Gemini's suggestion)
    altitude_pid = SimplePID(kp=20.0, ki=5.0, kd=15.0, setpoint=target_z)

    # Also add simple position hold for x,y
    x_pid = SimplePID(kp=5.0, ki=0.5, kd=8.0, setpoint=pos[0])
    y_pid = SimplePID(kp=5.0, ki=0.5, kd=8.0, setpoint=pos[1])

    print("\nStarting hover control at 100Hz...")
    print("-" * 60)

    dt = 0.01  # 100 Hz for better responsiveness
    t = 0
    max_time = 30
    last_print = 0

    prev_pos = pos.copy()
    prev_time = time.time()
    vel = np.zeros(3)

    while t < max_time:
        loop_start = time.time()

        # Get current position
        new_pos = get_pose()
        if new_pos is not None:
            pos = new_pos

        # Estimate velocity
        now = time.time()
        actual_dt = now - prev_time
        if 0.005 < actual_dt < 0.2:
            raw_vel = (pos - prev_pos) / actual_dt
            raw_vel = np.clip(raw_vel, -30, 30)
            vel = 0.3 * raw_vel + 0.7 * vel
        prev_pos = pos.copy()
        prev_time = now

        # Compute PID corrections
        fz_correction = altitude_pid.compute(pos[2], dt)
        fx_correction = x_pid.compute(pos[0], dt)
        fy_correction = y_pid.compute(pos[1], dt)

        # Total force = hover + correction
        fz = HOVER_THRUST + fz_correction
        fx = fx_correction
        fy = fy_correction

        # Clamp forces
        fz = np.clip(fz, 0.3 * HOVER_THRUST, 2.0 * HOVER_THRUST)
        fx = np.clip(fx, -MASS * 5, MASS * 5)
        fy = np.clip(fy, -MASS * 5, MASS * 5)

        # Send wrench
        send_wrench(fx, fy, fz)

        # Print status
        if time.time() - last_print > 0.5:
            alt_error = pos[2] - target_z
            print(f"[{t:5.1f}s] P:({pos[0]:6.1f},{pos[1]:5.1f},{pos[2]:5.1f}) "
                  f"V:({vel[0]:+5.1f},{vel[1]:+4.1f},{vel[2]:+5.1f}) "
                  f"Alt Err:{alt_error:+5.2f}m "
                  f"F:({fx:+5.1f},{fy:+4.1f},{fz:5.1f})")
            last_print = time.time()

        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)
        t += dt

    print("-" * 60)
    final = get_pose()
    if final is not None:
        print(f"Final position: ({final[0]:.1f}, {final[1]:.1f}, {final[2]:.1f})")
        print(f"Altitude error: {final[2] - target_z:+.2f}m")

    return 0


if __name__ == '__main__':
    sys.exit(main())
