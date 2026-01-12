#!/usr/bin/env python3
"""
PMP Landing Demo - Headless Terminal Visualization
Demonstrates quadrotor landing on moving ship deck matching displacement and velocity

Run: python3 demo_pmp_landing.py [--sea-state N]
"""

import numpy as np
import sys
import time
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ship_motion.ddg_motion import DDGMotionSimulator, SeaState, DDGParams
from quad_dynamics.quadrotor import QuadrotorDynamics, QuadrotorParams, QuadrotorState
from optimal_control.trajectory_planner import MinSnapTrajectory


class CostateEstimator:
    """Simple costate estimator for PMP feedback."""

    def __init__(self, kp=2.0, kv=1.0):
        self.kp = kp
        self.kv = kv
        self.lambda_pos = np.zeros(3)
        self.lambda_vel = np.zeros(3)

    def update(self, pos_err, vel_err, dt):
        self.lambda_pos += pos_err * dt * self.kp
        self.lambda_vel += vel_err * dt * self.kv
        # Decay for stability
        self.lambda_pos *= 0.995
        self.lambda_vel *= 0.995

    def get_correction(self):
        return self.lambda_pos * 0.1 + self.lambda_vel * 0.3


class TerminalDisplay:
    """ASCII terminal display for landing simulation"""

    def __init__(self, width=80, height=24):
        self.width = width
        self.height = height

    def clear(self):
        print("\033[2J\033[H", end="")

    def draw_landing(self, t, quad_pos, quad_vel, deck_pos, deck_vel, deck_att,
                     pos_err, vel_err, phase, t_land):
        """Draw ASCII visualization of landing"""
        self.clear()

        # Header
        print("=" * self.width)
        print(f"  PMP SHIPBOARD LANDING DEMO - t={t:.1f}s".center(self.width))
        print("=" * self.width)

        # Phase indicator
        phase_str = {
            'approach': '>>> APPROACH >>>',
            'landing': '*** LANDING ***',
            'touchdown': '=== TOUCHDOWN ==='
        }.get(phase, phase)
        print(f"\n  Phase: {phase_str}")
        if t_land:
            print(f"  Landing in: {max(0, t_land - t):.1f}s")

        # Position display
        print("\n  POSITION (m)")
        print("  " + "-" * 50)
        print(f"  {'':12} {'X (Fwd)':>10} {'Y (Stbd)':>10} {'Z (Alt)':>10}")
        print(f"  {'Quad:':12} {quad_pos[0]:>10.2f} {quad_pos[1]:>10.2f} {-quad_pos[2]:>10.2f}")
        print(f"  {'Deck:':12} {deck_pos[0]:>10.2f} {deck_pos[1]:>10.2f} {-deck_pos[2]:>10.2f}")
        print(f"  {'Error:':12} {pos_err[0]:>10.2f} {pos_err[1]:>10.2f} {pos_err[2]:>10.2f}")

        # Velocity display
        print("\n  VELOCITY (m/s)")
        print("  " + "-" * 50)
        print(f"  {'':12} {'Vx':>10} {'Vy':>10} {'Vz':>10}")
        print(f"  {'Quad:':12} {quad_vel[0]:>10.2f} {quad_vel[1]:>10.2f} {-quad_vel[2]:>10.2f}")
        print(f"  {'Deck:':12} {deck_vel[0]:>10.2f} {deck_vel[1]:>10.2f} {-deck_vel[2]:>10.2f}")
        print(f"  {'Error:':12} {vel_err[0]:>10.2f} {vel_err[1]:>10.2f} {vel_err[2]:>10.2f}")

        # Attitude display
        print("\n  DECK ATTITUDE (deg)")
        print("  " + "-" * 50)
        roll_deg = np.degrees(deck_att[0])
        pitch_deg = np.degrees(deck_att[1])
        print(f"  Roll: {roll_deg:>6.1f}°  Pitch: {pitch_deg:>6.1f}°")

        # ASCII side view
        print("\n  SIDE VIEW")
        print("  " + "-" * 50)
        self._draw_side_view(quad_pos, deck_pos, deck_att[1])

        # Error metrics
        pos_err_mag = np.linalg.norm(pos_err)
        vel_err_mag = np.linalg.norm(vel_err)
        print(f"\n  Position Error: {pos_err_mag:.3f} m")
        print(f"  Velocity Error: {vel_err_mag:.3f} m/s")

        # Progress bar for landing
        if phase == 'landing' and t_land:
            progress = min(1.0, t / t_land)
            bar_width = 40
            filled = int(bar_width * progress)
            bar = '█' * filled + '░' * (bar_width - filled)
            print(f"\n  Landing Progress: [{bar}] {progress*100:.0f}%")

        print("\n" + "=" * self.width)
        sys.stdout.flush()

    def _draw_side_view(self, quad_pos, deck_pos, pitch):
        """Draw ASCII side view of quad and ship"""
        view_width = 50
        view_height = 10

        # Scale positions to view
        x_range = 80  # meters
        z_range = 40  # meters

        def to_view(x, z):
            vx = int((x + 20) / x_range * view_width)
            vz = int((40 + z) / z_range * view_height)
            vx = max(0, min(view_width - 1, vx))
            vz = max(0, min(view_height - 1, vz))
            return vx, view_height - 1 - vz

        # Create view buffer
        view = [[' ' for _ in range(view_width)] for _ in range(view_height)]

        # Draw ocean
        for x in range(view_width):
            view[view_height - 1][x] = '~'

        # Draw ship deck
        deck_x, deck_z = to_view(deck_pos[0], deck_pos[2])
        ship_len = 8
        for dx in range(-ship_len, ship_len + 1):
            sx = deck_x + dx
            # Apply pitch to ship visual
            sz = deck_z + int(dx * np.sin(pitch) * 0.3)
            if 0 <= sx < view_width and 0 <= sz < view_height:
                view[sz][sx] = '='

        # Draw helipad marker
        if 0 <= deck_x < view_width and 0 <= deck_z < view_height:
            view[deck_z][deck_x] = 'H'

        # Draw quadrotor
        quad_vx, quad_vz = to_view(quad_pos[0], quad_pos[2])
        if 0 <= quad_vx < view_width and 0 <= quad_vz < view_height:
            view[quad_vz][quad_vx] = '*'
            # Draw rotor indicators
            for dx in [-1, 1]:
                rx = quad_vx + dx
                if 0 <= rx < view_width:
                    view[quad_vz][rx] = 'o'

        # Print view
        for row in view:
            print("  " + ''.join(row))


class SimplePMPController:
    """Simplified PMP controller for demo - outputs throttle commands."""

    def __init__(self, mass=2.5, kp=4.0, kd=3.0, kp_att=8.0, kd_att=2.0):
        self.mass = mass
        self.kp = kp
        self.kd = kd
        self.kp_att = kp_att
        self.kd_att = kd_att
        self.g = 9.81
        self.hover_throttle = 0.5  # Approximate hover throttle

    def compute_control(self, pos, vel, quat, omega,
                       pos_target, vel_target, acc_ff, yaw_target=0):
        """Compute control: returns [throttle1, throttle2, throttle3, throttle4] (0-1)"""

        # Position error (target - current for correct sign)
        pos_err = pos_target - pos
        vel_err = vel_target - vel

        # Desired acceleration (NED frame)
        a_des = self.kp * pos_err + self.kd * vel_err + acc_ff

        # Gravity compensation: need upward force to counter gravity
        # In NED, positive z is down, so we need negative z acceleration to go up
        # Total accel needed: a_des - g (since gravity pulls down at +g in NED)
        a_total = a_des.copy()
        a_total[2] = a_des[2] - self.g  # This gives negative value for hover (upward force)

        # Thrust magnitude (force = mass * accel, pointing up in body frame)
        thrust_accel = np.linalg.norm(a_total)

        # Convert to throttle (approximate)
        # At hover, throttle ~0.5 produces 1g upward accel
        throttle_base = thrust_accel / (2 * self.g) + 0.1  # Approximate mapping
        throttle_base = np.clip(throttle_base, 0.1, 0.95)

        # Desired attitude from acceleration direction
        if thrust_accel > 0.5:
            # Body z-axis should point opposite to desired acceleration
            z_body_des = -a_total / thrust_accel
        else:
            z_body_des = np.array([0, 0, -1])

        # Extract desired roll and pitch from desired z-axis
        # Small angle approximation for roll/pitch
        roll_des = np.arcsin(np.clip(z_body_des[1], -0.5, 0.5))
        pitch_des = np.arcsin(np.clip(-z_body_des[0], -0.5, 0.5))

        # Current attitude from quaternion
        qw, qx, qy, qz = quat
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))

        # Attitude errors
        roll_err = roll_des - roll
        pitch_err = pitch_des - pitch
        yaw_err = np.arctan2(np.sin(yaw_target - np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))),
                            np.cos(yaw_target - np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))))

        # Attitude rate commands
        roll_cmd = self.kp_att * roll_err - self.kd_att * omega[0]
        pitch_cmd = self.kp_att * pitch_err - self.kd_att * omega[1]
        yaw_cmd = 0.3 * self.kp_att * yaw_err - self.kd_att * omega[2]

        # Mix into motor commands (simplified X-quad mixer)
        # Motor layout: 1=FR, 2=BR, 3=BL, 4=FL
        # Roll: motors 1,2 vs 3,4
        # Pitch: motors 1,4 vs 2,3
        # Yaw: motors 1,3 vs 2,4
        mix_scale = 0.15

        t1 = throttle_base + mix_scale * (-roll_cmd + pitch_cmd - yaw_cmd)
        t2 = throttle_base + mix_scale * (-roll_cmd - pitch_cmd + yaw_cmd)
        t3 = throttle_base + mix_scale * (+roll_cmd - pitch_cmd - yaw_cmd)
        t4 = throttle_base + mix_scale * (+roll_cmd + pitch_cmd + yaw_cmd)

        throttles = np.array([t1, t2, t3, t4])
        throttles = np.clip(throttles, 0.05, 0.95)

        return throttles


def run_demo(sea_state_num=5, duration=20.0, dt=0.01):
    """Run the PMP landing demo"""

    display = TerminalDisplay()

    # Initialize models
    print("Initializing simulation...")

    ship_params = DDGParams()
    sea_state = SeaState.from_state_number(sea_state_num, direction=30)  # 30 deg quartering seas
    ship = DDGMotionSimulator(ship_params, sea_state, ship_speed_kts=12)

    quad_params = QuadrotorParams()
    quad_dynamics = QuadrotorDynamics(quad_params)

    # Get initial deck state
    motion_init = ship.get_motion(0)
    deck_pos_init = motion_init['deck_position']

    # Initial quad state - hovering above and behind deck
    quad_pos_init = deck_pos_init + np.array([-30, 0, -20])  # 30m behind, 20m above
    quad_state = QuadrotorState(
        pos=quad_pos_init,
        vel=np.array([ship.ship_speed, 0, 0]),  # Match ship forward speed
        quat=np.array([1, 0, 0, 0]),  # Level attitude
        omega=np.zeros(3)  # Zero angular velocity
    )

    # Controllers
    controller = SimplePMPController(mass=quad_params.mass)
    costate_est = CostateEstimator(kp=2.0, kv=1.0)

    # Landing timing
    landing_duration = 8.0

    # Planner
    planner = MinSnapTrajectory()
    traj_start_time = 0.0

    # Generate initial trajectory to deck
    motion_target = ship.get_motion(landing_duration)
    deck_pos_target = motion_target['deck_position']
    deck_vel_target = motion_target['deck_velocity']

    trajectory = planner.plan(
        pos_init=quad_pos_init,
        vel_init=np.array([ship.ship_speed, 0, 0]),
        pos_final=deck_pos_target,
        vel_final=deck_vel_target,
        tf=landing_duration
    )

    # Simulation loop
    t = 0
    phase = 'approach'
    last_display = -1
    last_replan = 0

    # Data logging
    log = {'t': [], 'pos_err': [], 'vel_err': []}

    print("Starting simulation...")
    print("  Sea state:", sea_state_num)
    print("  Landing duration:", landing_duration, "s")
    time.sleep(1)

    try:
        while t < duration:
            # Get current deck state
            motion = ship.get_motion(t)
            deck_pos = motion['deck_position']
            deck_vel = motion['deck_velocity']
            deck_att = motion['attitude']

            # Get quad state
            quad_pos = quad_state.pos.copy()
            quad_vel = quad_state.vel.copy()
            quad_quat = quad_state.quat.copy()
            quad_omega = quad_state.omega.copy()

            # Compute errors
            pos_err = quad_pos - deck_pos
            vel_err = quad_vel - deck_vel

            # Update phase
            if t >= landing_duration - 0.5:
                phase = 'touchdown'
            elif t >= landing_duration * 0.3:
                phase = 'landing'
            else:
                phase = 'approach'

            # Replan trajectory periodically (every 1s)
            if t - last_replan >= 1.0 and t < landing_duration - 1:
                # Get updated deck prediction
                motion_target = ship.get_motion(landing_duration)
                deck_pos_target = motion_target['deck_position']
                deck_vel_target = motion_target['deck_velocity']

                trajectory = planner.plan(
                    pos_init=quad_pos,
                    vel_init=quad_vel,
                    pos_final=deck_pos_target,
                    vel_final=deck_vel_target,
                    tf=landing_duration - t
                )
                traj_start_time = t
                last_replan = t

            # Get trajectory targets
            if t < landing_duration:
                t_traj = t - traj_start_time
                traj_pos, traj_vel, traj_acc = planner.sample(trajectory, t_traj)
            else:
                # After landing, track deck
                traj_pos = deck_pos
                traj_vel = deck_vel
                traj_acc = np.zeros(3)

            # Update costate estimator
            traj_err = quad_pos - traj_pos
            traj_vel_err = quad_vel - traj_vel
            costate_est.update(traj_err, traj_vel_err, dt)

            # Compute control with PMP
            costate_correction = costate_est.get_correction()

            u = controller.compute_control(
                pos=quad_pos,
                vel=quad_vel,
                quat=quad_quat,
                omega=quad_omega,
                pos_target=traj_pos,
                vel_target=traj_vel,
                acc_ff=traj_acc - costate_correction,
                yaw_target=deck_att[2]
            )

            # Step quadrotor dynamics
            quad_state = quad_dynamics.step(quad_state, u, dt)

            # Log data
            log['t'].append(t)
            log['pos_err'].append(np.linalg.norm(pos_err))
            log['vel_err'].append(np.linalg.norm(vel_err))

            # Update display at 10 Hz
            if t - last_display >= 0.1:
                display.draw_landing(
                    t=t,
                    quad_pos=quad_pos,
                    quad_vel=quad_vel,
                    deck_pos=deck_pos,
                    deck_vel=deck_vel,
                    deck_att=deck_att,
                    pos_err=pos_err,
                    vel_err=vel_err,
                    phase=phase,
                    t_land=landing_duration
                )
                last_display = t
                time.sleep(0.05)  # Slow down for visibility

            t += dt

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted.")

    # Print final results
    print("\n" + "=" * 60)
    print("  LANDING RESULTS")
    print("=" * 60)

    # Final errors (average of last 0.5s)
    n_final = int(0.5 / dt)
    if len(log['pos_err']) > n_final:
        final_pos_err = np.mean(log['pos_err'][-n_final:])
        final_vel_err = np.mean(log['vel_err'][-n_final:])

        print(f"\n  Final Position Error: {final_pos_err:.3f} m")
        print(f"  Final Velocity Error: {final_vel_err:.3f} m/s")

        # Success criteria
        pos_ok = final_pos_err < 0.5
        vel_ok = final_vel_err < 0.3

        print(f"\n  Landing Quality:")
        print(f"    Position: {'PASS' if pos_ok else 'FAIL'} (< 0.5m)")
        print(f"    Velocity: {'PASS' if vel_ok else 'FAIL'} (< 0.3m/s)")

        if pos_ok and vel_ok:
            print("\n  *** SUCCESSFUL LANDING ***")
        else:
            print("\n  --- LANDING NEEDS IMPROVEMENT ---")

    print("\n" + "=" * 60)

    return log


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PMP Landing Demo')
    parser.add_argument('--sea-state', type=int, default=5,
                        help='Sea state 1-9 (default: 5)')
    parser.add_argument('--duration', type=float, default=15.0,
                        help='Simulation duration in seconds')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode (no display delay)')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  PMP SHIPBOARD LANDING DEMONSTRATION")
    print("  Matching deck displacement and velocity at touchdown")
    print("=" * 60)
    print(f"\n  Sea State: {args.sea_state}")
    print(f"  Duration:  {args.duration}s")
    print("\n  Press Ctrl+C to stop\n")
    time.sleep(2)

    run_demo(sea_state_num=args.sea_state, duration=args.duration)
