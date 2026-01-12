#!/usr/bin/env python3
"""
PMP Landing Simulation with Matching Criteria

Evaluates PMP-based landing with terminal constraints for:
- Position matching
- Velocity matching
- Attitude matching
- Impact velocity limits
- Lateral drift limits

Compares against Simple MPC (ZEM/ZEV) baseline.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Tuple

from quad_dynamics.quadrotor import QuadrotorParams, QuadrotorState
from ship_motion.ddg_motion import DDGParams, SeaState, DDGMotionSimulator, ARMAPredictor
from optimal_control.pmp_landing import PMPLandingController, PMPLandingConfig, LandingMatchingCriteria


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles to quaternion."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ])


class PMPMatchingSimulator:
    """Simulate landing with PMP matching criteria."""

    def __init__(self, sea_state_num: int = 4, ship_speed_kts: float = 15.0):
        self.ship_params = DDGParams()
        self.sea_state = SeaState.from_state_number(sea_state_num, direction=45.0)
        self.ship_sim = DDGMotionSimulator(self.ship_params, self.sea_state, ship_speed_kts)

        self.quad_params = QuadrotorParams()

        # Configure PMP with matching criteria
        matching = LandingMatchingCriteria(
            pos_weight=200.0,
            vel_weight=150.0,
            att_weight=50.0,
            impact_vel_max=1.0,
            lateral_vel_max=0.5,
            impact_penalty=500.0,
            lateral_penalty=300.0
        )
        config = PMPLandingConfig(
            replan_interval=0.5,
            Kp_pos=np.array([2.5, 2.5, 4.0]),
            Kd_pos=np.array([3.5, 3.5, 5.0]),
            costate_gain=0.15,
            matching=matching
        )
        self.controller = PMPLandingController(self.quad_params, config)

        # ARMA predictor
        self.arma = ARMAPredictor(ar_order=8, ma_order=4)
        self.arma_fitted = False

        self.dt = 0.02

    def run_landing(self, approach_time: float = 60.0) -> dict:
        """Run landing simulation."""
        # Warmup ARMA
        print("Warming up predictor...")
        history = []
        for t in np.arange(0, 30, 0.1):
            motion = self.ship_sim.get_motion(t)
            obs = np.array([
                motion['deck_position'][2],
                motion['deck_velocity'][2],
                motion['deck_velocity'][0],
                motion['attitude'][0],
                motion['attitude'][1],
                motion['angular_rate'][0],
            ])
            history.append(obs)

        self.arma.fit(np.array(history), 0.1)
        self.arma_fitted = True

        # Initialize quad
        ship_motion = self.ship_sim.get_motion(30.0)
        deck_pos = ship_motion['deck_position']
        deck_vel = ship_motion['deck_velocity']

        quad_start_pos = deck_pos.copy()
        quad_start_pos[0] -= 80
        quad_start_pos[2] = -25

        # State vector: [pos, vel, att, omega]
        x = np.zeros(12)
        x[0:3] = quad_start_pos
        x[3:6] = deck_vel.copy()

        print("Running PMP landing simulation...")
        trajectory = {
            'time': [], 'quad_pos': [], 'quad_vel': [],
            'deck_pos': [], 'deck_vel': [], 'deck_att': [],
            'control': [], 'pos_error': [], 'matching_status': []
        }

        t = 30.0
        while t < approach_time:
            ship_motion = self.ship_sim.get_motion(t)
            deck_pos = ship_motion['deck_position']
            deck_vel = ship_motion['deck_velocity']
            deck_att = ship_motion['attitude']

            # Update ARMA predictor
            obs = np.array([
                ship_motion['deck_position'][2],
                ship_motion['deck_velocity'][2],
                ship_motion['deck_velocity'][0],
                ship_motion['attitude'][0],
                ship_motion['attitude'][1],
                ship_motion['angular_rate'][0],
            ])
            self.arma.update(obs)

            # Compute control
            time_to_touchdown = approach_time - t
            control = self.controller.compute_control(
                t, x, deck_pos, deck_vel, deck_att, time_to_touchdown
            )

            # Apply control
            thrust = control[0]
            roll_cmd = control[1]
            pitch_cmd = control[2]
            yaw_rate_cmd = control[3]

            # Simple attitude dynamics
            tau = 0.1
            roll_new = x[6] + (roll_cmd - x[6]) * self.dt / tau
            pitch_new = x[7] + (pitch_cmd - x[7]) * self.dt / tau
            yaw_new = x[8] + yaw_rate_cmd * self.dt

            x[6:9] = [roll_new, pitch_new, yaw_new]

            # Acceleration
            g = 9.81
            m = self.quad_params.mass
            roll, pitch, yaw = roll_new, pitch_new, yaw_new

            ax = thrust / m * (np.sin(yaw) * np.sin(roll) +
                              np.cos(yaw) * np.sin(pitch) * np.cos(roll))
            ay = thrust / m * (-np.cos(yaw) * np.sin(roll) +
                              np.sin(yaw) * np.sin(pitch) * np.cos(roll))
            az = thrust / m * np.cos(pitch) * np.cos(roll) - g

            # Update state
            x[3:6] += np.array([ax, ay, az]) * self.dt
            x[0:3] += x[3:6] * self.dt

            # Get matching status
            status = self.controller.get_matching_status(x, deck_pos, deck_vel, deck_att)

            # Record
            pos_error = x[0:3] - deck_pos
            trajectory['time'].append(t)
            trajectory['quad_pos'].append(x[0:3].copy())
            trajectory['quad_vel'].append(x[3:6].copy())
            trajectory['deck_pos'].append(deck_pos.copy())
            trajectory['deck_vel'].append(deck_vel.copy())
            trajectory['deck_att'].append(deck_att.copy())
            trajectory['control'].append(control.copy())
            trajectory['pos_error'].append(pos_error.copy())
            trajectory['matching_status'].append(status)

            # Check touchdown
            altitude_agl = -(x[2] - deck_pos[2])
            if altitude_agl < 0.5:
                print(f"Touchdown at t={t:.2f}s")
                break

            t += self.dt

        for key in ['time', 'quad_pos', 'quad_vel', 'deck_pos', 'deck_vel',
                   'deck_att', 'control', 'pos_error']:
            trajectory[key] = np.array(trajectory[key])

        # Final results
        final_pos_err = trajectory['pos_error'][-1]
        final_vel_err = trajectory['quad_vel'][-1] - trajectory['deck_vel'][-1]
        final_status = trajectory['matching_status'][-1]

        print("\nLanding Results:")
        print(f"  Position error: X={final_pos_err[0]:.3f}m, Y={final_pos_err[1]:.3f}m, Z={final_pos_err[2]:.3f}m")
        print(f"  Velocity error: X={final_vel_err[0]:.3f}m/s, Y={final_vel_err[1]:.3f}m/s, Z={final_vel_err[2]:.3f}m/s")
        print(f"  Total pos error: {np.linalg.norm(final_pos_err):.3f}m")
        print(f"  Total vel error: {np.linalg.norm(final_vel_err):.3f}m/s")
        print(f"\nMatching Criteria:")
        print(f"  Position OK: {final_status['pos_ok']} ({final_status['pos_error']:.3f}m)")
        print(f"  Velocity OK: {final_status['vel_ok']} ({final_status['vel_error']:.3f}m/s)")
        print(f"  Lateral vel: {final_status['lateral_vel']:.3f}m/s (limit: 0.5)")
        print(f"  Descent rate: {final_status['descent_rate']:.3f}m/s (limit: 1.0)")
        print(f"  ALL CRITERIA MET: {final_status['all_ok']}")

        return trajectory


def run_evaluation(n_trials: int = 10):
    """Evaluate PMP landing with matching criteria."""
    print("="*70)
    print("PMP LANDING WITH MATCHING CRITERIA EVALUATION")
    print("="*70)

    results = {
        'pos_error': [], 'vel_error': [],
        'lateral_vel': [], 'descent_rate': [],
        'pos_ok': [], 'vel_ok': [], 'all_ok': []
    }

    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        np.random.seed(trial * 100)

        sim = PMPMatchingSimulator(sea_state_num=4)
        traj = sim.run_landing(approach_time=60.0)

        if len(traj['time']) > 0:
            final_pos_err = np.linalg.norm(traj['pos_error'][-1])
            final_vel_err = np.linalg.norm(traj['quad_vel'][-1] - traj['deck_vel'][-1])
            status = traj['matching_status'][-1]

            results['pos_error'].append(final_pos_err)
            results['vel_error'].append(final_vel_err)
            results['lateral_vel'].append(status['lateral_vel'])
            results['descent_rate'].append(status['descent_rate'])
            results['pos_ok'].append(status['pos_ok'])
            results['vel_ok'].append(status['vel_ok'])
            results['all_ok'].append(status['all_ok'])

    print("\n" + "="*70)
    print("PMP MATCHING CRITERIA SUMMARY")
    print("="*70)
    print(f"Trials: {n_trials}")
    print(f"\nPosition:")
    print(f"  Mean error: {np.mean(results['pos_error']):.3f}m")
    print(f"  Std: {np.std(results['pos_error']):.3f}m")
    print(f"  Success rate (<0.5m): {100*np.mean(results['pos_ok']):.1f}%")
    print(f"\nVelocity:")
    print(f"  Mean error: {np.mean(results['vel_error']):.3f}m/s")
    print(f"  Std: {np.std(results['vel_error']):.3f}m/s")
    print(f"  Success rate (<0.3m/s): {100*np.mean(results['vel_ok']):.1f}%")
    print(f"\nConstraints:")
    print(f"  Mean lateral vel: {np.mean(results['lateral_vel']):.3f}m/s")
    print(f"  Mean descent rate: {np.mean(results['descent_rate']):.3f}m/s")
    print(f"\nOverall Success Rate (all criteria): {100*np.mean(results['all_ok']):.1f}%")

    return results


if __name__ == "__main__":
    run_evaluation(n_trials=10)
