#!/usr/bin/env python3
"""
Higher-Order Terminal Guidance for Shipboard Landing.

Implements polynomial guidance laws that account for:
1. Position, velocity, and acceleration matching at touchdown
2. Jerk-limited trajectories for smooth control
3. ARMA-predicted deck motion with derivatives
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Tuple

from quad_dynamics.quadrotor import QuadrotorParams, QuadrotorState
from ship_motion.ddg_motion import DDGParams, SeaState, DDGMotionSimulator, ARMAPredictor


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


@dataclass
class HigherOrderConfig:
    """Higher-order guidance configuration."""
    # Timing
    dt: float = 0.02
    replan_interval: float = 0.2  # Replan more frequently

    # Constraints
    max_accel: float = 8.0        # m/s²
    max_jerk: float = 10.0        # m/s³
    max_pitch: float = 0.5        # rad
    max_roll: float = 0.5         # rad

    # Gains for tracking (increased for better tracking)
    K_pos: float = 2.0
    K_vel: float = 3.0
    K_acc: float = 0.5


class QuinticPolynomial:
    """
    5th-order polynomial for smooth trajectory generation.

    Satisfies boundary conditions:
    - Initial: position, velocity, acceleration
    - Final: position, velocity, acceleration

    x(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
    """

    def __init__(self, x0: float, v0: float, a0: float,
                 xf: float, vf: float, af: float, T: float):
        """
        Generate quintic polynomial coefficients.

        Args:
            x0, v0, a0: Initial position, velocity, acceleration
            xf, vf, af: Final position, velocity, acceleration
            T: Duration
        """
        self.T = max(T, 0.1)
        T = self.T

        # Coefficients from boundary conditions
        self.a0 = x0
        self.a1 = v0
        self.a2 = a0 / 2

        # Solve for a3, a4, a5 from final conditions
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        # Matrix equation for [a3, a4, a5]
        A = np.array([
            [T3, T4, T5],
            [3*T2, 4*T3, 5*T4],
            [6*T, 12*T2, 20*T3]
        ])

        b = np.array([
            xf - self.a0 - self.a1*T - self.a2*T2,
            vf - self.a1 - 2*self.a2*T,
            af - 2*self.a2
        ])

        try:
            coeffs = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            coeffs = np.zeros(3)

        self.a3, self.a4, self.a5 = coeffs

    def evaluate(self, t: float) -> Tuple[float, float, float, float]:
        """
        Evaluate polynomial at time t.

        Returns:
            (position, velocity, acceleration, jerk)
        """
        t = np.clip(t, 0, self.T)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t

        pos = self.a0 + self.a1*t + self.a2*t2 + self.a3*t3 + self.a4*t4 + self.a5*t5
        vel = self.a1 + 2*self.a2*t + 3*self.a3*t2 + 4*self.a4*t3 + 5*self.a5*t4
        acc = 2*self.a2 + 6*self.a3*t + 12*self.a4*t2 + 20*self.a5*t3
        jerk = 6*self.a3 + 24*self.a4*t + 60*self.a5*t2

        return pos, vel, acc, jerk


class HigherOrderGuidance:
    """
    Higher-order terminal guidance with quintic polynomial trajectories.

    Plans smooth trajectories that match:
    - Position, velocity, acceleration at current time
    - Position, velocity, acceleration at touchdown (matching deck)
    """

    def __init__(self, config: HigherOrderConfig, quad_params: QuadrotorParams):
        self.config = config
        self.quad_params = quad_params
        self.g = 9.81

        # ARMA predictor
        self.arma = ARMAPredictor(ar_order=8, ma_order=4)
        self.arma_fitted = False
        self.arma_dt = 0.1

        # Current trajectory (one per axis)
        self.traj_x = None
        self.traj_y = None
        self.traj_z = None
        self.traj_start_time = 0
        self.traj_duration = 1.0

        # Deck acceleration estimate
        self.prev_deck_vel = None
        self.deck_accel = np.zeros(3)

    def fit_predictor(self, history: np.ndarray, dt: float):
        """Fit ARMA predictor."""
        self.arma.fit(history, dt)
        self.arma_dt = dt
        self.arma_fitted = True

    def update_predictor(self, observation: np.ndarray):
        """Update ARMA."""
        if self.arma_fitted:
            self.arma.update(observation)

    def estimate_deck_acceleration(self, deck_vel: np.ndarray, dt: float):
        """Estimate deck acceleration from velocity history."""
        if self.prev_deck_vel is not None:
            self.deck_accel = (deck_vel - self.prev_deck_vel) / dt
        self.prev_deck_vel = deck_vel.copy()

    def predict_deck_state(self, t_ahead: float, deck_pos: np.ndarray,
                          deck_vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict deck position, velocity, acceleration at future time.

        Returns:
            (position, velocity, acceleration)
        """
        if not self.arma_fitted or t_ahead < 0.1:
            # Simple constant velocity + acceleration model
            pred_pos = deck_pos + deck_vel * t_ahead + 0.5 * self.deck_accel * t_ahead**2
            pred_vel = deck_vel + self.deck_accel * t_ahead
            pred_acc = self.deck_accel.copy()
            return pred_pos, pred_vel, pred_acc

        # Use ARMA for oscillatory components
        n_steps = max(1, min(int(t_ahead / self.arma_dt), 100))
        preds = self.arma.predict(n_steps)

        if len(preds) == 0:
            return deck_pos + deck_vel * t_ahead, deck_vel.copy(), self.deck_accel.copy()

        pred = preds[-1]

        # Build predicted state
        pred_pos = deck_pos.copy()
        pred_vel = deck_vel.copy()
        pred_acc = self.deck_accel.copy()

        # X: constant velocity (ship forward motion)
        pred_pos[0] = deck_pos[0] + deck_vel[0] * t_ahead
        pred_vel[0] = pred[2] if len(pred) > 2 else deck_vel[0]

        # Z: from ARMA (heave)
        pred_pos[2] = pred[0]
        pred_vel[2] = pred[1]

        return pred_pos, pred_vel, pred_acc

    def plan_trajectory(self, quad_state: QuadrotorState, quad_accel: np.ndarray,
                       deck_pos: np.ndarray, deck_vel: np.ndarray,
                       time_to_touchdown: float, current_time: float):
        """
        Plan quintic polynomial trajectory to touchdown.
        Uses RELATIVE coordinates for better numerical stability.
        """
        T = max(time_to_touchdown, 0.5)

        # Predict deck at touchdown
        pred_deck_pos, pred_deck_vel, pred_deck_acc = self.predict_deck_state(
            T, deck_pos, deck_vel
        )

        # Current RELATIVE state (quad - deck)
        rel_pos = quad_state.pos - deck_pos
        rel_vel = quad_state.vel - deck_vel
        rel_acc = quad_accel - self.deck_accel

        # Target: zero relative position/velocity at touchdown
        target_rel_pos = np.zeros(3)
        target_rel_vel = np.zeros(3)
        target_rel_acc = np.zeros(3)

        # Plan trajectory for each axis IN RELATIVE FRAME
        self.traj_x = QuinticPolynomial(
            rel_pos[0], rel_vel[0], rel_acc[0],
            target_rel_pos[0], target_rel_vel[0], target_rel_acc[0], T
        )
        self.traj_y = QuinticPolynomial(
            rel_pos[1], rel_vel[1], rel_acc[1],
            target_rel_pos[1], target_rel_vel[1], target_rel_acc[1], T
        )
        self.traj_z = QuinticPolynomial(
            rel_pos[2], rel_vel[2], rel_acc[2],
            target_rel_pos[2], target_rel_vel[2], target_rel_acc[2], T
        )

        self.traj_start_time = current_time
        self.traj_duration = T

        # Store deck state at plan time for reference frame
        self.plan_deck_pos = deck_pos.copy()
        self.plan_deck_vel = deck_vel.copy()

    def compute_control(self, quad_state: QuadrotorState, deck_pos: np.ndarray,
                       deck_vel: np.ndarray, time_to_touchdown: float,
                       current_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute control using higher-order guidance in RELATIVE frame.

        Returns:
            control: [thrust, roll_cmd, pitch_cmd, yaw_rate_cmd]
            ref_accel: Reference acceleration for next iteration
        """
        cfg = self.config

        # Check if need to replan
        if (self.traj_x is None or
            current_time - self.traj_start_time > cfg.replan_interval):

            # Estimate current acceleration
            ref_accel = np.array([0, 0, 0])
            self.plan_trajectory(quad_state, ref_accel, deck_pos, deck_vel,
                               time_to_touchdown, current_time)

        # Evaluate trajectory (in RELATIVE frame)
        t_traj = current_time - self.traj_start_time

        ref_rel_pos_x, ref_rel_vel_x, ref_rel_acc_x, _ = self.traj_x.evaluate(t_traj)
        ref_rel_pos_y, ref_rel_vel_y, ref_rel_acc_y, _ = self.traj_y.evaluate(t_traj)
        ref_rel_pos_z, ref_rel_vel_z, ref_rel_acc_z, _ = self.traj_z.evaluate(t_traj)

        ref_rel_pos = np.array([ref_rel_pos_x, ref_rel_pos_y, ref_rel_pos_z])
        ref_rel_vel = np.array([ref_rel_vel_x, ref_rel_vel_y, ref_rel_vel_z])
        ref_rel_acc = np.array([ref_rel_acc_x, ref_rel_acc_y, ref_rel_acc_z])

        # Current relative state
        rel_pos = quad_state.pos - deck_pos
        rel_vel = quad_state.vel - deck_vel

        # Tracking controller in relative frame
        rel_pos_err = rel_pos - ref_rel_pos
        rel_vel_err = rel_vel - ref_rel_vel

        # Commanded RELATIVE acceleration = feedforward + feedback
        rel_a_cmd = ref_rel_acc - cfg.K_pos * rel_pos_err - cfg.K_vel * rel_vel_err

        # Convert to inertial acceleration (add deck acceleration)
        a_cmd = rel_a_cmd + self.deck_accel

        # Limit jerk (rate of acceleration change)
        # This is approximated by limiting acceleration magnitude
        a_horiz = np.sqrt(a_cmd[0]**2 + a_cmd[1]**2)
        if a_horiz > cfg.max_accel:
            a_cmd[0] *= cfg.max_accel / a_horiz
            a_cmd[1] *= cfg.max_accel / a_horiz

        # Convert to attitude commands
        yaw = quad_state.yaw
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

        a_forward = cos_yaw * a_cmd[0] + sin_yaw * a_cmd[1]
        a_right = -sin_yaw * a_cmd[0] + cos_yaw * a_cmd[1]

        pitch_cmd = np.arctan2(a_forward, self.g)
        roll_cmd = np.arctan2(-a_right, self.g)

        pitch_cmd = np.clip(pitch_cmd, -cfg.max_pitch, cfg.max_pitch)
        roll_cmd = np.clip(roll_cmd, -cfg.max_roll, cfg.max_roll)

        # Thrust
        thrust = self.quad_params.mass * (self.g + a_cmd[2]) / (np.cos(pitch_cmd) * np.cos(roll_cmd))
        thrust = np.clip(thrust, 0, self.quad_params.mass * self.g * 2)

        # Yaw rate
        dx = deck_pos[0] - quad_state.pos[0]
        dy = deck_pos[1] - quad_state.pos[1]
        desired_yaw = np.arctan2(dy, dx)
        yaw_error = np.arctan2(np.sin(desired_yaw - yaw), np.cos(desired_yaw - yaw))
        yaw_rate_cmd = np.clip(2.0 * yaw_error, -1.0, 1.0)

        control = np.array([thrust, roll_cmd, pitch_cmd, yaw_rate_cmd])
        return control, a_cmd


class HigherOrderSimulator:
    """Simulate landing with higher-order guidance."""

    def __init__(self, sea_state_num: int = 4, ship_speed_kts: float = 15.0):
        self.ship_params = DDGParams()
        self.sea_state = SeaState.from_state_number(sea_state_num, direction=45.0)
        self.ship_sim = DDGMotionSimulator(self.ship_params, self.sea_state, ship_speed_kts)

        self.quad_params = QuadrotorParams()
        self.config = HigherOrderConfig()
        self.guidance = HigherOrderGuidance(self.config, self.quad_params)

        self.dt = self.config.dt

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

        self.guidance.fit_predictor(np.array(history), 0.1)

        # Initialize quad
        ship_motion = self.ship_sim.get_motion(30.0)
        deck_pos = ship_motion['deck_position']
        deck_vel = ship_motion['deck_velocity']

        quad_start_pos = deck_pos.copy()
        quad_start_pos[0] -= 80
        quad_start_pos[2] = -25

        quad_state = QuadrotorState(
            pos=quad_start_pos,
            vel=deck_vel.copy(),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            omega=np.zeros(3)
        )

        print("Running landing simulation...")
        trajectory = {
            'time': [], 'quad_pos': [], 'quad_vel': [],
            'deck_pos': [], 'deck_vel': [], 'control': [], 'pos_error': []
        }

        t = 30.0
        prev_deck_vel = deck_vel.copy()

        while t < approach_time:
            ship_motion = self.ship_sim.get_motion(t)
            deck_pos = ship_motion['deck_position']
            deck_vel = ship_motion['deck_velocity']

            # Update predictor
            obs = np.array([
                ship_motion['deck_position'][2],
                ship_motion['deck_velocity'][2],
                ship_motion['deck_velocity'][0],
                ship_motion['attitude'][0],
                ship_motion['attitude'][1],
                ship_motion['angular_rate'][0],
            ])
            self.guidance.update_predictor(obs)
            self.guidance.estimate_deck_acceleration(deck_vel, self.dt)

            # Compute control
            time_to_touchdown = approach_time - t
            control, ref_accel = self.guidance.compute_control(
                quad_state, deck_pos, deck_vel, time_to_touchdown, t
            )

            # Apply control
            thrust = control[0]
            roll_cmd = control[1]
            pitch_cmd = control[2]
            yaw_rate_cmd = control[3]

            tau = 0.1
            roll_new = quad_state.roll + (roll_cmd - quad_state.roll) * self.dt / tau
            pitch_new = quad_state.pitch + (pitch_cmd - quad_state.pitch) * self.dt / tau
            yaw_new = quad_state.yaw + yaw_rate_cmd * self.dt

            quad_state.quat = euler_to_quat(roll_new, pitch_new, yaw_new)

            g = 9.81
            m = self.quad_params.mass
            ax = thrust / m * (np.sin(yaw_new) * np.sin(roll_new) +
                              np.cos(yaw_new) * np.sin(pitch_new) * np.cos(roll_new))
            ay = thrust / m * (-np.cos(yaw_new) * np.sin(roll_new) +
                              np.sin(yaw_new) * np.sin(pitch_new) * np.cos(roll_new))
            az = thrust / m * np.cos(pitch_new) * np.cos(roll_new) - g

            quad_state.vel += np.array([ax, ay, az]) * self.dt
            quad_state.pos += quad_state.vel * self.dt

            # Record
            pos_error = quad_state.pos - deck_pos
            trajectory['time'].append(t)
            trajectory['quad_pos'].append(quad_state.pos.copy())
            trajectory['quad_vel'].append(quad_state.vel.copy())
            trajectory['deck_pos'].append(deck_pos.copy())
            trajectory['deck_vel'].append(deck_vel.copy())
            trajectory['control'].append(control.copy())
            trajectory['pos_error'].append(pos_error.copy())

            # Check touchdown
            altitude_agl = -(quad_state.pos[2] - deck_pos[2])
            if altitude_agl < 0.5:
                print(f"Touchdown at t={t:.2f}s")
                break

            t += self.dt

        for key in trajectory:
            trajectory[key] = np.array(trajectory[key])

        final_pos_err = trajectory['pos_error'][-1]
        final_vel_err = trajectory['quad_vel'][-1] - trajectory['deck_vel'][-1]

        print("\nLanding Results:")
        print(f"  Position error: X={final_pos_err[0]:.3f}m, Y={final_pos_err[1]:.3f}m, Z={final_pos_err[2]:.3f}m")
        print(f"  Velocity error: X={final_vel_err[0]:.3f}m/s, Y={final_vel_err[1]:.3f}m/s, Z={final_vel_err[2]:.3f}m/s")
        print(f"  Total pos error: {np.linalg.norm(final_pos_err):.3f}m")
        print(f"  Total vel error: {np.linalg.norm(final_vel_err):.3f}m/s")

        return trajectory


def run_evaluation(n_trials: int = 10):
    """Evaluate higher-order guidance."""
    print("="*70)
    print("HIGHER-ORDER TERMINAL GUIDANCE EVALUATION")
    print("="*70)

    results = {'pos_error': [], 'vel_error': [], 'success': []}

    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        np.random.seed(trial * 100)

        sim = HigherOrderSimulator(sea_state_num=4)
        traj = sim.run_landing(approach_time=60.0)

        if len(traj['time']) > 0:
            final_pos_err = np.linalg.norm(traj['pos_error'][-1])
            final_vel_err = np.linalg.norm(traj['quad_vel'][-1] - traj['deck_vel'][-1])

            results['pos_error'].append(final_pos_err)
            results['vel_error'].append(final_vel_err)
            results['success'].append(final_pos_err < 3.0 and final_vel_err < 2.0)

    print("\n" + "="*70)
    print("HIGHER-ORDER GUIDANCE SUMMARY")
    print("="*70)
    print(f"Trials: {n_trials}")
    print(f"Success rate: {100 * np.mean(results['success']):.1f}%")
    print(f"Mean position error: {np.mean(results['pos_error']):.3f}m")
    print(f"Mean velocity error: {np.mean(results['vel_error']):.3f}m/s")
    print(f"Position error std: {np.std(results['pos_error']):.3f}m")
    print(f"Velocity error std: {np.std(results['vel_error']):.3f}m/s")

    return results


if __name__ == "__main__":
    run_evaluation(n_trials=10)
