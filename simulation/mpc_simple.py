#!/usr/bin/env python3
"""
Simplified MPC for shipboard landing using ZEM/ZEV guidance with prediction.

Instead of full nonlinear optimization, uses:
1. ARMA prediction of deck motion
2. ZEM/ZEV (Zero Effort Miss/Velocity) for optimal interception
3. Quadratic program for thrust allocation
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Tuple

from quad_dynamics.quadrotor import QuadrotorParams, QuadrotorState
from ship_motion.ddg_motion import DDGParams, SeaState, DDGMotionSimulator, ARMAPredictor


@dataclass
class SimpleMPCConfig:
    """Simple MPC configuration."""
    # Gains
    K_pos: float = 2.5           # Position error gain (increased)
    K_vel: float = 3.5           # Velocity error gain (increased)
    K_ff: float = 0.8            # Feedforward acceleration gain

    # Limits
    max_accel: float = 8.0       # Max horizontal acceleration (m/s²) - increased
    max_pitch: float = 0.5       # Max pitch angle (rad) - increased
    max_roll: float = 0.5        # Max roll angle (rad) - increased

    # Timing
    dt: float = 0.02             # Control timestep
    prediction_horizon: float = 3.0  # How far ahead to predict (s)
    update_interval: float = 0.5     # Predictor update interval (s)


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


class SimpleMPCController:
    """
    ZEM/ZEV-based MPC with deck motion prediction.

    Uses optimal interception guidance combined with ARMA prediction
    for anticipating deck motion.
    """

    def __init__(self, config: SimpleMPCConfig, quad_params: QuadrotorParams):
        self.config = config
        self.quad_params = quad_params
        self.g = 9.81

        # ARMA predictor
        self.arma = ARMAPredictor(ar_order=8, ma_order=4)
        self.arma_fitted = False
        self.arma_dt = 0.1

    def fit_predictor(self, history: np.ndarray, dt: float):
        """Fit ARMA predictor."""
        self.arma.fit(history, dt)
        self.arma_dt = dt
        self.arma_fitted = True

    def update_predictor(self, observation: np.ndarray):
        """Update ARMA with new observation."""
        if self.arma_fitted:
            self.arma.update(observation)

    def predict_deck_at_intercept(self, t_go: float, deck_pos: np.ndarray,
                                   deck_vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict deck position and velocity at intercept time.

        Uses ARMA for oscillatory motion and constant velocity for translation.
        """
        if not self.arma_fitted or t_go < 0.1:
            return deck_pos + deck_vel * t_go, deck_vel.copy()

        # Predict deck heave/roll/pitch
        n_steps = max(1, int(t_go / self.arma_dt))
        n_steps = min(n_steps, 100)  # Cap prediction horizon

        preds = self.arma.predict(n_steps)

        # Get final prediction
        pred = preds[-1] if len(preds) > 0 else np.zeros(6)

        # Build predicted deck state
        # ARMA outputs: [deck_z, deck_vz, deck_vx, roll, pitch, roll_rate]
        pred_pos = deck_pos.copy()
        pred_vel = deck_vel.copy()

        # Z position from ARMA (oscillatory)
        pred_pos[2] = pred[0]
        pred_vel[2] = pred[1]

        # X position from constant velocity
        pred_pos[0] = deck_pos[0] + deck_vel[0] * t_go
        pred_vel[0] = pred[2]

        # Y from roll coupling (approximate)
        roll_pred = pred[3]
        pred_pos[1] = deck_pos[1] + 0.3 * roll_pred * 20.0  # Roll-sway coupling

        return pred_pos, pred_vel

    def compute_control(self, quad_state: QuadrotorState, deck_pos: np.ndarray,
                       deck_vel: np.ndarray, time_to_touchdown: float) -> np.ndarray:
        """
        Compute control using ZEM/ZEV with prediction.

        Returns:
            [thrust, roll_cmd, pitch_cmd, yaw_rate_cmd]
        """
        cfg = self.config
        t_go = max(time_to_touchdown, 0.5)

        # Predict deck at intercept
        pred_deck_pos, pred_deck_vel = self.predict_deck_at_intercept(
            min(t_go, cfg.prediction_horizon), deck_pos, deck_vel
        )

        # Current relative state
        rel_pos = quad_state.pos - deck_pos
        rel_vel = quad_state.vel - deck_vel

        # Predicted intercept position (linear extrapolation)
        predicted_miss = rel_pos + rel_vel * t_go

        # ZEM/ZEV acceleration command
        # ZEM: -K_zem * (r + v*tgo) / tgo²
        # ZEV: -K_zev * (v - v_target) / tgo
        # Combined for optimal interception

        # Target velocity at touchdown (match deck)
        target_vel_rel = np.zeros(3)

        # Two-phase guidance: interception + soft landing
        dist_to_deck = np.linalg.norm(rel_pos)

        if dist_to_deck > 15.0:
            # Far out: ZEM/ZEV for interception
            a_zem = -6 * (rel_pos + rel_vel * t_go) / t_go**2
            a_zev = -2 * (rel_vel - target_vel_rel) / t_go
            a_cmd = a_zem + a_zev
        else:
            # Close to deck: proportional control for soft landing
            # Stronger gains when closer
            K_pos_scale = 1.0 + 0.5 * (1.0 - dist_to_deck / 15.0)
            K_vel_scale = 1.0 + 0.3 * (1.0 - dist_to_deck / 15.0)
            a_cmd = -cfg.K_pos * K_pos_scale * rel_pos - cfg.K_vel * K_vel_scale * rel_vel

        # Add feedforward for predicted deck motion
        deck_accel = (pred_deck_vel - deck_vel) / min(t_go, cfg.prediction_horizon)
        a_cmd += cfg.K_ff * deck_accel

        # Limit horizontal acceleration
        a_horiz = np.sqrt(a_cmd[0]**2 + a_cmd[1]**2)
        if a_horiz > cfg.max_accel:
            a_cmd[0] *= cfg.max_accel / a_horiz
            a_cmd[1] *= cfg.max_accel / a_horiz

        # Convert to attitude commands
        # For NED: positive pitch -> nose down -> forward acceleration
        # positive roll -> right wing down -> rightward acceleration
        yaw = quad_state.yaw

        # Rotate acceleration to body frame
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        a_forward = cos_yaw * a_cmd[0] + sin_yaw * a_cmd[1]
        a_right = -sin_yaw * a_cmd[0] + cos_yaw * a_cmd[1]

        # Desired pitch and roll (small angle approximation)
        pitch_cmd = np.arctan2(a_forward, self.g)
        roll_cmd = np.arctan2(-a_right, self.g)

        # Clamp
        pitch_cmd = np.clip(pitch_cmd, -cfg.max_pitch, cfg.max_pitch)
        roll_cmd = np.clip(roll_cmd, -cfg.max_roll, cfg.max_roll)

        # Thrust for vertical acceleration
        thrust = self.quad_params.mass * (self.g + a_cmd[2]) / (np.cos(pitch_cmd) * np.cos(roll_cmd))
        thrust = np.clip(thrust, 0, self.quad_params.mass * self.g * 2)

        # Yaw rate (point at deck)
        dx = deck_pos[0] - quad_state.pos[0]
        dy = deck_pos[1] - quad_state.pos[1]
        desired_yaw = np.arctan2(dy, dx)
        yaw_error = desired_yaw - yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # Wrap
        yaw_rate_cmd = np.clip(2.0 * yaw_error, -1.0, 1.0)

        return np.array([thrust, roll_cmd, pitch_cmd, yaw_rate_cmd])


class SimpleMPCSimulator:
    """Simulate landing with simplified MPC."""

    def __init__(self, sea_state_num: int = 4, ship_speed_kts: float = 15.0):
        self.ship_params = DDGParams()
        self.sea_state = SeaState.from_state_number(sea_state_num, direction=45.0)
        self.ship_sim = DDGMotionSimulator(self.ship_params, self.sea_state, ship_speed_kts)

        self.quad_params = QuadrotorParams()
        self.config = SimpleMPCConfig()
        self.controller = SimpleMPCController(self.config, self.quad_params)

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

        self.controller.fit_predictor(np.array(history), 0.1)

        # Initialize quad
        ship_motion = self.ship_sim.get_motion(30.0)
        deck_pos = ship_motion['deck_position']
        deck_vel = ship_motion['deck_velocity']

        quad_start_pos = deck_pos.copy()
        quad_start_pos[0] -= 80  # Behind ship
        quad_start_pos[2] = -25  # Above deck

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
            self.controller.update_predictor(obs)

            # Compute control
            time_to_touchdown = approach_time - t
            control = self.controller.compute_control(
                quad_state, deck_pos, deck_vel, time_to_touchdown
            )

            # Apply control
            thrust = control[0]
            roll_cmd = control[1]
            pitch_cmd = control[2]
            yaw_rate_cmd = control[3]

            # Simple attitude dynamics
            tau = 0.1
            roll_new = quad_state.roll + (roll_cmd - quad_state.roll) * self.dt / tau
            pitch_new = quad_state.pitch + (pitch_cmd - quad_state.pitch) * self.dt / tau
            yaw_new = quad_state.yaw + yaw_rate_cmd * self.dt

            quad_state.quat = euler_to_quat(roll_new, pitch_new, yaw_new)

            # Acceleration
            g = 9.81
            m = self.quad_params.mass
            ax = thrust / m * (np.sin(yaw_new) * np.sin(roll_new) +
                              np.cos(yaw_new) * np.sin(pitch_new) * np.cos(roll_new))
            ay = thrust / m * (-np.cos(yaw_new) * np.sin(roll_new) +
                              np.sin(yaw_new) * np.sin(pitch_new) * np.cos(roll_new))
            az = thrust / m * np.cos(pitch_new) * np.cos(roll_new) - g

            # Update state
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
    """Evaluate Simple MPC landing."""
    print("="*70)
    print("SIMPLE MPC (ZEM/ZEV) LANDING EVALUATION")
    print("="*70)

    results = {'pos_error': [], 'vel_error': [], 'success': []}

    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        np.random.seed(trial * 100)

        sim = SimpleMPCSimulator(sea_state_num=4)
        traj = sim.run_landing(approach_time=60.0)

        if len(traj['time']) > 0:
            final_pos_err = np.linalg.norm(traj['pos_error'][-1])
            final_vel_err = np.linalg.norm(traj['quad_vel'][-1] - traj['deck_vel'][-1])

            results['pos_error'].append(final_pos_err)
            results['vel_error'].append(final_vel_err)
            results['success'].append(final_pos_err < 3.0 and final_vel_err < 2.0)

    print("\n" + "="*70)
    print("SIMPLE MPC EVALUATION SUMMARY")
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
