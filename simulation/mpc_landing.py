#!/usr/bin/env python3
"""
MPC-based shipboard landing controller.

Uses Model Predictive Control with:
- Quadrotor dynamics prediction
- Deck motion forecasting (ARMA or Higher-Order)
- Receding horizon optimization
- Glide slope / approach cone constraints
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Tuple, Optional
import scipy.optimize as opt

from quad_dynamics.quadrotor import QuadrotorParams, QuadrotorState, QuadrotorDynamics
from ship_motion.ddg_motion import DDGParams, SeaState, DDGMotionSimulator, ARMAPredictor


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles to quaternion [qw, qx, qy, qz]."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])


@dataclass
class MPCConfig:
    """MPC controller configuration."""
    horizon_steps: int = 10        # Prediction horizon (steps)
    dt: float = 0.5                # MPC timestep (s)
    control_dt: float = 0.1        # Control loop rate (s)

    # Cost weights
    Q_pos: float = 50.0            # Position tracking weight (increased)
    Q_vel: float = 20.0            # Velocity tracking weight (increased)
    Q_terminal: float = 200.0      # Terminal cost weight (increased)
    R_thrust: float = 0.001        # Thrust cost (reduced)
    R_attitude: float = 0.05       # Attitude cost (reduced)
    R_rate: float = 0.02           # Attitude rate cost (reduced)

    # Constraints
    max_thrust: float = 30.0       # N
    min_thrust: float = 0.0        # N
    max_roll: float = 0.5          # rad (~30 deg)
    max_pitch: float = 0.5         # rad
    max_rate: float = 2.0          # rad/s

    # Approach cone
    cone_half_angle: float = 20.0  # deg
    min_altitude_agl: float = 2.0  # m above deck


class MPCController:
    """
    Model Predictive Controller for shipboard landing.

    At each timestep:
    1. Forecast deck motion over horizon using ARMA
    2. Optimize control sequence to minimize tracking error
    3. Apply first control, repeat
    """

    def __init__(self, config: MPCConfig, quad_params: QuadrotorParams):
        self.config = config
        self.quad_params = quad_params
        self.dynamics = QuadrotorDynamics(quad_params)

        # ARMA predictor for deck motion
        self.arma = ARMAPredictor(ar_order=8, ma_order=4)
        self.arma_fitted = False

        # State dimensions
        self.n_states = 9  # pos (3) + vel (3) + attitude (3)
        self.n_controls = 4  # thrust + roll_cmd + pitch_cmd + yaw_rate_cmd

        # Optimization warm start
        self.u_prev = None

    def fit_predictor(self, history: np.ndarray, dt: float):
        """Fit ARMA predictor with historical deck data."""
        self.arma.fit(history, dt)
        self.arma_fitted = True

    def update_predictor(self, observation: np.ndarray):
        """Update ARMA predictor with new observation."""
        if self.arma_fitted:
            self.arma.update(observation)

    def predict_deck(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict deck position and velocity over horizon.

        Returns:
            deck_pos: (n_steps, 3) predicted deck positions
            deck_vel: (n_steps, 3) predicted deck velocities
        """
        if not self.arma_fitted:
            # Return zeros if not fitted
            return np.zeros((n_steps, 3)), np.zeros((n_steps, 3))

        # ARMA prediction
        preds = self.arma.predict(n_steps)

        # Extract deck z and velocity from ARMA
        # ARMA outputs: [deck_z, deck_vz, deck_vx, roll, pitch, roll_rate]
        deck_pos = np.zeros((n_steps, 3))
        deck_vel = np.zeros((n_steps, 3))

        for k in range(n_steps):
            deck_pos[k, 2] = preds[k, 0]  # deck z
            deck_vel[k, 0] = preds[k, 2]  # deck vx (ship forward)
            deck_vel[k, 2] = preds[k, 1]  # deck vz

        return deck_pos, deck_vel

    def compute_control(self, quad_state: QuadrotorState, deck_pos: np.ndarray,
                       deck_vel: np.ndarray, time_to_touchdown: float) -> np.ndarray:
        """
        Compute optimal control using MPC.

        Args:
            quad_state: Current quadrotor state
            deck_pos: Current deck position
            deck_vel: Current deck velocity
            time_to_touchdown: Estimated time to landing

        Returns:
            control: [thrust, roll_cmd, pitch_cmd, yaw_rate_cmd]
        """
        cfg = self.config
        N = cfg.horizon_steps
        dt = cfg.dt

        # Predict deck motion
        deck_pos_pred, deck_vel_pred = self.predict_deck(N)

        # Current deck as base (ARMA predicts relative motion)
        for k in range(N):
            deck_pos_pred[k] += deck_pos
            deck_vel_pred[k] += deck_vel

        # Reference trajectory: track predicted deck position
        ref_pos = np.zeros((N, 3))
        ref_vel = np.zeros((N, 3))

        # Current relative position
        rel_pos = quad_state.pos - deck_pos

        for k in range(N):
            t_k = k * dt
            alpha = min(t_k / max(time_to_touchdown, 1.0), 1.0)

            # Target deck position at this timestep
            target_deck = deck_pos_pred[min(k, N-1)]

            # Reference: reduce relative error to zero over time_to_touchdown
            ref_pos[k] = target_deck + (1 - alpha) * rel_pos

            # Velocity: match deck velocity with closing component
            closing_vel = -rel_pos / max(time_to_touchdown - t_k, 0.5)
            ref_vel[k] = deck_vel_pred[min(k, N-1)] + (1 - alpha) * closing_vel

        # Optimization
        # Decision variables: [u_0, u_1, ..., u_{N-1}] where u_k = [T, phi_cmd, theta_cmd, r_cmd]
        n_vars = N * self.n_controls

        # Initial guess (hover or warm start)
        if self.u_prev is not None:
            u0 = np.roll(self.u_prev, -self.n_controls)
            u0[-self.n_controls:] = u0[-2*self.n_controls:-self.n_controls]  # Repeat last
        else:
            u0 = np.zeros(n_vars)
            hover_thrust = self.quad_params.mass * 9.81
            for k in range(N):
                u0[k * self.n_controls] = hover_thrust

        # Bounds
        lb = np.zeros(n_vars)
        ub = np.zeros(n_vars)
        for k in range(N):
            idx = k * self.n_controls
            lb[idx] = cfg.min_thrust
            ub[idx] = cfg.max_thrust
            lb[idx + 1] = -cfg.max_roll
            ub[idx + 1] = cfg.max_roll
            lb[idx + 2] = -cfg.max_pitch
            ub[idx + 2] = cfg.max_pitch
            lb[idx + 3] = -cfg.max_rate
            ub[idx + 3] = cfg.max_rate

        bounds = opt.Bounds(lb, ub)

        # Cost function
        def cost(u):
            total_cost = 0.0

            # Simulate forward
            state = np.concatenate([quad_state.pos, quad_state.vel,
                                   [quad_state.roll, quad_state.pitch, quad_state.yaw]])

            for k in range(N):
                idx = k * self.n_controls
                thrust = u[idx]
                roll_cmd = u[idx + 1]
                pitch_cmd = u[idx + 2]
                yaw_rate_cmd = u[idx + 3]

                # Simple dynamics model
                # Attitude follows command with first-order lag
                tau_att = 0.1
                roll = state[6] + (roll_cmd - state[6]) * dt / tau_att
                pitch = state[7] + (pitch_cmd - state[7]) * dt / tau_att
                yaw = state[8] + yaw_rate_cmd * dt

                # Acceleration from thrust and attitude
                g = 9.81
                m = self.quad_params.mass
                ax = thrust / m * (np.sin(yaw) * np.sin(roll) + np.cos(yaw) * np.sin(pitch) * np.cos(roll))
                ay = thrust / m * (-np.cos(yaw) * np.sin(roll) + np.sin(yaw) * np.sin(pitch) * np.cos(roll))
                az = thrust / m * np.cos(pitch) * np.cos(roll) - g

                # Update velocity
                vx = state[3] + ax * dt
                vy = state[4] + ay * dt
                vz = state[5] + az * dt

                # Update position
                px = state[0] + state[3] * dt + 0.5 * ax * dt**2
                py = state[1] + state[4] * dt + 0.5 * ay * dt**2
                pz = state[2] + state[5] * dt + 0.5 * az * dt**2

                state = np.array([px, py, pz, vx, vy, vz, roll, pitch, yaw])

                # Position cost
                pos_err = state[:3] - ref_pos[k]
                total_cost += cfg.Q_pos * np.sum(pos_err**2)

                # Velocity cost
                vel_err = state[3:6] - ref_vel[k]
                total_cost += cfg.Q_vel * np.sum(vel_err**2)

                # Control cost
                total_cost += cfg.R_thrust * (thrust - self.quad_params.mass * g)**2
                total_cost += cfg.R_attitude * (roll_cmd**2 + pitch_cmd**2)
                total_cost += cfg.R_rate * yaw_rate_cmd**2

            # Terminal cost (match deck at end)
            terminal_pos_err = state[:3] - deck_pos_pred[-1]
            terminal_vel_err = state[3:6] - deck_vel_pred[-1]
            total_cost += cfg.Q_terminal * (np.sum(terminal_pos_err**2) + np.sum(terminal_vel_err**2))

            return total_cost

        # Optimize (fast SLSQP with limited iterations)
        result = opt.minimize(cost, u0, method='SLSQP', bounds=bounds,
                             options={'maxiter': 10, 'disp': False, 'ftol': 1e-3})

        self.u_prev = result.x

        # Extract first control
        control = result.x[:self.n_controls]
        return control


class MPCLandingSimulator:
    """Simulate MPC-controlled shipboard landing."""

    def __init__(self, sea_state_num: int = 4, ship_speed_kts: float = 15.0):
        # Ship setup
        self.ship_params = DDGParams()
        self.sea_state = SeaState.from_state_number(sea_state_num, direction=45.0)
        self.ship_sim = DDGMotionSimulator(self.ship_params, self.sea_state, ship_speed_kts)

        # Quad setup
        self.quad_params = QuadrotorParams()
        self.dynamics = QuadrotorDynamics(self.quad_params)

        # MPC setup
        self.config = MPCConfig()
        self.mpc = MPCController(self.config, self.quad_params)

        # Timing
        self.dt = 0.02  # Simulation timestep
        self.t = 0.0

    def run_landing(self, approach_time: float = 60.0) -> dict:
        """
        Run landing simulation.

        Args:
            approach_time: Time from start to planned touchdown

        Returns:
            Dictionary with trajectory and results
        """
        # Collect warmup data for ARMA (before initializing quad)
        # This simulates collecting ship motion history
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

        self.mpc.fit_predictor(np.array(history), 0.1)

        # Initialize quad behind and above ship at t=30
        ship_motion_start = self.ship_sim.get_motion(30.0)
        deck_pos_start = ship_motion_start['deck_position']
        deck_vel_start = ship_motion_start['deck_velocity']

        quad_start_pos = deck_pos_start.copy()
        quad_start_pos[0] -= 80  # 80m behind
        quad_start_pos[2] = -25   # 25m altitude (up is negative in NED)

        self.quad_state = QuadrotorState(
            pos=quad_start_pos,
            vel=deck_vel_start.copy(),  # Match ship velocity
            quat=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
            omega=np.zeros(3)
        )

        # Run simulation
        print("Running MPC landing simulation...")
        trajectory = {
            'time': [],
            'quad_pos': [],
            'quad_vel': [],
            'deck_pos': [],
            'deck_vel': [],
            'control': [],
            'pos_error': [],
        }

        t = 30.0  # Start after warmup
        last_mpc_t = 0
        mpc_interval = self.config.control_dt

        while t < approach_time:
            # Get current deck state
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
            self.mpc.update_predictor(obs)

            # Compute MPC control
            time_to_touchdown = approach_time - t
            if t - last_mpc_t >= mpc_interval:
                control = self.mpc.compute_control(
                    self.quad_state, deck_pos, deck_vel, time_to_touchdown
                )
                last_mpc_t = t

            # Apply control
            thrust = control[0]
            roll_cmd = control[1]
            pitch_cmd = control[2]
            yaw_rate_cmd = control[3]

            # Get current Euler angles from quaternion
            roll_curr = self.quad_state.roll
            pitch_curr = self.quad_state.pitch
            yaw_curr = self.quad_state.yaw

            # Simple attitude tracking
            tau = 0.1
            roll_new = roll_curr + (roll_cmd - roll_curr) * self.dt / tau
            pitch_new = pitch_curr + (pitch_cmd - pitch_curr) * self.dt / tau
            yaw_new = yaw_curr + yaw_rate_cmd * self.dt

            # Convert Euler back to quaternion
            self.quad_state.quat = euler_to_quat(roll_new, pitch_new, yaw_new)

            # Compute acceleration
            g = 9.81
            m = self.quad_params.mass
            roll, pitch, yaw = roll_new, pitch_new, yaw_new

            ax = thrust / m * (np.sin(yaw) * np.sin(roll) + np.cos(yaw) * np.sin(pitch) * np.cos(roll))
            ay = thrust / m * (-np.cos(yaw) * np.sin(roll) + np.sin(yaw) * np.sin(pitch) * np.cos(roll))
            az = thrust / m * np.cos(pitch) * np.cos(roll) - g

            # Update velocity and position
            self.quad_state.vel += np.array([ax, ay, az]) * self.dt
            self.quad_state.pos += self.quad_state.vel * self.dt

            # Record
            pos_error = self.quad_state.pos - deck_pos
            trajectory['time'].append(t)
            trajectory['quad_pos'].append(self.quad_state.pos.copy())
            trajectory['quad_vel'].append(self.quad_state.vel.copy())
            trajectory['deck_pos'].append(deck_pos.copy())
            trajectory['deck_vel'].append(deck_vel.copy())
            trajectory['control'].append(control.copy())
            trajectory['pos_error'].append(pos_error.copy())

            # Check touchdown
            altitude_agl = -(self.quad_state.pos[2] - deck_pos[2])  # NED: up is negative
            if altitude_agl < 0.5:
                print(f"Touchdown at t={t:.2f}s")
                break

            t += self.dt

        # Convert to arrays
        for key in trajectory:
            trajectory[key] = np.array(trajectory[key])

        # Compute metrics
        final_pos_err = trajectory['pos_error'][-1]
        final_vel_err = trajectory['quad_vel'][-1] - trajectory['deck_vel'][-1]

        print("\nLanding Results:")
        print(f"  Position error: X={final_pos_err[0]:.3f}m, Y={final_pos_err[1]:.3f}m, Z={final_pos_err[2]:.3f}m")
        print(f"  Velocity error: X={final_vel_err[0]:.3f}m/s, Y={final_vel_err[1]:.3f}m/s, Z={final_vel_err[2]:.3f}m/s")
        print(f"  Total pos error: {np.linalg.norm(final_pos_err):.3f}m")
        print(f"  Total vel error: {np.linalg.norm(final_vel_err):.3f}m/s")

        return trajectory


def run_mpc_evaluation(n_trials: int = 5):
    """Evaluate MPC landing across multiple trials."""
    print("="*70)
    print("MPC LANDING EVALUATION")
    print("="*70)

    results = {
        'pos_error': [],
        'vel_error': [],
        'success': [],
    }

    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        np.random.seed(trial * 100)

        sim = MPCLandingSimulator(sea_state_num=4)
        traj = sim.run_landing(approach_time=60.0)

        if len(traj['time']) > 0:
            final_pos_err = np.linalg.norm(traj['pos_error'][-1])
            final_vel_err = np.linalg.norm(traj['quad_vel'][-1] - traj['deck_vel'][-1])

            results['pos_error'].append(final_pos_err)
            results['vel_error'].append(final_vel_err)
            results['success'].append(final_pos_err < 3.0 and final_vel_err < 2.0)

    print("\n" + "="*70)
    print("MPC EVALUATION SUMMARY")
    print("="*70)
    print(f"Trials: {n_trials}")
    print(f"Success rate: {100 * np.mean(results['success']):.1f}%")
    print(f"Mean position error: {np.mean(results['pos_error']):.3f}m")
    print(f"Mean velocity error: {np.mean(results['vel_error']):.3f}m/s")
    print(f"Position error std: {np.std(results['pos_error']):.3f}m")
    print(f"Velocity error std: {np.std(results['vel_error']):.3f}m/s")

    return results


if __name__ == "__main__":
    run_mpc_evaluation(n_trials=5)
