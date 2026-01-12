#!/usr/bin/env python3
"""
Variable Horizon MPC (VH-MPC) for Shipboard Landing

Based on Penn State research (Ngo & Sultan, AIAA SciTech 2024):
"Robust Variable Horizon MPC with Move Blocking for Helicopter
Shipboard Landing on Moving Decks"

Key innovations:
1. Evaluates multiple horizon lengths in parallel
2. Selects horizon with lowest cost (implicit touchdown time optimization)
3. Move blocking reduces decision variables
4. Parallel computation - latency = max(solve_times), not sum

Theory:
- Standard MPC: fixed horizon N, may miss optimal touchdown window
- VH-MPC: horizons N1, N2, ..., Nk evaluated in parallel
- Select: argmin_k J(N_k) where J includes terminal matching cost
- Move blocking: u[0:3] = u_a, u[3:6] = u_b, etc.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quad_dynamics.quadrotor import QuadrotorParams


@dataclass
class VHMPCConfig:
    """Configuration for Variable Horizon MPC."""
    # Horizon options (in time steps)
    horizon_options: List[int] = None  # Will be set in __post_init__
    dt: float = 0.1                    # Time step for MPC

    # Move blocking pattern [3, 3, 2] means:
    # - First 3 steps use same control
    # - Next 3 steps use second control
    # - Last 2 steps use third control
    # This reduces 8 decision variables to 3
    move_blocks: List[int] = None      # Will be set in __post_init__

    # Cost weights
    Q_pos: float = 100.0               # Position tracking weight
    Q_vel: float = 50.0                # Velocity tracking weight
    Q_att: float = 20.0                # Attitude weight
    R_thrust: float = 0.001            # Thrust cost
    R_torque: float = 0.01             # Torque cost
    Qf_pos: float = 500.0              # Terminal position weight
    Qf_vel: float = 200.0              # Terminal velocity weight

    # Constraints
    max_accel: float = 6.0             # Max acceleration (m/s²)
    max_tilt: float = 0.5              # Max roll/pitch (rad)

    # Parallel execution
    n_threads: int = 4                 # Parallel solver threads

    def __post_init__(self):
        if self.horizon_options is None:
            self.horizon_options = [20, 30, 40, 50]  # 2s, 3s, 4s, 5s at dt=0.1
        if self.move_blocks is None:
            self.move_blocks = [5, 5, 5, 5]  # 4 control blocks


class VHMPCSolver:
    """
    Single-horizon MPC solver for quadrotor landing.

    Uses simplified dynamics for fast solving:
    - Point mass + first-order attitude dynamics
    - QP formulation with linearized constraints
    """

    def __init__(self, params: QuadrotorParams, config: VHMPCConfig):
        self.params = params
        self.config = config
        self.g = 9.81

    def solve(self, x0: np.ndarray, deck_trajectory: List[dict],
              horizon: int) -> Tuple[np.ndarray, float, bool]:
        """
        Solve MPC for given horizon.

        Args:
            x0: Initial state [pos(3), vel(3), att(3)]
            deck_trajectory: List of {pos, vel, att} dicts at each timestep
            horizon: Number of steps

        Returns:
            u_opt: Optimal control sequence [horizon, 3] (ax, ay, az)
            cost: Total cost
            success: Whether solver succeeded
        """
        cfg = self.config
        dt = cfg.dt

        # Expand move blocks to horizon
        control_blocks = self._expand_blocks(horizon)
        n_blocks = len(set(control_blocks))

        # Decision variables: acceleration for each block [n_blocks, 3]
        # Initialize with ZEM/ZEV
        u_init = self._initialize_control(x0, deck_trajectory, horizon, n_blocks, control_blocks)

        # Simple gradient descent optimization
        u_opt = u_init.copy()
        best_cost = float('inf')
        best_u = u_opt.copy()

        for iteration in range(10):
            # Evaluate cost and gradient
            cost, grad = self._cost_and_gradient(x0, deck_trajectory, u_opt, horizon, control_blocks)

            if cost < best_cost:
                best_cost = cost
                best_u = u_opt.copy()

            # Gradient step
            step_size = 0.1 / (1 + iteration * 0.2)
            u_opt -= step_size * grad

            # Project onto constraints
            u_opt = self._project_constraints(u_opt)

        # Expand to full control sequence
        u_full = np.zeros((horizon, 3))
        for t in range(horizon):
            block_idx = control_blocks[t]
            u_full[t] = best_u[block_idx]

        return u_full, best_cost, True

    def _expand_blocks(self, horizon: int) -> List[int]:
        """Map timesteps to control blocks."""
        blocks = self.config.move_blocks
        result = []
        block_idx = 0
        steps_in_block = 0

        for t in range(horizon):
            if block_idx < len(blocks):
                result.append(block_idx)
                steps_in_block += 1
                if steps_in_block >= blocks[block_idx]:
                    block_idx += 1
                    steps_in_block = 0
            else:
                result.append(len(blocks) - 1)

        return result

    def _initialize_control(self, x0: np.ndarray, deck_traj: List[dict],
                           horizon: int, n_blocks: int,
                           control_blocks: List[int]) -> np.ndarray:
        """Initialize control using ZEM/ZEV."""
        cfg = self.config
        dt = cfg.dt

        u_init = np.zeros((n_blocks, 3))

        # Use ZEM/ZEV for initialization
        pos = x0[0:3]
        vel = x0[3:6]

        for b in range(n_blocks):
            # Find representative timestep for this block
            block_times = [t for t, bi in enumerate(control_blocks) if bi == b]
            t_mid = block_times[len(block_times) // 2]

            if t_mid < len(deck_traj):
                deck = deck_traj[t_mid]
                t_go = (horizon - t_mid) * dt

                rel_pos = pos - deck['pos']
                rel_vel = vel - deck['vel']

                # ZEM/ZEV
                t_go = max(t_go, 1.0)
                a_zem = -6.0 * (rel_pos + rel_vel * t_go) / t_go**2
                a_zev = -2.0 * rel_vel / t_go

                u_init[b] = np.clip(a_zem + a_zev, -cfg.max_accel, cfg.max_accel)

        return u_init

    def _cost_and_gradient(self, x0: np.ndarray, deck_traj: List[dict],
                           u_blocks: np.ndarray, horizon: int,
                           control_blocks: List[int]) -> Tuple[float, np.ndarray]:
        """Compute cost and gradient using forward simulation."""
        cfg = self.config
        dt = cfg.dt
        m = self.params.mass

        # Forward simulate
        x = x0.copy()
        cost = 0.0
        grad = np.zeros_like(u_blocks)

        for t in range(horizon):
            block_idx = control_blocks[t]
            a_cmd = u_blocks[block_idx]

            if t < len(deck_traj):
                deck = deck_traj[t]

                # Running cost: tracking error
                pos_err = x[0:3] - deck['pos']
                vel_err = x[3:6] - deck['vel']

                cost += cfg.Q_pos * np.dot(pos_err, pos_err)
                cost += cfg.Q_vel * np.dot(vel_err, vel_err)
                cost += cfg.R_thrust * np.dot(a_cmd, a_cmd)

                # Gradient contribution
                grad[block_idx] += 2 * cfg.R_thrust * a_cmd
                # Approximate gradient through dynamics (simplified)
                grad[block_idx] += 2 * cfg.Q_vel * vel_err * dt

            # Dynamics update (simplified point mass)
            x[3:6] += a_cmd * dt
            x[0:3] += x[3:6] * dt

        # Terminal cost
        if len(deck_traj) > 0:
            deck_f = deck_traj[-1]
            pos_err_f = x[0:3] - deck_f['pos']
            vel_err_f = x[3:6] - deck_f['vel']

            cost += cfg.Qf_pos * np.dot(pos_err_f, pos_err_f)
            cost += cfg.Qf_vel * np.dot(vel_err_f, vel_err_f)

        return cost, grad

    def _project_constraints(self, u: np.ndarray) -> np.ndarray:
        """Project control onto feasible set."""
        cfg = self.config
        return np.clip(u, -cfg.max_accel, cfg.max_accel)


class VariableHorizonMPC:
    """
    Variable Horizon MPC Controller.

    Evaluates multiple horizons in parallel, selects best.
    """

    def __init__(self, params: QuadrotorParams = None, config: VHMPCConfig = None):
        self.params = params if params is not None else QuadrotorParams()
        self.config = config if config is not None else VHMPCConfig()
        self.g = 9.81

        # Create solver for each horizon
        self.solver = VHMPCSolver(self.params, self.config)

        # State
        self.last_solution = None
        self.last_horizon = None

    def compute_control(self, x: np.ndarray, deck_pos: np.ndarray,
                       deck_vel: np.ndarray, deck_att: np.ndarray,
                       t_go: float, arma_predict: callable = None) -> Tuple[np.ndarray, dict]:
        """
        Compute optimal control using VH-MPC.

        Args:
            x: Current state [pos(3), vel(3), att(3)]
            deck_pos: Current deck position
            deck_vel: Current deck velocity
            deck_att: Current deck attitude
            t_go: Approximate time to go
            arma_predict: Optional ARMA prediction function

        Returns:
            acc_cmd: Acceleration command [3]
            info: Debug info
        """
        cfg = self.config

        # Build deck trajectory predictions
        deck_traj = self._predict_deck_trajectory(
            deck_pos, deck_vel, deck_att,
            max(cfg.horizon_options) * cfg.dt,
            arma_predict
        )

        # Solve for each horizon in parallel
        results = []

        def solve_horizon(horizon):
            if horizon * cfg.dt > t_go + 1.0:
                # Skip very long horizons when close to landing
                return None, float('inf'), False
            return self.solver.solve(x, deck_traj, horizon)

        # Parallel execution
        with ThreadPoolExecutor(max_workers=cfg.n_threads) as executor:
            futures = {executor.submit(solve_horizon, h): h
                      for h in cfg.horizon_options}
            for future in futures:
                horizon = futures[future]
                try:
                    u_opt, cost, success = future.result()
                    if success:
                        results.append((horizon, u_opt, cost))
                except Exception:
                    pass

        if not results:
            # Fallback to ZEM/ZEV
            return self._fallback_control(x, deck_pos, deck_vel, t_go), {'horizon': 0, 'cost': float('inf')}

        # Select best horizon
        best = min(results, key=lambda r: r[2])
        best_horizon, best_u, best_cost = best

        self.last_solution = best_u
        self.last_horizon = best_horizon

        # Return first control (acceleration command)
        acc_cmd = best_u[0]

        info = {
            'horizon': best_horizon,
            'cost': best_cost,
            'horizons_evaluated': len(results),
            'all_costs': {r[0]: r[2] for r in results}
        }

        return acc_cmd, info

    def _predict_deck_trajectory(self, deck_pos: np.ndarray, deck_vel: np.ndarray,
                                  deck_att: np.ndarray, horizon_time: float,
                                  arma_predict: callable = None) -> List[dict]:
        """Predict deck trajectory over horizon."""
        cfg = self.config
        n_steps = int(horizon_time / cfg.dt) + 1

        trajectory = []

        for i in range(n_steps):
            t = i * cfg.dt

            if arma_predict is not None:
                # Use ARMA prediction
                try:
                    pred = arma_predict(int(t / 0.1) + 1)
                    if pred is not None and len(pred) > 0:
                        pred_z = pred[-1][0]
                        pred_vz = pred[-1][1]
                    else:
                        pred_z = deck_pos[2]
                        pred_vz = deck_vel[2]
                except:
                    pred_z = deck_pos[2]
                    pred_vz = deck_vel[2]
            else:
                pred_z = deck_pos[2]
                pred_vz = deck_vel[2]

            pred_pos = deck_pos.copy()
            pred_pos[0] += deck_vel[0] * t
            pred_pos[2] = pred_z

            pred_vel = deck_vel.copy()
            pred_vel[2] = pred_vz

            trajectory.append({
                'pos': pred_pos,
                'vel': pred_vel,
                'att': deck_att.copy()
            })

        return trajectory

    def _fallback_control(self, x: np.ndarray, deck_pos: np.ndarray,
                          deck_vel: np.ndarray, t_go: float) -> np.ndarray:
        """Fallback ZEM/ZEV control."""
        rel_pos = x[0:3] - deck_pos
        rel_vel = x[3:6] - deck_vel
        t_go = max(t_go, 1.0)

        a_zem = -6.0 * (rel_pos + rel_vel * t_go) / t_go**2
        a_zev = -2.0 * rel_vel / t_go

        return np.clip(a_zem + a_zev, -self.config.max_accel, self.config.max_accel)

    def acc_to_thrust_attitude(self, acc_cmd: np.ndarray, yaw: float,
                               deck_att: np.ndarray, height: float) -> Tuple[float, float, float]:
        """Convert acceleration to thrust/attitude commands."""
        m = self.params.mass
        cfg = self.config

        # Thrust magnitude
        thrust_vec = m * (acc_cmd - np.array([0, 0, self.g]))
        thrust = np.linalg.norm(thrust_vec)
        thrust = np.clip(thrust, 0.2 * m * self.g, 1.5 * m * self.g)

        # Attitude from acceleration
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        a_forward = cos_yaw * acc_cmd[0] + sin_yaw * acc_cmd[1]
        a_right = -sin_yaw * acc_cmd[0] + cos_yaw * acc_cmd[1]

        pitch_cmd = np.arctan2(a_forward, self.g)
        roll_cmd = np.arctan2(-a_right, self.g)

        # Blend to deck attitude near landing
        if height < 3.0 and height > 0:
            blend = 1 - height / 3.0
            roll_cmd = (1 - blend) * roll_cmd + blend * deck_att[0]
            pitch_cmd = (1 - blend) * pitch_cmd + blend * deck_att[1]

        roll_cmd = np.clip(roll_cmd, -cfg.max_tilt, cfg.max_tilt)
        pitch_cmd = np.clip(pitch_cmd, -cfg.max_tilt, cfg.max_tilt)

        return thrust, roll_cmd, pitch_cmd


def demo():
    """Demonstrate VH-MPC."""
    print("Variable Horizon MPC Demo")
    print("=" * 50)

    params = QuadrotorParams()
    config = VHMPCConfig(
        horizon_options=[20, 30, 40],
        move_blocks=[5, 5, 5, 5]
    )

    controller = VariableHorizonMPC(params, config)

    # Test scenario
    x = np.array([-30, 0, -20, 5, 0, 1, 0, 0, 0])  # pos, vel, att
    deck_pos = np.array([0, 0, -8])
    deck_vel = np.array([7.7, 0, 0])
    deck_att = np.array([0.05, 0.02, 0])

    acc_cmd, info = controller.compute_control(
        x, deck_pos, deck_vel, deck_att, t_go=10.0
    )

    print(f"\nScenario:")
    print(f"  UAV: pos={x[0:3]}, vel={x[3:6]}")
    print(f"  Deck: pos={deck_pos}, vel={deck_vel}")

    print(f"\nVH-MPC Results:")
    print(f"  Selected horizon: {info['horizon']} steps ({info['horizon'] * config.dt:.1f}s)")
    print(f"  Cost: {info['cost']:.2f}")
    print(f"  Horizons evaluated: {info['horizons_evaluated']}")
    print(f"  Acceleration cmd: [{acc_cmd[0]:.2f}, {acc_cmd[1]:.2f}, {acc_cmd[2]:.2f}] m/s²")

    thrust, roll, pitch = controller.acc_to_thrust_attitude(
        acc_cmd, 0.0, deck_att, 12.0
    )
    print(f"\nThrust/Attitude:")
    print(f"  Thrust: {thrust:.1f} N")
    print(f"  Roll: {np.degrees(roll):.1f}°")
    print(f"  Pitch: {np.degrees(pitch):.1f}°")


if __name__ == "__main__":
    demo()
