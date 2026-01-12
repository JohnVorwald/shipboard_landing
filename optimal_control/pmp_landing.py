#!/usr/bin/env python3
"""
PMP-Based Landing Controller with Enhanced Matching Criteria

Extends standard PMP trajectory tracking with terminal constraints for:
1. Position matching - Zero position error at touchdown
2. Velocity matching - Zero relative velocity at touchdown
3. Attitude matching - Match deck roll/pitch at touchdown
4. Impact velocity limit - Soft landing with bounded descent rate
5. Lateral drift limit - No sideways motion at touchdown

Formulation:
    min J = ∫(x'Qx + u'Ru)dt + (x_f - x_deck)'Qf(x_f - x_deck) + penalty terms

Terminal constraints (soft):
    ||p_rel(tf)|| < ε_pos           (position matching)
    ||v_rel(tf)|| < ε_vel           (velocity matching)
    ||[φ,θ](tf) - [φ,θ]_deck|| < ε_att  (attitude matching)
    v_z_rel(tf) > -v_impact_max      (soft landing)
    ||v_xy_rel(tf)|| < v_lateral_max  (no drift)

Reference: Combines approaches from:
- Penn State EiLQR (Pravitra 2021)
- RPI shrinking horizon MPC
- Differential flatness trajectory planning
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quad_dynamics.quadrotor import QuadrotorParams, QuadrotorState


@dataclass
class LandingMatchingCriteria:
    """Terminal constraints for landing matching."""
    # Position matching
    pos_weight: float = 200.0       # Terminal position weight
    pos_tolerance: float = 0.5      # m - acceptable position error

    # Velocity matching
    vel_weight: float = 150.0       # Terminal velocity weight
    vel_tolerance: float = 0.3      # m/s - acceptable velocity error

    # Attitude matching
    att_weight: float = 50.0        # Terminal attitude weight
    att_tolerance: float = 0.05     # rad (~3 deg) - acceptable attitude error

    # Impact velocity constraint
    impact_vel_max: float = 1.0     # m/s - max descent rate at touchdown
    impact_penalty: float = 500.0   # Penalty for exceeding impact velocity

    # Lateral drift constraint
    lateral_vel_max: float = 0.5    # m/s - max lateral velocity at touchdown
    lateral_penalty: float = 300.0  # Penalty for lateral drift

    # Attitude rate constraint (for smooth touchdown)
    att_rate_max: float = 0.2       # rad/s - max roll/pitch rate at touchdown
    att_rate_penalty: float = 100.0


@dataclass
class PMPLandingConfig:
    """Configuration for PMP landing controller."""
    # Trajectory
    replan_interval: float = 0.5    # Replan every 0.5s
    prediction_horizon: float = 5.0  # Forward prediction for deck motion

    # Control gains (outer loop)
    Kp_pos: np.ndarray = None
    Kd_pos: np.ndarray = None

    # Attitude gains (inner loop)
    Kp_att: np.ndarray = None
    Kd_att: np.ndarray = None

    # Costate feedback
    costate_gain: float = 0.15

    # Matching criteria
    matching: LandingMatchingCriteria = None

    def __post_init__(self):
        if self.Kp_pos is None:
            self.Kp_pos = np.array([2.5, 2.5, 4.0])
        if self.Kd_pos is None:
            self.Kd_pos = np.array([3.5, 3.5, 5.0])
        if self.Kp_att is None:
            self.Kp_att = np.array([35.0, 35.0, 20.0])
        if self.Kd_att is None:
            self.Kd_att = np.array([10.0, 10.0, 6.0])
        if self.matching is None:
            self.matching = LandingMatchingCriteria()


class PMPLandingController:
    """
    PMP-based landing controller with matching criteria.

    Uses optimal control theory with terminal constraints for
    precise shipboard landing.
    """

    def __init__(self, params: QuadrotorParams = None, config: PMPLandingConfig = None):
        self.params = params if params is not None else QuadrotorParams()
        self.config = config if config is not None else PMPLandingConfig()
        self.g = 9.81

        # Trajectory state
        self.traj_time = None
        self.traj_states = None
        self.traj_costates = None
        self.traj_start = 0.0

        # Deck prediction
        self.deck_history = []

    def compute_terminal_cost(self, x_final: np.ndarray, deck_state: np.ndarray) -> float:
        """
        Compute terminal cost with matching criteria penalties.

        Args:
            x_final: Final state [pos, vel, att, omega]
            deck_state: Deck state [pos, vel, att, omega]

        Returns:
            Terminal cost value
        """
        cfg = self.config.matching

        # Relative states
        rel_pos = x_final[0:3] - deck_state[0:3]
        rel_vel = x_final[3:6] - deck_state[3:6]
        rel_att = x_final[6:9] - deck_state[6:9]
        rel_omega = x_final[9:12] - deck_state[9:12]

        cost = 0.0

        # Position matching (quadratic)
        pos_err = np.linalg.norm(rel_pos)
        cost += cfg.pos_weight * pos_err**2

        # Velocity matching (quadratic)
        vel_err = np.linalg.norm(rel_vel)
        cost += cfg.vel_weight * vel_err**2

        # Attitude matching (roll/pitch only)
        att_err = np.linalg.norm(rel_att[:2])
        cost += cfg.att_weight * att_err**2

        # Impact velocity penalty (barrier-like for descent rate)
        # rel_vel[2] is positive when UAV descending faster than deck (NED)
        if rel_vel[2] > cfg.impact_vel_max:
            excess = rel_vel[2] - cfg.impact_vel_max
            cost += cfg.impact_penalty * excess**2

        # Lateral drift penalty
        lateral_vel = np.linalg.norm(rel_vel[:2])
        if lateral_vel > cfg.lateral_vel_max:
            excess = lateral_vel - cfg.lateral_vel_max
            cost += cfg.lateral_penalty * excess**2

        # Attitude rate penalty
        att_rate = np.linalg.norm(rel_omega[:2])
        if att_rate > cfg.att_rate_max:
            excess = att_rate - cfg.att_rate_max
            cost += cfg.att_rate_penalty * excess**2

        return cost

    def compute_terminal_costate(self, x_final: np.ndarray, deck_state: np.ndarray) -> np.ndarray:
        """
        Compute terminal costate (gradient of terminal cost).

        λ(tf) = ∂Φ/∂x evaluated at x_final

        Returns:
            Terminal costate [12]
        """
        cfg = self.config.matching

        rel_pos = x_final[0:3] - deck_state[0:3]
        rel_vel = x_final[3:6] - deck_state[3:6]
        rel_att = x_final[6:9] - deck_state[6:9]
        rel_omega = x_final[9:12] - deck_state[9:12]

        lam = np.zeros(12)

        # Position gradient
        lam[0:3] = 2 * cfg.pos_weight * rel_pos

        # Velocity gradient
        lam[3:6] = 2 * cfg.vel_weight * rel_vel

        # Add impact velocity penalty gradient
        if rel_vel[2] > cfg.impact_vel_max:
            excess = rel_vel[2] - cfg.impact_vel_max
            lam[5] += 2 * cfg.impact_penalty * excess

        # Add lateral velocity penalty gradient
        lateral_vel = np.linalg.norm(rel_vel[:2])
        if lateral_vel > cfg.lateral_vel_max:
            excess = lateral_vel - cfg.lateral_vel_max
            if lateral_vel > 1e-6:
                lam[3] += 2 * cfg.lateral_penalty * excess * rel_vel[0] / lateral_vel
                lam[4] += 2 * cfg.lateral_penalty * excess * rel_vel[1] / lateral_vel

        # Attitude gradient (roll/pitch only)
        lam[6:8] = 2 * cfg.att_weight * rel_att[:2]

        # Attitude rate gradient
        att_rate = np.linalg.norm(rel_omega[:2])
        if att_rate > cfg.att_rate_max:
            excess = att_rate - cfg.att_rate_max
            if att_rate > 1e-6:
                lam[9] += 2 * cfg.att_rate_penalty * excess * rel_omega[0] / att_rate
                lam[10] += 2 * cfg.att_rate_penalty * excess * rel_omega[1] / att_rate

        return lam

    def plan_trajectory(self, x0: np.ndarray, deck_pos: np.ndarray, deck_vel: np.ndarray,
                       deck_att: np.ndarray, time_to_go: float, current_time: float):
        """
        Plan optimal trajectory using simplified dynamics and matching criteria.

        Uses polynomial trajectory with terminal constraints.
        """
        T = max(time_to_go, 0.5)
        N = max(10, int(T / 0.1))

        # Target deck state at touchdown
        deck_state = np.zeros(12)
        deck_state[0:3] = deck_pos + deck_vel * T  # Predict deck position
        deck_state[3:6] = deck_vel
        deck_state[6:9] = deck_att

        # Generate trajectory satisfying boundary conditions
        t = np.linspace(0, T, N)
        x_traj = np.zeros((N, 12))
        lam_traj = np.zeros((N, 12))

        # Polynomial coefficients for smooth trajectory
        # x(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
        # Boundary conditions: x(0), v(0), a(0), x(T), v(T), a(T)

        for axis in range(3):
            # Position
            x0_ax = x0[axis]
            v0_ax = x0[3 + axis]
            xf_ax = deck_state[axis]
            vf_ax = deck_state[3 + axis]

            # Assume zero acceleration at boundaries for smoothness
            a0_ax = 0
            af_ax = 0

            # Quintic polynomial coefficients
            coeffs = self._quintic_coeffs(x0_ax, v0_ax, a0_ax, xf_ax, vf_ax, af_ax, T)

            for i, ti in enumerate(t):
                x_traj[i, axis] = self._eval_poly(coeffs, ti)
                x_traj[i, 3 + axis] = self._eval_poly_deriv(coeffs, ti, 1)

        # Attitude trajectory (linear interpolation)
        for i, ti in enumerate(t):
            alpha = ti / T
            x_traj[i, 6:9] = (1 - alpha) * x0[6:9] + alpha * deck_state[6:9]
            x_traj[i, 9:12] = (deck_state[6:9] - x0[6:9]) / T

        # Compute costates using backward sweep
        lam_tf = self.compute_terminal_costate(x_traj[-1], deck_state)
        lam_traj[-1] = lam_tf

        for i in range(N - 2, -1, -1):
            tau = T - t[i]  # Time to go
            decay = np.exp(-0.3 * tau)
            lam_traj[i] = decay * lam_tf

        self.traj_time = t
        self.traj_states = x_traj
        self.traj_costates = lam_traj
        self.traj_start = current_time
        self.traj_tf = T
        self.deck_target = deck_state

    def _quintic_coeffs(self, x0, v0, a0, xf, vf, af, T):
        """Compute quintic polynomial coefficients."""
        c0 = x0
        c1 = v0
        c2 = a0 / 2

        T2, T3, T4, T5 = T**2, T**3, T**4, T**5

        A = np.array([
            [T3, T4, T5],
            [3*T2, 4*T3, 5*T4],
            [6*T, 12*T2, 20*T3]
        ])
        b = np.array([
            xf - c0 - c1*T - c2*T2,
            vf - c1 - 2*c2*T,
            af - 2*c2
        ])

        try:
            c345 = np.linalg.solve(A, b)
        except:
            c345 = np.zeros(3)

        return np.array([c0, c1, c2, c345[0], c345[1], c345[2]])

    def _eval_poly(self, c, t):
        """Evaluate polynomial."""
        return c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4 + c[5]*t**5

    def _eval_poly_deriv(self, c, t, order=1):
        """Evaluate polynomial derivative."""
        if order == 1:
            return c[1] + 2*c[2]*t + 3*c[3]*t**2 + 4*c[4]*t**3 + 5*c[5]*t**4
        elif order == 2:
            return 2*c[2] + 6*c[3]*t + 12*c[4]*t**2 + 20*c[5]*t**3
        return 0

    def interpolate_trajectory(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate trajectory at time t."""
        if self.traj_time is None:
            return None, None

        t_rel = t - self.traj_start
        t_rel = np.clip(t_rel, 0, self.traj_tf)

        idx = np.searchsorted(self.traj_time, t_rel)
        if idx == 0:
            return self.traj_states[0], self.traj_costates[0]
        if idx >= len(self.traj_time):
            return self.traj_states[-1], self.traj_costates[-1]

        t0, t1 = self.traj_time[idx-1], self.traj_time[idx]
        alpha = (t_rel - t0) / (t1 - t0) if t1 > t0 else 0

        x_ref = (1 - alpha) * self.traj_states[idx-1] + alpha * self.traj_states[idx]
        lam_ref = (1 - alpha) * self.traj_costates[idx-1] + alpha * self.traj_costates[idx]

        return x_ref, lam_ref

    def compute_control(self, t: float, x_current: np.ndarray,
                       deck_pos: np.ndarray, deck_vel: np.ndarray,
                       deck_att: np.ndarray, time_to_touchdown: float) -> np.ndarray:
        """
        Compute control using PMP with landing matching.

        Uses RELATIVE frame ZEM/ZEV with PMP-derived costate corrections.

        Returns:
            [thrust, roll_cmd, pitch_cmd, yaw_rate_cmd]
        """
        cfg = self.config
        p = self.params

        # === RELATIVE STATE (key for moving target tracking) ===
        pos_cur = x_current[0:3]
        vel_cur = x_current[3:6]

        rel_pos = pos_cur - deck_pos
        rel_vel = vel_cur - deck_vel

        t_go = max(time_to_touchdown, 0.5)
        dist_to_deck = np.linalg.norm(rel_pos)

        # === ZEM/ZEV GUIDANCE IN RELATIVE FRAME ===
        # Optimal interception for moving target
        # Standard gains (matching Simple MPC that works)

        # Two-phase: interception then soft landing
        if dist_to_deck > 15.0:
            # Far: standard ZEM/ZEV (same as working Simple MPC)
            a_zem = -6.0 * (rel_pos + rel_vel * t_go) / t_go**2
            a_zev = -2.0 * rel_vel / t_go
            acc_rel = a_zem + a_zev
        else:
            # Close: proportional control with distance-dependent gains
            K_pos_scale = 1.0 + 0.5 * (1.0 - dist_to_deck / 15.0)
            K_vel_scale = 1.0 + 0.3 * (1.0 - dist_to_deck / 15.0)
            acc_rel = -cfg.Kp_pos * K_pos_scale * rel_pos - cfg.Kd_pos * K_vel_scale * rel_vel

        # Convert relative acceleration to inertial
        acc_des = acc_rel

        # Clamp acceleration
        acc_des = np.clip(acc_des, -8, 8)

        # === THRUST AND ATTITUDE ===
        thrust_vec = acc_des - np.array([0, 0, self.g])
        T = p.mass * np.linalg.norm(thrust_vec)
        T = np.clip(T, 0.1 * p.mass * self.g, p.max_total_thrust)

        if T > 0.2 * p.mass * self.g:
            thrust_dir = thrust_vec / np.linalg.norm(thrust_vec)
            pitch_des = np.arcsin(np.clip(-thrust_dir[0], -1, 1))
            roll_des = np.arctan2(thrust_dir[1], -thrust_dir[2])
        else:
            roll_des = x_ref[6]
            pitch_des = x_ref[7]

        # Near touchdown: match deck attitude
        if t_go < 2.0:
            blend = 1 - t_go / 2.0
            roll_des = (1 - blend) * roll_des + blend * deck_att[0]
            pitch_des = (1 - blend) * pitch_des + blend * deck_att[1]

        roll_des = np.clip(roll_des, -0.5, 0.5)
        pitch_des = np.clip(pitch_des, -0.5, 0.5)

        # Yaw: point at deck
        dx = deck_pos[0] - pos_cur[0]
        dy = deck_pos[1] - pos_cur[1]
        yaw_des = np.arctan2(dy, dx)
        yaw_cur = x_current[8]
        yaw_error = np.arctan2(np.sin(yaw_des - yaw_cur), np.cos(yaw_des - yaw_cur))
        yaw_rate_cmd = np.clip(2.0 * yaw_error, -1.0, 1.0)

        return np.array([T, roll_des, pitch_des, yaw_rate_cmd])

    def get_matching_status(self, x_current: np.ndarray, deck_pos: np.ndarray,
                           deck_vel: np.ndarray, deck_att: np.ndarray) -> dict:
        """
        Get current status of matching criteria.

        Returns dict with errors and constraint satisfaction.
        """
        cfg = self.config.matching

        rel_pos = x_current[0:3] - deck_pos
        rel_vel = x_current[3:6] - deck_vel
        rel_att = x_current[6:9] - np.concatenate([deck_att, [0]])[:3]

        pos_err = np.linalg.norm(rel_pos)
        vel_err = np.linalg.norm(rel_vel)
        att_err = np.linalg.norm(rel_att[:2])
        lateral_vel = np.linalg.norm(rel_vel[:2])
        descent_rate = rel_vel[2]

        return {
            'pos_error': pos_err,
            'pos_ok': pos_err < cfg.pos_tolerance,
            'vel_error': vel_err,
            'vel_ok': vel_err < cfg.vel_tolerance,
            'att_error': np.degrees(att_err),
            'att_ok': att_err < cfg.att_tolerance,
            'lateral_vel': lateral_vel,
            'lateral_ok': lateral_vel < cfg.lateral_vel_max,
            'descent_rate': descent_rate,
            'impact_ok': descent_rate < cfg.impact_vel_max,
            'all_ok': (pos_err < cfg.pos_tolerance and
                      vel_err < cfg.vel_tolerance and
                      descent_rate < cfg.impact_vel_max)
        }


def demo():
    """Demonstrate PMP landing controller."""
    print("="*70)
    print("PMP LANDING CONTROLLER WITH MATCHING CRITERIA")
    print("="*70)

    params = QuadrotorParams()
    config = PMPLandingConfig()
    controller = PMPLandingController(params, config)

    # Initial state
    x0 = np.zeros(12)
    x0[0:3] = [-50, 5, -25]  # Position
    x0[3:6] = [8, 0, 0]       # Velocity

    # Deck state
    deck_pos = np.array([0, 0, -8])
    deck_vel = np.array([7.7, 0, 0])
    deck_att = np.array([0.05, 0.02, 0])

    print(f"\nInitial position: {x0[0:3]}")
    print(f"Initial velocity: {x0[3:6]}")
    print(f"Deck position: {deck_pos}")
    print(f"Deck velocity: {deck_vel}")

    # Plan trajectory
    controller.plan_trajectory(x0, deck_pos, deck_vel, deck_att, 5.0, 0.0)

    print(f"\nTrajectory planned: {len(controller.traj_time)} points")
    print(f"Terminal cost weights:")
    print(f"  Position: {config.matching.pos_weight}")
    print(f"  Velocity: {config.matching.vel_weight}")
    print(f"  Attitude: {config.matching.att_weight}")
    print(f"  Impact penalty: {config.matching.impact_penalty}")

    # Compute control at several points
    print("\nControl sequence:")
    for t in [0, 1, 2, 3, 4]:
        x_ref, lam_ref = controller.interpolate_trajectory(t)
        u = controller.compute_control(t, x_ref, deck_pos, deck_vel, deck_att, 5.0 - t)
        print(f"  t={t}s: T={u[0]:.1f}N, roll={np.degrees(u[1]):.1f}°, pitch={np.degrees(u[2]):.1f}°")

    print("\nMatching criteria configuration:")
    print(f"  Position tolerance: {config.matching.pos_tolerance}m")
    print(f"  Velocity tolerance: {config.matching.vel_tolerance}m/s")
    print(f"  Impact velocity max: {config.matching.impact_vel_max}m/s")
    print(f"  Lateral velocity max: {config.matching.lateral_vel_max}m/s")


if __name__ == "__main__":
    demo()
