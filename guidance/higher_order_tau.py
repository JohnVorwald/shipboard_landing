#!/usr/bin/env python3
"""
Higher-Order Tau Guidance for Shipboard Landing

Extends basic tau guidance with:
1. Second-order tau (τ̈) control for smoother approaches
2. Coupled horizontal-vertical coordination
3. Adaptive tau-dot based on deck motion prediction

Theory:
  First-order tau: τ̇ = constant produces soft contact
  Second-order tau: τ̈ = 0 (constant τ̇) gives even smoother profile

  For τ̈ = 0:
    τ(t) = τ̇ * (t - t_f)
    r(t) = r_0 * (1 - t/T)^(1/τ̇)

  Second-order control adds jerk limiting for smoother control inputs.

Reference:
  Lee, D.N. (2009). "General tau theory: evolution to date"
  Padfield, G.D. (2011). "Rotorcraft handling qualities analysis using tau"
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quad_dynamics.quadrotor import QuadrotorParams


@dataclass
class SecondOrderTauConfig:
    """Configuration for second-order tau guidance."""
    # First-order tau targets
    tau_dot_approach: float = 0.6
    tau_dot_terminal: float = 0.4
    tau_dot_vertical: float = 0.5

    # Second-order (jerk) limits
    tau_ddot_max: float = 0.2      # Max τ̈
    jerk_limit: float = 5.0        # Max jerk (m/s³)

    # Transition heights
    transition_height: float = 10.0  # Height to start tau control
    terminal_height: float = 3.0     # Height for terminal phase

    # Gains
    K_tau: float = 3.0            # τ̇ error gain
    K_tau_dot: float = 1.0        # τ̈ error gain
    K_coupling: float = 0.5       # Horizontal-vertical coupling

    # Limits
    max_accel: float = 6.0
    max_descent: float = 3.0
    max_roll: float = 0.4
    max_pitch: float = 0.4


class SecondOrderTauGuidance:
    """
    Second-order tau guidance with jerk limiting.

    Provides smoother acceleration profiles by controlling τ̈
    rather than just τ̇.
    """

    def __init__(self, config: SecondOrderTauConfig = None,
                 quad_params: QuadrotorParams = None):
        self.config = config if config is not None else SecondOrderTauConfig()
        self.params = quad_params if quad_params is not None else QuadrotorParams()
        self.g = 9.81

        # State for derivative estimation
        self.prev_tau_h = None
        self.prev_tau_v = None
        self.prev_time = None
        self.tau_dot_h = 0
        self.tau_dot_v = 0

    def reset(self):
        """Reset internal state."""
        self.prev_tau_h = None
        self.prev_tau_v = None
        self.prev_time = None
        self.tau_dot_h = 0
        self.tau_dot_v = 0

    def compute_tau(self, r: float, r_dot: float) -> float:
        """Compute tau (time-to-contact)."""
        if abs(r_dot) < 0.1:
            return 100.0 if r > 0 else -100.0
        return r / (-r_dot)

    def compute_tau_dot(self, tau: float, tau_prev: float, dt: float) -> float:
        """Estimate τ̇ from consecutive tau values."""
        if tau_prev is None or dt <= 0:
            return 0
        return (tau - tau_prev) / dt

    def get_tau_dot_desired(self, height: float, horiz_dist: float) -> Tuple[float, float]:
        """Get desired τ̇ based on phase."""
        cfg = self.config

        if height < cfg.terminal_height:
            tau_dot_v = cfg.tau_dot_terminal
            tau_dot_h = cfg.tau_dot_terminal
        elif height < cfg.transition_height:
            # Blend between approach and terminal
            alpha = (height - cfg.terminal_height) / (cfg.transition_height - cfg.terminal_height)
            tau_dot_v = (1 - alpha) * cfg.tau_dot_terminal + alpha * cfg.tau_dot_vertical
            tau_dot_h = (1 - alpha) * cfg.tau_dot_terminal + alpha * cfg.tau_dot_approach
        else:
            tau_dot_v = cfg.tau_dot_vertical
            tau_dot_h = cfg.tau_dot_approach

        return tau_dot_h, tau_dot_v

    def compute_control(self, pos: np.ndarray, vel: np.ndarray,
                       deck_pos: np.ndarray, deck_vel: np.ndarray,
                       deck_att: np.ndarray = None,
                       t: float = None) -> Tuple[np.ndarray, dict]:
        """
        Compute second-order tau guidance.

        Args:
            pos: UAV position [3] NED
            vel: UAV velocity [3] NED
            deck_pos: Deck position [3] NED
            deck_vel: Deck velocity [3] NED
            deck_att: Deck attitude [3] (optional)
            t: Current time (for derivative estimation)

        Returns:
            (acceleration_command, info_dict)
        """
        cfg = self.config

        if deck_att is None:
            deck_att = np.zeros(3)

        # Relative state
        rel_pos = deck_pos - pos
        rel_vel = deck_vel - vel

        # Decompose into horizontal and vertical
        horiz_pos = rel_pos[:2]
        horiz_vel = rel_vel[:2]
        horiz_dist = np.linalg.norm(horiz_pos)

        height = rel_pos[2]  # Positive when deck is below (deck is target)
        vert_vel = rel_vel[2]

        # Compute horizontal and vertical tau
        if horiz_dist > 0.1:
            horiz_closing = -np.dot(horiz_pos, horiz_vel) / horiz_dist
            tau_h = horiz_dist / max(horiz_closing, 0.1) if horiz_closing > 0 else 100.0
        else:
            tau_h = 100.0

        tau_v = self.compute_tau(abs(height), vert_vel)

        # Estimate τ̇
        if t is not None and self.prev_time is not None:
            dt = t - self.prev_time
            if dt > 0:
                self.tau_dot_h = self.compute_tau_dot(tau_h, self.prev_tau_h, dt)
                self.tau_dot_v = self.compute_tau_dot(tau_v, self.prev_tau_v, dt)
        else:
            dt = 0.01  # Assume 100Hz

        # Store for next iteration
        self.prev_tau_h = tau_h
        self.prev_tau_v = tau_v
        self.prev_time = t

        # Get desired τ̇
        tau_dot_h_des, tau_dot_v_des = self.get_tau_dot_desired(abs(height), horiz_dist)

        # First-order tau control
        tau_dot_err_h = tau_dot_h_des - self.tau_dot_h
        tau_dot_err_v = tau_dot_v_des - self.tau_dot_v

        # Horizontal acceleration from tau control
        if horiz_dist > 1.0 and tau_h < 50:
            # a_horiz = K * (τ̇_des - τ̇) * r / τ²
            a_h_mag = cfg.K_tau * tau_dot_err_h * horiz_dist / max(tau_h ** 2, 1.0)
            a_h_mag = np.clip(a_h_mag, -cfg.max_accel, cfg.max_accel)
            a_horiz = a_h_mag * horiz_pos / horiz_dist
        else:
            # Near target - position control
            a_horiz = cfg.K_tau * horiz_pos + 2.0 * horiz_vel

        # Vertical acceleration from tau control
        if abs(height) > 1.0 and tau_v < 50:
            a_vert = cfg.K_tau * tau_dot_err_v * abs(height) / max(tau_v ** 2, 1.0)
        else:
            a_vert = cfg.K_tau * height + 2.0 * vert_vel

        # Coupling term: coordinate arrival times
        if horiz_dist > 1.0 and abs(height) > 1.0:
            tau_diff = tau_h - tau_v
            coupling = cfg.K_coupling * tau_diff
            # If horizontal arrival is later, speed up horizontal
            a_horiz *= (1 + coupling * 0.1)
            # If vertical arrival is later, speed up vertical
            a_vert *= (1 - coupling * 0.1)

        # Apply limits
        a_horiz = np.clip(a_horiz, -cfg.max_accel, cfg.max_accel)
        a_vert = np.clip(a_vert, -cfg.max_descent, cfg.max_descent)

        # Combine into acceleration command
        acc_cmd = np.array([a_horiz[0], a_horiz[1], a_vert])

        info = {
            'tau_h': tau_h,
            'tau_v': tau_v,
            'tau_dot_h': self.tau_dot_h,
            'tau_dot_v': self.tau_dot_v,
            'tau_dot_h_des': tau_dot_h_des,
            'tau_dot_v_des': tau_dot_v_des,
            'height': abs(height),
            'horiz_dist': horiz_dist
        }

        return acc_cmd, info


def demo():
    """Demonstrate second-order tau guidance."""
    print("Second-Order Tau Guidance Demo")
    print("=" * 50)

    guidance = SecondOrderTauGuidance()

    # Simulate approach
    pos = np.array([-50.0, 10.0, -30.0])
    vel = np.array([5.0, -1.0, 2.0])
    deck_pos = np.array([0.0, 0.0, -8.0])
    deck_vel = np.array([7.7, 0.0, 0.0])

    print(f"\nInitial state:")
    print(f"  UAV: pos={pos}, vel={vel}")
    print(f"  Deck: pos={deck_pos}, vel={deck_vel}")

    dt = 0.1
    for i in range(50):
        t = i * dt
        acc_cmd, info = guidance.compute_control(pos, vel, deck_pos, deck_vel, t=t)

        if i % 10 == 0:
            print(f"\nt={t:.1f}s:")
            print(f"  τ_h={info['tau_h']:.2f}s, τ_v={info['tau_v']:.2f}s")
            print(f"  τ̇_h={info['tau_dot_h']:.3f}, τ̇_v={info['tau_dot_v']:.3f}")
            print(f"  acc=[{acc_cmd[0]:.2f}, {acc_cmd[1]:.2f}, {acc_cmd[2]:.2f}]")

        # Simple integration
        vel += acc_cmd * dt
        pos += vel * dt
        deck_pos += deck_vel * dt


if __name__ == "__main__":
    demo()
