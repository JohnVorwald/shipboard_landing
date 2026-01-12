#!/usr/bin/env python3
"""
Tau-Based Landing Guidance for Shipboard Landing

Based on Penn State research (Holmes 2017):
- Uses optical flow variable τ = range / range-rate
- τ represents time-to-contact with target
- Controlling τ̇ produces smooth deceleration profiles

Theory:
  τ = r / ṙ  where r = distance, ṙ = closing rate

  For constant τ̇ = k (0 < k < 1):
    r(t) = r0 * (1 - t/τ0)^(1/k)
    ṙ(t) = ṙ0 * (1 - t/τ0)^(1/k - 1)

  k = 0.5 gives smooth deceleration profile
  k = 1.0 gives constant velocity (collision)
  k < 0.5 gives very aggressive braking near target

Reference: Holmes, W. (2017). "Vision-based relative deck state
estimation used with tau-based landings." MS Thesis, Penn State.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quad_dynamics.quadrotor import QuadrotorParams


@dataclass
class TauGuidanceConfig:
    """Configuration for tau-based guidance."""
    # Tau rate targets
    tau_dot_approach: float = 0.6    # τ̇ during approach (higher = faster)
    tau_dot_terminal: float = 0.4    # τ̇ near touchdown (lower = gentler)
    tau_transition_height: float = 5.0  # Height to transition to terminal τ̇

    # Separate tau control for vertical vs horizontal
    tau_dot_vertical: float = 0.5    # τ̇ for vertical (descent)
    tau_dot_horizontal: float = 0.6  # τ̇ for horizontal (approach)

    # Gains
    K_tau: float = 3.0              # Tau rate error gain
    K_pos_near: float = 2.0         # Position gain near target
    K_vel: float = 3.0              # Velocity tracking gain

    # Limits
    max_accel: float = 6.0          # Max horizontal acceleration (m/s²)
    max_descent: float = 3.0        # Max descent rate (m/s)
    min_tau: float = 1.0            # Minimum tau before switching to position control

    # Attitude limits
    max_roll: float = 0.4           # Max roll angle (rad)
    max_pitch: float = 0.4          # Max pitch angle (rad)


class TauGuidanceController:
    """
    Tau-based landing guidance controller.

    Uses time-to-contact (τ) as the primary feedback variable.
    """

    def __init__(self, config: TauGuidanceConfig = None,
                 quad_params: QuadrotorParams = None):
        self.config = config if config is not None else TauGuidanceConfig()
        self.params = quad_params if quad_params is not None else QuadrotorParams()
        self.g = 9.81

    def compute_tau(self, r: float, r_dot: float) -> float:
        """
        Compute tau (time-to-contact).

        τ = r / ṙ (range / range-rate)

        Positive τ means approaching target.
        Negative τ means moving away from target.
        """
        if abs(r_dot) < 0.1:
            # Not closing - return large tau
            return 100.0 if r > 0 else -100.0

        tau = r / (-r_dot)  # Negative because closing = negative r_dot
        return tau

    def compute_tau_dot_desired(self, height: float, horizontal_dist: float) -> Tuple[float, float]:
        """
        Compute desired τ̇ based on flight phase.

        Returns:
            (tau_dot_horiz, tau_dot_vert)
        """
        cfg = self.config

        # Vertical tau dot
        if height < cfg.tau_transition_height:
            # Terminal phase - gentler deceleration
            tau_dot_vert = cfg.tau_dot_terminal
        else:
            tau_dot_vert = cfg.tau_dot_vertical

        # Horizontal tau dot
        if horizontal_dist < 10.0:
            # Close to target - blend to position control
            blend = horizontal_dist / 10.0
            tau_dot_horiz = blend * cfg.tau_dot_horizontal + (1 - blend) * 0.3
        else:
            tau_dot_horiz = cfg.tau_dot_horizontal

        return tau_dot_horiz, tau_dot_vert

    def compute_control(self, pos: np.ndarray, vel: np.ndarray,
                       deck_pos: np.ndarray, deck_vel: np.ndarray,
                       deck_att: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Compute acceleration command using tau guidance.

        Args:
            pos: UAV position [3]
            vel: UAV velocity [3]
            deck_pos: Deck position [3]
            deck_vel: Deck velocity [3]
            deck_att: Deck attitude [roll, pitch, yaw]

        Returns:
            acc_cmd: Desired acceleration [3]
            info: Debug info dict
        """
        cfg = self.config

        # Relative state (UAV relative to deck)
        rel_pos = pos - deck_pos      # Vector from deck to UAV
        rel_vel = vel - deck_vel      # Relative velocity

        # Heights and distances
        height = -rel_pos[2]  # Height above deck (positive)
        horiz_vec = rel_pos[:2]
        horiz_dist = np.linalg.norm(horiz_vec)

        # Direction to deck (horizontal)
        if horiz_dist > 0.1:
            horiz_dir = -horiz_vec / horiz_dist  # Points toward deck
        else:
            horiz_dir = np.zeros(2)

        # === HORIZONTAL TAU GUIDANCE ===
        # Range rate: positive = opening, negative = closing
        horiz_vel_rel = rel_vel[:2]
        horiz_range_rate = np.dot(horiz_vel_rel, horiz_vec) / (horiz_dist + 0.1)
        closing_rate = -horiz_range_rate  # Positive when closing

        # Current horizontal tau
        tau_horiz = self.compute_tau(horiz_dist, horiz_range_rate)

        # Desired tau rate
        tau_dot_horiz_des, tau_dot_vert_des = self.compute_tau_dot_desired(height, horiz_dist)

        # Desired closing rate based on distance (pursuit guidance)
        # Far: close faster, Near: close slower
        if horiz_dist > 30:
            desired_closing_rate = 8.0  # Fast approach
        elif horiz_dist > 10:
            desired_closing_rate = 5.0  # Medium approach
        else:
            desired_closing_rate = 2.0  # Slow for landing

        # === HORIZONTAL GUIDANCE (True Tau Guidance) ===
        # For constant τ̇ = k, required acceleration is:
        #   a = (1 - k) * ṙ² / r  (applied against closing direction)

        if horiz_dist < 3.0:
            # TERMINAL: Position + velocity control for final approach
            # High gains for tight tracking, focus on velocity matching
            K_pos = cfg.K_pos_near * 1.5
            K_vel = cfg.K_vel * 2.0  # Strong velocity damping
            acc_horiz_cmd = -K_pos * rel_pos[:2] - K_vel * rel_vel[:2]

        elif closing_rate < desired_closing_rate:
            # NOT CLOSING FAST ENOUGH: Need pursuit guidance to establish approach
            # Desired velocity toward deck (proportional to distance)
            vel_desired = min(desired_closing_rate, horiz_dist / 3.0)

            # Velocity error (how much faster we need to close)
            vel_error = vel_desired - closing_rate

            # Acceleration to achieve desired closing rate
            acc_approach = cfg.K_vel * vel_error
            acc_approach = np.clip(acc_approach, 0, cfg.max_accel)

            # Apply in direction toward deck
            acc_horiz_cmd = acc_approach * horiz_dir

            # Lateral correction (perpendicular to approach direction)
            # lateral_vel = rel_vel - (component along approach)
            # component along approach = closing_rate * horiz_dir
            lateral_vel = rel_vel[:2] - closing_rate * horiz_dir
            acc_horiz_cmd -= 2.0 * lateral_vel

        else:
            # CLOSING: Use tau guidance for smooth deceleration
            # Tau guidance formula: a = (1 - k) * ṙ² / r
            # This automatically produces smooth braking profile

            # Use tau_dot = 0.5 for smooth deceleration (k=0.5)
            k = tau_dot_horiz_des

            # Deceleration required for constant tau_dot
            # Note: closing_rate is positive when approaching
            a_tau = (1.0 - k) * closing_rate**2 / (horiz_dist + 0.1)

            # Clamp deceleration to feasible limits
            a_tau = min(a_tau, cfg.max_accel)

            # Apply AGAINST closing direction (i.e., decelerate)
            # horiz_dir points toward deck, so -horiz_dir is braking direction
            acc_horiz_cmd = -a_tau * horiz_dir

            # Add small attraction to deck center (prevents lateral drift)
            lateral_vel = rel_vel[:2] - closing_rate * horiz_dir
            acc_horiz_cmd -= 1.5 * lateral_vel

            # If tau is too small (closing too fast), add emergency braking
            if tau_horiz < 2.0 and tau_horiz > 0:
                emergency_brake = cfg.max_accel * (2.0 - tau_horiz) / 2.0
                acc_horiz_cmd -= emergency_brake * horiz_dir

        # Limit horizontal acceleration
        acc_horiz_mag = np.linalg.norm(acc_horiz_cmd)
        if acc_horiz_mag > cfg.max_accel:
            acc_horiz_cmd *= cfg.max_accel / acc_horiz_mag

        # === VERTICAL TAU GUIDANCE ===
        # In NED: positive z = down, so positive rel_vel[2] = descending
        vert_range_rate = rel_vel[2]

        # Current vertical tau
        tau_vert = self.compute_tau(height, vert_range_rate)

        if height < 1.0:
            # TOUCHDOWN: Arrest descent, match deck velocity precisely
            # Target: hover at deck level with matched velocity
            target_descent = 0.2 * max(height, 0)  # Very gentle final descent
            vel_error = target_descent - vert_range_rate
            # Higher gain for velocity matching
            acc_vert_cmd = cfg.K_vel * 2.0 * vel_error

        elif height < 3.0:
            # TERMINAL: Soft landing with gentle descent
            # Descent rate proportional to height
            target_descent = max(0.3, height / 3.0)
            vel_error = target_descent - vert_range_rate
            acc_vert_cmd = cfg.K_vel * vel_error

        elif horiz_dist > 15:
            # FAR: Descend gradually while approaching
            # Don't descend too fast - coordinate with horizontal approach
            target_descent = min(height / 10.0, 1.5)
            vel_error = target_descent - vert_range_rate
            acc_vert_cmd = cfg.K_vel * vel_error

        else:
            # NEAR: Use tau guidance for smooth deceleration
            if height > 3.0 and vert_range_rate > 0.3:
                # Descending - apply tau guidance
                k = tau_dot_vert_des
                a_tau = (1.0 - k) * vert_range_rate**2 / (height + 0.1)
                a_tau = min(a_tau, 4.0)  # Limit vertical deceleration

                # Desired tau for smooth arrival
                tau_desired = height / 1.0  # Aim for ~1s final approach
                tau_error = tau_vert - tau_desired
                tau_correction = -0.5 * np.clip(tau_error, -2.0, 2.0)

                acc_vert_cmd = a_tau + tau_correction
            else:
                # Not descending or low height - velocity control
                target_descent = max(0.5, height / 4.0)
                target_descent = min(target_descent, cfg.max_descent)
                vel_error = target_descent - vert_range_rate
                acc_vert_cmd = cfg.K_vel * vel_error

        # Combine into 3D acceleration command
        acc_cmd = np.array([acc_horiz_cmd[0], acc_horiz_cmd[1], acc_vert_cmd])

        # Debug info
        info = {
            'tau_horiz': tau_horiz,
            'tau_vert': tau_vert,
            'tau_dot_horiz_des': tau_dot_horiz_des,
            'tau_dot_vert_des': tau_dot_vert_des,
            'height': height,
            'horiz_dist': horiz_dist,
            'closing_rate': closing_rate,
            'descent_rate': vert_range_rate
        }

        return acc_cmd, info

    def compute_thrust_attitude(self, acc_cmd: np.ndarray, yaw: float,
                                deck_att: np.ndarray, height: float) -> Tuple[float, float, float]:
        """
        Convert acceleration command to thrust and attitude.

        In NED frame:
        - Positive x = North/forward
        - Positive z = Down
        - Gravity = [0, 0, +g]
        - Quadrotor thrust points along body -z (up when level)

        Acceleration equation:
        a = (T/m) * R * [0, 0, -1] + [0, 0, g]

        For small angles:
        a_x ≈ (T/m) * sin(pitch)
        a_y ≈ -(T/m) * sin(roll)
        a_z ≈ g - (T/m) * cos(pitch) * cos(roll)

        Args:
            acc_cmd: Desired acceleration [3] in NED
            yaw: Current yaw angle
            deck_att: Deck attitude [roll, pitch, yaw]
            height: Height above deck

        Returns:
            (thrust, roll_cmd, pitch_cmd)
        """
        cfg = self.config
        m = self.params.mass

        # Vertical thrust needed: T_z/m = g - a_z
        # For descent (a_z > 0), need less lift
        # For climb (a_z < 0), need more lift
        T_vert = m * (self.g - acc_cmd[2])

        # Horizontal thrust components
        T_horiz = np.sqrt(acc_cmd[0]**2 + acc_cmd[1]**2) * m

        # Total thrust (approximate for small angles)
        thrust = np.sqrt(T_vert**2 + T_horiz**2)
        thrust = np.clip(thrust, 0.2 * m * self.g, 1.8 * m * self.g)

        # Attitude from acceleration (small angle approximation)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

        # Rotate to body frame
        a_forward = cos_yaw * acc_cmd[0] + sin_yaw * acc_cmd[1]
        a_right = -sin_yaw * acc_cmd[0] + cos_yaw * acc_cmd[1]

        # pitch = arcsin(a_forward * m / T) ≈ arctan(a_forward / (g - a_z))
        # Use effective gravity for more accurate attitude
        g_eff = max(self.g - acc_cmd[2], 0.5)
        pitch_cmd = np.arctan2(a_forward, g_eff)
        roll_cmd = np.arctan2(-a_right, g_eff)

        # Near deck: blend toward deck attitude
        if height < 3.0 and height > 0:
            blend = 1 - height / 3.0
            roll_cmd = (1 - blend) * roll_cmd + blend * deck_att[0]
            pitch_cmd = (1 - blend) * pitch_cmd + blend * deck_att[1]

        # Clamp
        roll_cmd = np.clip(roll_cmd, -cfg.max_roll, cfg.max_roll)
        pitch_cmd = np.clip(pitch_cmd, -cfg.max_pitch, cfg.max_pitch)

        return thrust, roll_cmd, pitch_cmd


def demo():
    """Demonstrate tau guidance."""
    print("Tau-Based Landing Guidance Demo")
    print("=" * 50)

    controller = TauGuidanceController()

    # Test scenario: UAV approaching deck
    pos = np.array([-30.0, 5.0, -20.0])   # 30m behind, 5m lateral, 20m up
    vel = np.array([5.0, -0.5, 1.0])       # Moving forward and down
    deck_pos = np.array([0.0, 0.0, -8.0]) # Deck position
    deck_vel = np.array([7.7, 0.0, 0.0])  # Ship moving forward
    deck_att = np.array([0.05, 0.02, 0.0])

    acc_cmd, info = controller.compute_control(pos, vel, deck_pos, deck_vel, deck_att)

    print(f"\nScenario:")
    print(f"  UAV: pos={pos}, vel={vel}")
    print(f"  Deck: pos={deck_pos}, vel={deck_vel}")

    print(f"\nTau Values:")
    print(f"  Horizontal τ: {info['tau_horiz']:.2f} s")
    print(f"  Vertical τ: {info['tau_vert']:.2f} s")
    print(f"  Height: {info['height']:.1f} m")
    print(f"  Horiz dist: {info['horiz_dist']:.1f} m")

    print(f"\nAcceleration Command:")
    print(f"  acc = [{acc_cmd[0]:.2f}, {acc_cmd[1]:.2f}, {acc_cmd[2]:.2f}] m/s²")

    thrust, roll, pitch = controller.compute_thrust_attitude(
        acc_cmd, 0.0, deck_att, info['height']
    )
    print(f"\nThrust/Attitude:")
    print(f"  Thrust: {thrust:.1f} N")
    print(f"  Roll: {np.degrees(roll):.1f}°")
    print(f"  Pitch: {np.degrees(pitch):.1f}°")


if __name__ == "__main__":
    demo()
