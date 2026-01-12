#!/usr/bin/env python3
"""
Zero Effort Miss / Zero Effort Velocity (ZEM-ZEV) Guidance

Classical optimal guidance law based on predictive interception.
Used extensively in missile guidance and spacecraft rendezvous.

Theory:
  Zero Effort Miss (ZEM): Miss distance if no control is applied
    ZEM = (r + v*t_go) where r = relative position, v = relative velocity

  For constant acceleration guidance:
    a_cmd = N * (ZEM / t_go²) where N = navigation constant (typically 3-5)

  This produces an acceleration profile that nulls the miss distance
  at the specified time-to-go.

For shipboard landing:
  - Target position = predicted deck position at t_go
  - Target velocity = deck velocity (for velocity matching)
  - ZEV guidance added for velocity matching at touchdown

References:
  - Zarchan, P. "Tactical and Strategic Missile Guidance"
  - Ebrahimi et al. "Optimal Sliding-Mode Guidance with Terminal Velocity Constraint"
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quad_dynamics.quadrotor import QuadrotorParams


@dataclass
class ZEMGuidanceConfig:
    """Configuration for ZEM-ZEV guidance."""
    # Navigation constants
    N_position: float = 4.0       # Position nulling gain (3-5 typical)
    N_velocity: float = 2.0       # Velocity nulling gain

    # Time-to-go estimation
    min_t_go: float = 0.5         # Minimum t_go to prevent singularity
    t_go_method: str = 'range'    # 'range', 'fixed', or 'optimal'

    # Acceleration limits
    max_accel: float = 8.0        # Max horizontal acceleration (m/s²)
    max_descent: float = 4.0      # Max descent acceleration (m/s²)
    max_climb: float = 6.0        # Max climb acceleration (m/s²)

    # Terminal phase
    terminal_height: float = 3.0  # Height to switch to terminal guidance
    terminal_N: float = 3.0       # Lower N for gentler terminal phase


class ZEMGuidance:
    """
    Zero Effort Miss / Zero Effort Velocity guidance for landing.

    Provides optimal acceleration commands to intercept a moving target
    with specified terminal velocity.
    """

    def __init__(self, config: ZEMGuidanceConfig = None,
                 quad_params: QuadrotorParams = None):
        self.config = config if config is not None else ZEMGuidanceConfig()
        self.params = quad_params if quad_params is not None else QuadrotorParams()
        self.g = 9.81

    def estimate_time_to_go(self, pos: np.ndarray, vel: np.ndarray,
                            target_pos: np.ndarray, target_vel: np.ndarray) -> float:
        """
        Estimate time-to-go to target.

        Uses closing velocity to estimate intercept time.
        """
        cfg = self.config

        # Relative state
        rel_pos = target_pos - pos
        rel_vel = target_vel - vel

        # Range and range-rate
        range_mag = np.linalg.norm(rel_pos)

        if range_mag < 0.1:
            return cfg.min_t_go

        # Closing velocity (positive = approaching)
        closing_vel = -np.dot(rel_pos, rel_vel) / range_mag

        if closing_vel <= 0:
            # Not closing - use range-based estimate
            avg_speed = max(np.linalg.norm(vel), 3.0)
            t_go = range_mag / avg_speed
        else:
            # Simple range/range-rate estimate
            t_go = range_mag / closing_vel

        return max(t_go, cfg.min_t_go)

    def compute_zem(self, pos: np.ndarray, vel: np.ndarray,
                    target_pos: np.ndarray, target_vel: np.ndarray,
                    t_go: float) -> np.ndarray:
        """
        Compute Zero Effort Miss vector.

        ZEM = predicted miss if no acceleration applied.
        Assumes target moves with constant velocity.
        """
        # Predicted target position at t_go
        target_pred = target_pos + target_vel * t_go

        # Predicted UAV position at t_go (constant velocity)
        uav_pred = pos + vel * t_go

        # Miss vector
        zem = target_pred - uav_pred

        return zem

    def compute_zev(self, vel: np.ndarray, target_vel: np.ndarray) -> np.ndarray:
        """
        Compute Zero Effort Velocity error.

        ZEV = velocity error at intercept if no acceleration applied.
        """
        return target_vel - vel

    def compute_acceleration(self, pos: np.ndarray, vel: np.ndarray,
                            target: np.ndarray, t_go: float = None,
                            target_vel: np.ndarray = None) -> np.ndarray:
        """
        Compute ZEM-ZEV guidance acceleration command.

        Args:
            pos: Current position [3]
            vel: Current velocity [3]
            target: Target position [3]
            t_go: Time to go (estimated if not provided)
            target_vel: Target velocity [3] (zero if not provided)

        Returns:
            Acceleration command [3]
        """
        cfg = self.config

        if target_vel is None:
            target_vel = np.zeros(3)

        # Estimate t_go if not provided
        if t_go is None:
            t_go = self.estimate_time_to_go(pos, vel, target, target_vel)
        t_go = max(t_go, cfg.min_t_go)

        # Compute ZEM and ZEV
        zem = self.compute_zem(pos, vel, target, target_vel, t_go)
        zev = self.compute_zev(vel, target_vel)

        # Determine navigation constant based on phase
        height = abs(target[2] - pos[2])
        if height < cfg.terminal_height:
            N = cfg.terminal_N
        else:
            N = cfg.N_position

        # ZEM-ZEV acceleration command
        # a = N * ZEM / t_go² + N_v * ZEV / t_go
        a_zem = N * zem / (t_go ** 2)
        a_zev = cfg.N_velocity * zev / t_go

        accel = a_zem + a_zev

        # Apply limits
        accel[0] = np.clip(accel[0], -cfg.max_accel, cfg.max_accel)
        accel[1] = np.clip(accel[1], -cfg.max_accel, cfg.max_accel)
        accel[2] = np.clip(accel[2], -cfg.max_climb, cfg.max_descent)

        return accel

    def compute_control(self, pos: np.ndarray, vel: np.ndarray,
                       deck_pos: np.ndarray, deck_vel: np.ndarray,
                       deck_att: np.ndarray = None) -> Tuple[np.ndarray, dict]:
        """
        Full control computation with info dictionary.

        Args:
            pos: UAV position [3] NED
            vel: UAV velocity [3] NED
            deck_pos: Deck position [3] NED
            deck_vel: Deck velocity [3] NED
            deck_att: Deck attitude [3] (optional)

        Returns:
            (acceleration_command, info_dict)
        """
        # Estimate time to go
        t_go = self.estimate_time_to_go(pos, vel, deck_pos, deck_vel)

        # Compute ZEM and ZEV
        zem = self.compute_zem(pos, vel, deck_pos, deck_vel, t_go)
        zev = self.compute_zev(vel, deck_vel)

        # Compute acceleration
        accel = self.compute_acceleration(pos, vel, deck_pos, t_go, deck_vel)

        # Compute range
        rel_pos = deck_pos - pos
        range_mag = np.linalg.norm(rel_pos)
        height = abs(rel_pos[2])

        info = {
            't_go': t_go,
            'zem': zem,
            'zev': zev,
            'zem_mag': np.linalg.norm(zem),
            'zev_mag': np.linalg.norm(zev),
            'range': range_mag,
            'height': height
        }

        return accel, info


def demo():
    """Demonstrate ZEM-ZEV guidance."""
    print("ZEM-ZEV Guidance Demo")
    print("=" * 50)

    guidance = ZEMGuidance()

    # Test scenario
    pos = np.array([-50.0, 10.0, -30.0])  # 50m behind, 10m lateral, 30m up
    vel = np.array([8.0, -1.0, 2.0])       # Approaching
    target = np.array([0.0, 0.0, -8.0])    # Deck
    target_vel = np.array([7.7, 0.0, 0.0]) # Ship moving forward

    accel, info = guidance.compute_control(pos, vel, target, target_vel)

    print(f"\nScenario:")
    print(f"  UAV: pos={pos}, vel={vel}")
    print(f"  Deck: pos={target}, vel={target_vel}")

    print(f"\nGuidance Output:")
    print(f"  t_go: {info['t_go']:.2f} s")
    print(f"  ZEM magnitude: {info['zem_mag']:.1f} m")
    print(f"  ZEV magnitude: {info['zev_mag']:.1f} m/s")
    print(f"  Acceleration: [{accel[0]:.2f}, {accel[1]:.2f}, {accel[2]:.2f}] m/s²")


if __name__ == "__main__":
    demo()
