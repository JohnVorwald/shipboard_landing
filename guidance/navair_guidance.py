#!/usr/bin/env python3
"""
NAVAIR-Style Shipboard Landing Guidance

Based on NAVAIR demonstration: small quad landing on YP (Yard Patrol boat)
in the bay using vision-integrated guidance and control.

Flight Profile:
1. APPROACH: Horizontal flight toward ship at approach altitude
2. GLIDE: 60° glide slope descent to 1m above deck
3. VERTICAL: Vertical descent to deck

Parameters:
- Approach altitude: ~10-15m above deck
- Glide slope: 60° (steep approach)
- Transition altitude: 1m above deck
- Ship speed: ~5 knots (demonstrated)

Reference: NAVAIR shipboard landing demonstrations
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quad_dynamics.quadrotor import QuadrotorParams


class FlightPhase(Enum):
    APPROACH = "approach"      # Horizontal flight to glide slope intercept
    GLIDE = "glide"           # 60° glide slope descent
    VERTICAL = "vertical"      # Vertical descent to deck
    LANDED = "landed"


@dataclass
class NAVAIRConfig:
    """Configuration for NAVAIR-style guidance."""
    # Approach phase
    approach_altitude: float = 12.0     # Altitude above deck for approach (m)
    approach_speed: float = 12.0        # Horizontal approach speed (m/s) - must exceed ship speed

    # Glide slope phase
    glide_angle_deg: float = 60.0       # Glide slope angle (degrees)
    glide_speed: float = 6.0            # Speed along glide slope (m/s)

    # Vertical descent phase
    vertical_altitude: float = 1.5       # Altitude to start vertical descent (m)
    vertical_speed: float = 0.8          # Vertical descent speed (m/s)

    # Intercept geometry
    glide_intercept_distance: float = 15.0  # Horizontal distance to start glide (m)

    # Control gains
    K_pos: float = 3.0                  # Position gain
    K_vel: float = 5.0                  # Velocity gain
    K_pos_terminal: float = 5.0         # Terminal position gain
    K_vel_terminal: float = 8.0         # Terminal velocity gain

    # Limits
    max_accel: float = 6.0              # Max acceleration (m/s²)
    max_roll: float = 0.4               # Max roll angle (rad)
    max_pitch: float = 0.4              # Max pitch angle (rad)


class NAVAIRGuidance:
    """
    NAVAIR-style three-phase landing guidance.

    Phase 1 (APPROACH): Fly horizontally at approach altitude toward ship
    Phase 2 (GLIDE): Descend on 60° glide slope to 1m above deck
    Phase 3 (VERTICAL): Descend vertically onto deck
    """

    def __init__(self, config: NAVAIRConfig = None,
                 quad_params: QuadrotorParams = None):
        self.config = config if config is not None else NAVAIRConfig()
        self.params = quad_params if quad_params is not None else QuadrotorParams()
        self.g = 9.81

        # Precompute glide slope parameters
        self.glide_angle = np.radians(self.config.glide_angle_deg)
        self.sin_glide = np.sin(self.glide_angle)
        self.cos_glide = np.cos(self.glide_angle)

        # Phase tracking
        self.phase = FlightPhase.APPROACH
        self.phase_start_time = 0.0

    def determine_phase(self, pos: np.ndarray, vel: np.ndarray,
                       deck_pos: np.ndarray, deck_vel: np.ndarray) -> FlightPhase:
        """
        Determine current flight phase based on geometry.

        Phase transitions:
        - APPROACH → GLIDE: When horizontal distance < glide_intercept_distance
                            AND at or below approach altitude
        - GLIDE → VERTICAL: When altitude above deck < vertical_altitude
        - VERTICAL → LANDED: When altitude < 0.2m
        """
        cfg = self.config

        # Relative position (UAV relative to deck)
        rel_pos = pos - deck_pos
        height = -rel_pos[2]  # Height above deck (positive)
        horiz_dist = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)

        # Check for landing
        if height < 0.2:
            return FlightPhase.LANDED

        # Check for vertical descent phase
        if height < cfg.vertical_altitude + 0.5:
            return FlightPhase.VERTICAL

        # Check for glide slope phase
        # Enter glide when close enough horizontally AND at/below approach altitude
        approach_alt_above_deck = cfg.approach_altitude
        if horiz_dist < cfg.glide_intercept_distance and height < approach_alt_above_deck + 2.0:
            return FlightPhase.GLIDE

        # Otherwise approach phase
        return FlightPhase.APPROACH

    def compute_approach_control(self, pos: np.ndarray, vel: np.ndarray,
                                 deck_pos: np.ndarray, deck_vel: np.ndarray) -> np.ndarray:
        """
        Phase 1: Horizontal approach at constant altitude.

        Fly toward ship at approach altitude using pursuit guidance.
        Must close the gap with moving ship.
        """
        cfg = self.config

        rel_pos = pos - deck_pos
        rel_vel = vel - deck_vel
        horiz_vec = rel_pos[:2]
        horiz_dist = np.linalg.norm(horiz_vec)

        # Target: fly toward deck at approach altitude
        target_altitude = deck_pos[2] - cfg.approach_altitude  # NED: negative = up

        # Horizontal: pursuit toward deck with closing rate
        if horiz_dist > 1.0:
            approach_dir = -horiz_vec / horiz_dist  # Toward deck
        else:
            approach_dir = np.array([1.0, 0.0])  # Default forward

        # Closing rate needed - scale with distance
        # Far: close faster; Near: transition to glide
        if horiz_dist > 30:
            closing_rate = cfg.approach_speed
        else:
            closing_rate = max(cfg.approach_speed * horiz_dist / 30.0, 3.0)

        # Desired velocity: closing rate toward deck + deck velocity
        desired_vel_horiz = closing_rate * approach_dir + deck_vel[:2]
        vel_error_horiz = desired_vel_horiz - vel[:2]

        # Vertical: maintain approach altitude with smooth blending
        alt_error = target_altitude - pos[2]
        desired_vel_vert = cfg.K_pos * alt_error + deck_vel[2]
        vel_error_vert = desired_vel_vert - vel[2]

        # Acceleration command with stronger gains
        acc_horiz = cfg.K_vel * vel_error_horiz
        acc_vert = cfg.K_vel * vel_error_vert

        return np.array([acc_horiz[0], acc_horiz[1], acc_vert])

    def compute_glide_control(self, pos: np.ndarray, vel: np.ndarray,
                              deck_pos: np.ndarray, deck_vel: np.ndarray) -> np.ndarray:
        """
        Phase 2: Glide slope descent (60°).

        Descend along steep glide slope toward point 1m above deck.
        """
        cfg = self.config

        rel_pos = pos - deck_pos
        rel_vel = vel - deck_vel
        height = -rel_pos[2]
        horiz_vec = rel_pos[:2]
        horiz_dist = np.linalg.norm(horiz_vec)

        # Direction toward deck (horizontal)
        if horiz_dist > 0.1:
            horiz_dir = -horiz_vec / horiz_dist
        else:
            horiz_dir = np.array([1.0, 0.0])  # Default forward

        # Glide slope target point: 1m above deck (vertical_altitude)
        # From current position, compute desired descent along glide slope

        # Height remaining to vertical phase
        height_to_transition = height - cfg.vertical_altitude

        if height_to_transition > 0:
            # On glide slope: velocity along slope
            # Glide angle from vertical: horizontal component / vertical component
            # For 60° from horizontal (30° from vertical):
            # descent_rate / horiz_speed = tan(60°) = 1.73

            # Desired velocity vector along glide slope
            # horiz_speed = glide_speed * cos(glide_angle)
            # descent_rate = glide_speed * sin(glide_angle)
            glide_horiz_speed = cfg.glide_speed * self.cos_glide
            glide_descent_rate = cfg.glide_speed * self.sin_glide

            desired_vel_horiz = glide_horiz_speed * horiz_dir + deck_vel[:2]
            desired_vel_vert = deck_vel[2] + glide_descent_rate  # NED: positive = down
        else:
            # Below transition - slow down for vertical phase
            desired_vel_horiz = deck_vel[:2]
            desired_vel_vert = deck_vel[2] + cfg.vertical_speed

        vel_error_horiz = desired_vel_horiz - vel[:2]
        vel_error_vert = desired_vel_vert - vel[2]

        # Path error correction: stay on glide slope
        # Ideal height for current horizontal distance on 60° slope
        ideal_height = cfg.vertical_altitude + horiz_dist * np.tan(self.glide_angle)
        height_error = height - ideal_height

        # Add altitude correction to stay on glide slope
        alt_correction = -1.0 * height_error  # Descend if too high

        acc_horiz = cfg.K_vel * vel_error_horiz
        acc_vert = cfg.K_vel * vel_error_vert + 0.5 * alt_correction

        return np.array([acc_horiz[0], acc_horiz[1], acc_vert])

    def compute_vertical_control(self, pos: np.ndarray, vel: np.ndarray,
                                 deck_pos: np.ndarray, deck_vel: np.ndarray) -> np.ndarray:
        """
        Phase 3: Vertical descent to deck.

        Descend vertically while station-keeping above deck.
        Match deck velocity at touchdown.
        """
        cfg = self.config

        rel_pos = pos - deck_pos
        rel_vel = vel - deck_vel
        height = -rel_pos[2]

        # Horizontal: station-keep above deck
        # Use high gains for precise positioning
        pos_error_horiz = -rel_pos[:2]  # Error toward deck
        vel_error_horiz = -rel_vel[:2]  # Match deck velocity

        acc_horiz = cfg.K_pos_terminal * pos_error_horiz + cfg.K_vel_terminal * vel_error_horiz

        # Vertical: controlled descent matching deck at touchdown
        # Desired descent rate decreases as we approach deck
        if height > 0.5:
            desired_descent = cfg.vertical_speed
        else:
            # Slow down near deck
            desired_descent = cfg.vertical_speed * (height / 0.5)
            desired_descent = max(desired_descent, 0.1)

        desired_vel_vert = deck_vel[2] + desired_descent
        vel_error_vert = desired_vel_vert - vel[2]

        acc_vert = cfg.K_vel_terminal * vel_error_vert

        return np.array([acc_horiz[0], acc_horiz[1], acc_vert])

    def compute_control(self, pos: np.ndarray, vel: np.ndarray,
                       deck_pos: np.ndarray, deck_vel: np.ndarray,
                       deck_att: np.ndarray = None) -> Tuple[np.ndarray, dict]:
        """
        Main control computation.

        Args:
            pos: UAV position [3] NED
            vel: UAV velocity [3] NED
            deck_pos: Deck position [3] NED
            deck_vel: Deck velocity [3] NED
            deck_att: Deck attitude [roll, pitch, yaw] (optional)

        Returns:
            (acceleration_command, info_dict)
        """
        cfg = self.config

        # Determine phase
        self.phase = self.determine_phase(pos, vel, deck_pos, deck_vel)

        # Compute phase-specific control
        if self.phase == FlightPhase.APPROACH:
            acc_cmd = self.compute_approach_control(pos, vel, deck_pos, deck_vel)
        elif self.phase == FlightPhase.GLIDE:
            acc_cmd = self.compute_glide_control(pos, vel, deck_pos, deck_vel)
        elif self.phase == FlightPhase.VERTICAL:
            acc_cmd = self.compute_vertical_control(pos, vel, deck_pos, deck_vel)
        else:  # LANDED
            acc_cmd = np.array([0.0, 0.0, 0.0])

        # Limit accelerations
        acc_horiz = np.sqrt(acc_cmd[0]**2 + acc_cmd[1]**2)
        if acc_horiz > cfg.max_accel:
            acc_cmd[0] *= cfg.max_accel / acc_horiz
            acc_cmd[1] *= cfg.max_accel / acc_horiz
        acc_cmd[2] = np.clip(acc_cmd[2], -cfg.max_accel, cfg.max_accel)

        # Compute metrics
        rel_pos = pos - deck_pos
        height = -rel_pos[2]
        horiz_dist = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)

        info = {
            'phase': self.phase.value,
            'height': height,
            'horiz_dist': horiz_dist,
            'glide_angle': self.config.glide_angle_deg
        }

        return acc_cmd, info

    def compute_thrust_attitude(self, acc_cmd: np.ndarray, yaw: float,
                                deck_att: np.ndarray, height: float) -> Tuple[float, float, float]:
        """
        Convert acceleration command to thrust and attitude.

        Args:
            acc_cmd: Desired acceleration [3] NED
            yaw: Current yaw angle
            deck_att: Deck attitude [roll, pitch, yaw]
            height: Height above deck

        Returns:
            (thrust, roll_cmd, pitch_cmd)
        """
        cfg = self.config
        m = self.params.mass

        # Thrust required for vertical acceleration
        # NED: gravity is +g in z, thrust is -T*cos(angles)
        T_vert = m * (self.g - acc_cmd[2])

        # Total thrust (approximate for small angles)
        T_horiz = np.sqrt(acc_cmd[0]**2 + acc_cmd[1]**2) * m
        thrust = np.sqrt(T_vert**2 + T_horiz**2)
        thrust = np.clip(thrust, 0.2 * m * self.g, 1.8 * m * self.g)

        # Attitude from acceleration (rotate to body frame)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        a_forward = cos_yaw * acc_cmd[0] + sin_yaw * acc_cmd[1]
        a_right = -sin_yaw * acc_cmd[0] + cos_yaw * acc_cmd[1]

        g_eff = max(self.g - acc_cmd[2], 0.5)
        pitch_cmd = np.arctan2(a_forward, g_eff)
        roll_cmd = np.arctan2(-a_right, g_eff)

        # Blend toward deck attitude in vertical phase
        if height < 2.0 and height > 0 and deck_att is not None:
            blend = 1 - height / 2.0
            roll_cmd = (1 - blend) * roll_cmd + blend * deck_att[0]
            pitch_cmd = (1 - blend) * pitch_cmd + blend * deck_att[1]

        # Clamp
        roll_cmd = np.clip(roll_cmd, -cfg.max_roll, cfg.max_roll)
        pitch_cmd = np.clip(pitch_cmd, -cfg.max_pitch, cfg.max_pitch)

        return thrust, roll_cmd, pitch_cmd


def demo():
    """Demonstrate NAVAIR guidance."""
    print("NAVAIR-Style Landing Guidance Demo")
    print("=" * 50)
    print("\nFlight Profile:")
    print("  1. APPROACH: Horizontal flight at 12m altitude")
    print("  2. GLIDE: 60° descent to 1m above deck")
    print("  3. VERTICAL: Vertical descent to deck")

    config = NAVAIRConfig(
        approach_altitude=12.0,
        glide_angle_deg=60.0,
        vertical_altitude=1.0,
        glide_intercept_distance=20.0
    )

    guidance = NAVAIRGuidance(config)

    # Test scenarios for each phase
    deck_pos = np.array([0.0, 0.0, -2.0])   # Deck at 2m elevation
    deck_vel = np.array([2.5, 0.0, 0.0])    # 5 knots forward
    deck_att = np.array([0.02, 0.01, 0.0])  # Slight roll/pitch

    print("\n" + "-" * 50)

    # Phase 1: Approach
    pos1 = np.array([-50.0, 5.0, -14.0])  # 50m behind, 12m above deck
    vel1 = np.array([5.0, -0.5, 0.0])
    acc1, info1 = guidance.compute_control(pos1, vel1, deck_pos, deck_vel, deck_att)
    print(f"\nPHASE: {info1['phase'].upper()}")
    print(f"  Position: 50m behind, 12m above")
    print(f"  Acceleration: [{acc1[0]:.2f}, {acc1[1]:.2f}, {acc1[2]:.2f}] m/s²")

    # Phase 2: Glide
    pos2 = np.array([-10.0, 0.5, -8.0])   # 10m behind, 6m above deck
    vel2 = np.array([4.0, 0.0, 2.0])      # Descending
    acc2, info2 = guidance.compute_control(pos2, vel2, deck_pos, deck_vel, deck_att)
    print(f"\nPHASE: {info2['phase'].upper()}")
    print(f"  Position: 10m behind, 6m above")
    print(f"  Acceleration: [{acc2[0]:.2f}, {acc2[1]:.2f}, {acc2[2]:.2f}] m/s²")

    # Phase 3: Vertical
    pos3 = np.array([-0.5, 0.2, -2.8])    # Nearly above deck, 0.8m up
    vel3 = np.array([2.5, 0.0, 0.3])      # Matching ship speed, slow descent
    acc3, info3 = guidance.compute_control(pos3, vel3, deck_pos, deck_vel, deck_att)
    print(f"\nPHASE: {info3['phase'].upper()}")
    print(f"  Position: Above deck, 0.8m up")
    print(f"  Acceleration: [{acc3[0]:.2f}, {acc3[1]:.2f}, {acc3[2]:.2f}] m/s²")

    thrust, roll, pitch = guidance.compute_thrust_attitude(
        acc3, 0.0, deck_att, info3['height']
    )
    print(f"\nThrust/Attitude (vertical phase):")
    print(f"  Thrust: {thrust:.1f} N")
    print(f"  Roll: {np.degrees(roll):.1f}°")
    print(f"  Pitch: {np.degrees(pitch):.1f}°")


if __name__ == "__main__":
    demo()
