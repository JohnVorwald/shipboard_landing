#!/usr/bin/env python3
"""
Robust Autonomous Landing Controller with Terminal Phase

Features:
- Multi-phase landing (APPROACH -> DESCENT -> TERMINAL -> TOUCHDOWN -> LANDED)
- Velocity matching in terminal phase to prevent bouncing
- Aggressive force limiting to prevent instability
- Robust pose reading with outlier rejection
- Integration with ship wave motion model
"""

import subprocess
import time
import sys
import os
import re
import math
import json
import numpy as np
from enum import Enum
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guidance.zem_guidance import ZEMGuidance, ZEMGuidanceConfig
from ship_wave_motion import ShipWaveMotion, SEA_STATE_4

# Gazebo environment
GZ_ENV = os.environ.copy()
GZ_ENV['GZ_IP'] = '127.0.0.1'
GZ_ENV['GZ_PARTITION'] = 'gazebo_default'

WORLD = "ship_landing_debug"

# Vehicle parameters
MASS = 2.0
G = 9.81
HOVER_THRUST = MASS * G
HELIPAD_OFFSET = np.array([15.0, 0.0, 4.0])

# Phase thresholds
APPROACH_HEIGHT = 8.0  # Above this, approach phase
DESCENT_HEIGHT = 3.0   # Above this (below approach), descent phase
TERMINAL_HEIGHT = 1.0  # Below this, terminal phase
LANDED_HEIGHT = 0.3    # Below this, consider landed
HORIZ_TOLERANCE = 2.0  # Horizontal distance for landing


class LandingPhase(Enum):
    APPROACH = "APPROACH"
    DESCENT = "DESCENT"
    TERMINAL = "TERMINAL"
    TOUCHDOWN = "TOUCHDOWN"
    LANDED = "LANDED"
    ABORT = "ABORT"


@dataclass
class VehicleState:
    position: np.ndarray
    velocity: np.ndarray
    timestamp: float


@dataclass
class TargetState:
    position: np.ndarray
    velocity: np.ndarray


class RobustPoseReader:
    """Robust pose reading with outlier rejection."""

    def __init__(self, max_velocity=30.0, filter_alpha=0.3):
        self.max_velocity = max_velocity
        self.filter_alpha = filter_alpha
        self.prev_pos = None
        self.prev_time = None
        self.velocity = np.zeros(3)
        self.position_history = deque(maxlen=5)

    def get_pose(self) -> Optional[VehicleState]:
        """Get quadcopter pose with validation."""
        raw_pos = self._read_gazebo_pose('quadcopter')
        if raw_pos is None:
            return None

        now = time.time()

        # Outlier rejection
        if self.position_history:
            median_pos = np.median(list(self.position_history), axis=0)
            if np.linalg.norm(raw_pos - median_pos) > 50:  # Reject huge jumps
                return None

        self.position_history.append(raw_pos)

        # Velocity estimation with limiting
        if self.prev_pos is not None and self.prev_time is not None:
            dt = now - self.prev_time
            if 0.01 < dt < 0.3:
                raw_vel = (raw_pos - self.prev_pos) / dt
                # Reject unreasonable velocities
                if np.linalg.norm(raw_vel) < self.max_velocity:
                    self.velocity = self.filter_alpha * raw_vel + (1 - self.filter_alpha) * self.velocity

        self.prev_pos = raw_pos.copy()
        self.prev_time = now

        return VehicleState(
            position=raw_pos,
            velocity=self.velocity.copy(),
            timestamp=now
        )

    def _read_gazebo_pose(self, model_name: str) -> Optional[np.ndarray]:
        """Read pose from Gazebo with proper parsing."""
        try:
            result = subprocess.run(
                ['gz', 'topic', '-e', '-t', f'/world/{WORLD}/dynamic_pose/info', '-n', '1'],
                capture_output=True, text=True, timeout=1.5, env=GZ_ENV
            )
            if result.returncode != 0:
                return None

            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if f'name: "{model_name}"' in line:
                    pos = [0.0, 0.0, 0.0]
                    in_position = False
                    for j in range(i, min(i + 20, len(lines))):
                        pline = lines[j].strip()
                        if pline == 'position {':
                            in_position = True
                        elif pline.startswith('}') and in_position:
                            break
                        elif in_position:
                            for idx, coord in enumerate(['x:', 'y:', 'z:']):
                                if pline.startswith(coord):
                                    match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', pline)
                                    if match:
                                        pos[idx] = float(match.group())
                    return np.array(pos)
        except Exception as e:
            pass
        return None


class ForceController:
    """Force controller with safety limits."""

    def __init__(self):
        self.force_history = deque(maxlen=10)

    def apply_force(self, force: np.ndarray):
        """Apply force with rate limiting."""
        # Clamp forces
        force = self._limit_force(force)

        # Rate limiting
        if self.force_history:
            last_force = self.force_history[-1]
            max_change = np.array([5.0, 5.0, 8.0])  # Max force change per update
            force = np.clip(force, last_force - max_change, last_force + max_change)

        self.force_history.append(force.copy())

        # Send to Gazebo
        self._send_wrench(force)

    def _limit_force(self, force: np.ndarray) -> np.ndarray:
        """Apply hard limits on forces."""
        # Vertical limits
        force[2] = np.clip(force[2], 0.5 * HOVER_THRUST, 1.5 * HOVER_THRUST)

        # Horizontal limits
        max_horiz = MASS * 3.0  # 3 m/s² max horizontal accel
        horiz = np.linalg.norm(force[:2])
        if horiz > max_horiz:
            force[:2] *= max_horiz / horiz

        return force

    def _send_wrench(self, force: np.ndarray):
        """Send wrench to Gazebo."""
        msg = f'entity: {{name: "quadcopter::base_link", type: LINK}}, wrench: {{force: {{x: {force[0]}, y: {force[1]}, z: {force[2]}}}}}'
        subprocess.Popen(
            ['gz', 'topic', '-t', f'/world/{WORLD}/wrench/persistent',
             '-m', 'gz.msgs.EntityWrench', '-p', msg],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=GZ_ENV
        )

    def clear_force(self):
        """Clear persistent force."""
        msg = 'entity: {name: "quadcopter::base_link", type: LINK}'
        try:
            subprocess.run(
                ['gz', 'topic', '-t', f'/world/{WORLD}/wrench/clear',
                 '-m', 'gz.msgs.Entity', '-p', msg],
                capture_output=True, timeout=0.5, env=GZ_ENV
            )
        except:
            pass


class TerminalPhaseLandingController:
    """
    Multi-phase landing controller with terminal phase velocity matching.
    """

    def __init__(self):
        # ZEM guidance for approach and descent
        config = ZEMGuidanceConfig(
            N_position=3.0,
            N_velocity=2.0,
            max_accel=3.0,
            max_descent=2.0,
            max_climb=3.0,
            terminal_height=2.0
        )
        self.guidance = ZEMGuidance(config)

        # Ship motion model
        self.wave_model = ShipWaveMotion(sea_state=SEA_STATE_4, forward_speed=2.5)
        self.ship_initial = np.array([0.0, 0.0, 0.0])

        # Components
        self.pose_reader = RobustPoseReader()
        self.force_controller = ForceController()

        # State
        self.phase = LandingPhase.APPROACH
        self.start_time = None
        self.landed = False
        self.abort_reason = None

    def get_target_state(self, sim_time: float) -> TargetState:
        """Get helipad target state from wave model."""
        pos, vel = self.wave_model.get_helipad_state(sim_time, self.ship_initial, HELIPAD_OFFSET)
        return TargetState(position=pos, velocity=vel)

    def determine_phase(self, vehicle: VehicleState, target: TargetState) -> LandingPhase:
        """Determine current landing phase."""
        rel_pos = vehicle.position - target.position
        height = rel_pos[2]
        horiz_dist = np.linalg.norm(rel_pos[:2])
        rel_vel = np.linalg.norm(vehicle.velocity - target.velocity)

        # Check for landing
        if height < LANDED_HEIGHT and horiz_dist < HORIZ_TOLERANCE and rel_vel < 1.5:
            return LandingPhase.LANDED

        # Check for abort conditions
        if height < -1.0:  # Below deck
            return LandingPhase.ABORT
        if horiz_dist > 50 and height < 5:  # Way off horizontally
            return LandingPhase.ABORT

        # Phase determination
        if height > APPROACH_HEIGHT:
            return LandingPhase.APPROACH
        elif height > DESCENT_HEIGHT:
            return LandingPhase.DESCENT
        elif height > TERMINAL_HEIGHT:
            return LandingPhase.TERMINAL
        else:
            return LandingPhase.TOUCHDOWN

    def compute_approach_force(self, vehicle: VehicleState, target: TargetState) -> np.ndarray:
        """Compute force during approach phase using ZEM guidance."""
        # Convert to NED for guidance
        quad_ned = np.array([vehicle.position[0], vehicle.position[1], -vehicle.position[2]])
        vel_ned = np.array([vehicle.velocity[0], vehicle.velocity[1], -vehicle.velocity[2]])
        target_ned = np.array([target.position[0], target.position[1], -target.position[2]])
        target_vel_ned = np.array([target.velocity[0], target.velocity[1], -target.velocity[2]])

        acc_ned, _ = self.guidance.compute_control(quad_ned, vel_ned, target_ned, target_vel_ned)

        # Convert back to Gazebo frame
        acc = np.array([acc_ned[0], acc_ned[1], -acc_ned[2]])

        # Force = mass * acceleration + gravity compensation
        force = MASS * acc
        force[2] += HOVER_THRUST

        return force

    def compute_terminal_force(self, vehicle: VehicleState, target: TargetState) -> np.ndarray:
        """
        Compute force during terminal phase - matches ship's velocity.

        Key insight: In terminal phase, we want to:
        1. Match the ship's vertical velocity (heave)
        2. Slowly descend relative to the deck
        3. Stay centered over helipad
        """
        rel_pos = vehicle.position - target.position
        rel_vel = vehicle.velocity - target.velocity

        # Horizontal position control (proportional)
        kp_horiz = 1.5
        horiz_force = -kp_horiz * rel_pos[:2]

        # Velocity damping
        kd_horiz = 2.0
        horiz_force -= kd_horiz * rel_vel[:2]

        # Vertical control: match ship velocity + slow descent
        desired_descent_rate = 0.3  # m/s relative to deck
        vertical_vel_error = rel_vel[2] - (-desired_descent_rate)

        # PD vertical control
        kp_vert = 3.0
        kd_vert = 4.0
        vert_force = -kp_vert * max(rel_pos[2] - 0.5, 0) - kd_vert * vertical_vel_error

        force = np.array([horiz_force[0], horiz_force[1], vert_force])
        force = MASS * force
        force[2] += HOVER_THRUST

        return force

    def compute_touchdown_force(self, vehicle: VehicleState, target: TargetState) -> np.ndarray:
        """
        Compute force during touchdown - gentle contact with velocity matching.
        """
        rel_pos = vehicle.position - target.position
        rel_vel = vehicle.velocity - target.velocity

        # Strong velocity matching
        kd = 5.0
        force = -kd * MASS * rel_vel

        # Very gentle position hold
        kp = 0.5
        force -= kp * MASS * rel_pos[:2].tolist() + [0]

        # Reduced thrust for settling
        force[2] += HOVER_THRUST * 0.7

        return force

    def run(self, max_time: float = 60.0, dt: float = 0.05):
        """Run the landing controller."""
        print("=" * 70)
        print("ROBUST TERMINAL PHASE LANDING CONTROLLER")
        print("=" * 70)
        print(f"Ship speed: 2.5 m/s (~5 knots)")
        print(f"Sea State: {SEA_STATE_4.significant_wave_height:.1f}m significant wave height")
        print(f"Phases: APPROACH -> DESCENT -> TERMINAL -> TOUCHDOWN -> LANDED")
        print()

        # Get initial state
        state = self.pose_reader.get_pose()
        if state is None:
            print("ERROR: Cannot read quadcopter pose")
            print(f"Start Gazebo with: gz sim -s -r worlds/{WORLD}.world")
            return False

        print(f"Initial position: ({state.position[0]:.1f}, {state.position[1]:.1f}, {state.position[2]:.1f})")
        self.start_time = time.time()

        print("\nStarting landing sequence...")
        print("-" * 70)

        t = 0
        last_print = 0

        while t < max_time:
            loop_start = time.time()
            sim_time = time.time() - self.start_time

            # Get vehicle state
            state = self.pose_reader.get_pose()
            if state is None:
                time.sleep(dt)
                t += dt
                continue

            # Get target
            target = self.get_target_state(sim_time)

            # Determine phase
            new_phase = self.determine_phase(state, target)
            if new_phase != self.phase:
                print(f"\n>>> Phase transition: {self.phase.value} -> {new_phase.value}")
                self.phase = new_phase

            # Check termination
            if self.phase == LandingPhase.LANDED:
                self.landed = True
                break
            if self.phase == LandingPhase.ABORT:
                self.abort_reason = "Safety limits exceeded"
                break

            # Compute control based on phase
            if self.phase in [LandingPhase.APPROACH, LandingPhase.DESCENT]:
                force = self.compute_approach_force(state, target)
            elif self.phase == LandingPhase.TERMINAL:
                force = self.compute_terminal_force(state, target)
            else:  # TOUCHDOWN
                force = self.compute_touchdown_force(state, target)

            # Apply force
            self.force_controller.apply_force(force)

            # Print status
            if time.time() - last_print > 0.5:
                rel_pos = state.position - target.position
                height = rel_pos[2]
                horiz = np.linalg.norm(rel_pos[:2])
                print(f"[{t:5.1f}s] {self.phase.value:10s} | "
                      f"Q:({state.position[0]:6.1f},{state.position[1]:5.1f},{state.position[2]:5.1f}) | "
                      f"V:({state.velocity[0]:+5.1f},{state.velocity[1]:+4.1f},{state.velocity[2]:+5.1f}) | "
                      f"H:{height:5.1f}m D:{horiz:5.1f}m | "
                      f"F:({force[0]:+5.1f},{force[1]:+5.1f},{force[2]:5.1f})")
                last_print = time.time()

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            t += dt

        # Cleanup
        self.force_controller.clear_force()

        print("\n" + "-" * 70)
        if self.landed:
            print("SUCCESS: Landing complete!")
            final = self.pose_reader.get_pose()
            if final:
                print(f"Final position: ({final.position[0]:.1f}, {final.position[1]:.1f}, {final.position[2]:.1f})")
        elif self.abort_reason:
            print(f"ABORT: {self.abort_reason}")
        else:
            print("TIMEOUT: Landing not completed")

        return self.landed


def main():
    controller = TerminalPhaseLandingController()
    success = controller.run()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
