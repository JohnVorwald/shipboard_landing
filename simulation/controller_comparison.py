#!/usr/bin/env python3
"""
Controller Comparison for Shipboard Landing

Compares multiple guidance and control strategies:
1. ZEM/ZEV - Zero Effort Miss/Velocity (baseline)
2. Tau-Based - Time-to-contact guidance (Penn State)
3. PMP - Pontryagin Minimum Principle with trajectory tracking
4. Proportional Navigation - Classic pursuit guidance

Performance metrics:
- Position error at touchdown
- Velocity error at touchdown
- Success rate (all criteria met)
- Fuel consumption (integral of thrust)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Tuple, Dict, List, Callable
from enum import Enum

from quad_dynamics.quadrotor import QuadrotorParams, QuadrotorState
from ship_motion.ddg_motion import DDGParams, SeaState, DDGMotionSimulator, ARMAPredictor
from guidance.tau_guidance import TauGuidanceController, TauGuidanceConfig
from optimal_control.trajectory_planner import LandingTrajectoryPlanner
from optimal_control.pmp_controller import PMPController, create_pmp_trajectory, ControllerGains
from optimal_control.variable_horizon_mpc import VariableHorizonMPC, VHMPCConfig


class ControllerType(Enum):
    ZEM_ZEV = "ZEM/ZEV"
    TAU = "Tau-Based"
    VH_MPC = "VH-MPC"
    PMP = "PMP"
    PROP_NAV = "Prop Nav"


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
class SimConfig:
    """Simulation configuration."""
    sea_state: int = 4
    ship_speed_kts: float = 15.0
    initial_distance: float = 80.0      # Behind ship (m)
    initial_altitude: float = 25.0      # Above deck (m)
    approach_time: float = 60.0         # Max approach time (s)
    dt: float = 0.02                    # Time step (s)
    warmup_time: float = 30.0           # ARMA warmup (s)


class ZEMZEVController:
    """ZEM/ZEV optimal interception guidance."""

    def __init__(self, params: QuadrotorParams):
        self.params = params
        self.g = 9.81

    def compute_control(self, state: QuadrotorState, deck_pos: np.ndarray,
                       deck_vel: np.ndarray, deck_att: np.ndarray,
                       t_go: float) -> np.ndarray:
        """Compute control using ZEM/ZEV."""
        rel_pos = state.pos - deck_pos
        rel_vel = state.vel - deck_vel
        dist = np.linalg.norm(rel_pos)
        height = -rel_pos[2]

        # Estimate proper time-to-go based on distance and safe approach speed
        # Target approach: ~8 m/s horizontal, ~2 m/s vertical
        horiz_dist = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
        t_go_horiz = horiz_dist / 8.0 + 2.0
        t_go_vert = max(height, 0) / 2.0 + 2.0
        t_go = max(min(t_go_horiz, t_go_vert), 3.0)  # At least 3s

        if dist > 10.0:
            # Far: ZEM/ZEV interception
            a_zem = -6.0 * (rel_pos + rel_vel * t_go) / t_go**2
            a_zev = -2.0 * rel_vel / t_go
            acc_cmd = a_zem + a_zev
        else:
            # Near: Aggressive PD control for terminal guidance
            # Scale gains with proximity for tight tracking
            scale = 1.0 + 1.0 * (1.0 - dist / 10.0)  # Up to 2x gain at close range
            K_pos = 3.0 * scale
            K_vel = 6.0 * scale  # Strong velocity damping for matching
            acc_cmd = -K_pos * rel_pos - K_vel * rel_vel

        # Limit acceleration
        acc_horiz = np.sqrt(acc_cmd[0]**2 + acc_cmd[1]**2)
        if acc_horiz > 8.0:
            acc_cmd[0] *= 8.0 / acc_horiz
            acc_cmd[1] *= 8.0 / acc_horiz
        acc_cmd[2] = np.clip(acc_cmd[2], -6.0, 6.0)

        return self._acc_to_control(acc_cmd, state, deck_att, height)

    def _acc_to_control(self, acc_cmd: np.ndarray, state: QuadrotorState,
                        deck_att: np.ndarray, height: float) -> np.ndarray:
        """Convert acceleration to thrust/attitude."""
        m = self.params.mass
        yaw = state.yaw

        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        a_forward = cos_yaw * acc_cmd[0] + sin_yaw * acc_cmd[1]
        a_right = -sin_yaw * acc_cmd[0] + cos_yaw * acc_cmd[1]

        pitch_cmd = np.clip(np.arctan2(a_forward, self.g), -0.5, 0.5)
        roll_cmd = np.clip(np.arctan2(-a_right, self.g), -0.5, 0.5)

        # Blend to deck attitude near touchdown
        if height < 3.0 and height > 0:
            blend = 1 - height / 3.0
            roll_cmd = (1 - blend) * roll_cmd + blend * deck_att[0]
            pitch_cmd = (1 - blend) * pitch_cmd + blend * deck_att[1]

        # NED: positive acc_cmd[2] = descend = need less thrust
        thrust = m * (self.g - acc_cmd[2]) / (np.cos(pitch_cmd) * np.cos(roll_cmd))
        thrust = np.clip(thrust, 0.1 * m * self.g, 2.0 * m * self.g)

        return np.array([thrust, roll_cmd, pitch_cmd])


class PropNavController:
    """Proportional navigation guidance."""

    def __init__(self, params: QuadrotorParams):
        self.params = params
        self.g = 9.81
        self.N = 4.0  # Navigation constant

    def compute_control(self, state: QuadrotorState, deck_pos: np.ndarray,
                       deck_vel: np.ndarray, deck_att: np.ndarray,
                       t_go: float) -> np.ndarray:
        """Compute control using proportional navigation."""
        rel_pos = deck_pos - state.pos  # Vector to target
        rel_vel = deck_vel - state.vel

        range_sq = np.dot(rel_pos, rel_pos)
        range_val = np.sqrt(range_sq)
        closing_rate = -np.dot(rel_pos, rel_vel) / (range_val + 0.1)

        if range_val < 0.1:
            return np.array([self.params.mass * self.g, 0, 0])

        # Line of sight rate
        # ω_los = (r × v) / |r|²
        los_rate = np.cross(rel_pos, rel_vel) / (range_sq + 1.0)

        # PN acceleration: a = N * V_c * ω_los
        acc_pn = self.N * closing_rate * los_rate

        # Add velocity matching component
        height = rel_pos[2]
        t_to_intercept = range_val / (closing_rate + 1.0)
        t_to_intercept = np.clip(t_to_intercept, 2.0, 15.0)

        vel_error = deck_vel - state.vel
        acc_vel_match = 0.5 * vel_error / t_to_intercept

        acc_cmd = acc_pn + acc_vel_match
        acc_cmd = np.clip(acc_cmd, -6, 6)

        return self._acc_to_control(acc_cmd, state, deck_att, -height)

    def _acc_to_control(self, acc_cmd: np.ndarray, state: QuadrotorState,
                        deck_att: np.ndarray, height: float) -> np.ndarray:
        """Convert acceleration to thrust/attitude."""
        m = self.params.mass
        yaw = state.yaw

        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        a_forward = cos_yaw * acc_cmd[0] + sin_yaw * acc_cmd[1]
        a_right = -sin_yaw * acc_cmd[0] + cos_yaw * acc_cmd[1]

        pitch_cmd = np.clip(np.arctan2(a_forward, self.g), -0.5, 0.5)
        roll_cmd = np.clip(np.arctan2(-a_right, self.g), -0.5, 0.5)

        if height < 3.0 and height > 0:
            blend = 1 - height / 3.0
            roll_cmd = (1 - blend) * roll_cmd + blend * deck_att[0]
            pitch_cmd = (1 - blend) * pitch_cmd + blend * deck_att[1]

        # NED: positive acc_cmd[2] = descend = need less thrust
        thrust = m * (self.g - acc_cmd[2]) / (np.cos(pitch_cmd) * np.cos(roll_cmd))
        thrust = np.clip(thrust, 0.1 * m * self.g, 2.0 * m * self.g)

        return np.array([thrust, roll_cmd, pitch_cmd])


class TauController:
    """Wrapper for tau-based guidance."""

    def __init__(self, params: QuadrotorParams):
        self.params = params
        self.tau_controller = TauGuidanceController(quad_params=params)
        self.g = 9.81

    def compute_control(self, state: QuadrotorState, deck_pos: np.ndarray,
                       deck_vel: np.ndarray, deck_att: np.ndarray,
                       t_go: float) -> np.ndarray:
        """Compute control using tau guidance."""
        acc_cmd, info = self.tau_controller.compute_control(
            state.pos, state.vel, deck_pos, deck_vel, deck_att
        )

        thrust, roll_cmd, pitch_cmd = self.tau_controller.compute_thrust_attitude(
            acc_cmd, state.yaw, deck_att, info['height']
        )

        return np.array([thrust, roll_cmd, pitch_cmd])


class VHMPCController:
    """Wrapper for Variable Horizon MPC."""

    def __init__(self, params: QuadrotorParams):
        self.params = params
        config = VHMPCConfig(
            horizon_options=[20, 30, 40],
            move_blocks=[5, 5, 5, 5],
            dt=0.1
        )
        self.vhmpc = VariableHorizonMPC(params, config)
        self.g = 9.81

    def compute_control(self, state: QuadrotorState, deck_pos: np.ndarray,
                       deck_vel: np.ndarray, deck_att: np.ndarray,
                       t_go: float, arma_predict: callable = None) -> np.ndarray:
        """Compute control using VH-MPC."""
        x = np.concatenate([state.pos, state.vel,
                           np.array([state.roll, state.pitch, state.yaw])])

        acc_cmd, info = self.vhmpc.compute_control(
            x, deck_pos, deck_vel, deck_att, t_go, arma_predict
        )

        height = -(state.pos[2] - deck_pos[2])
        thrust, roll_cmd, pitch_cmd = self.vhmpc.acc_to_thrust_attitude(
            acc_cmd, state.yaw, deck_att, height
        )

        return np.array([thrust, roll_cmd, pitch_cmd])


class PMPControllerWrapper:
    """Min-snap trajectory tracking with ZEM/ZEV-style guidance."""

    def __init__(self, params: QuadrotorParams):
        self.params = params
        self.g = 9.81

    def compute_control(self, state: QuadrotorState, deck_pos: np.ndarray,
                       deck_vel: np.ndarray, deck_att: np.ndarray,
                       t_go: float, current_time: float = 0) -> np.ndarray:
        """Compute control using ZEM/ZEV with coordinated descent."""

        rel_pos = state.pos - deck_pos
        rel_vel = state.vel - deck_vel
        dist = np.linalg.norm(rel_pos)
        height = -rel_pos[2]
        horiz_dist = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)

        # Estimate time-to-go based on horizontal distance
        t_go = max(horiz_dist / 8.0 + 2.0, 3.0)

        if dist > 8.0:
            # FAR: ZEM/ZEV for interception
            a_zem = -6.0 * (rel_pos + rel_vel * t_go) / t_go**2
            a_zev = -2.0 * rel_vel / t_go
            acc_cmd = a_zem + a_zev

            # Coordinate descent: don't descend faster than approach
            # Keep some altitude until horizontally close
            if horiz_dist > 10.0 and height > 3.0:
                max_descent = horiz_dist / 15.0  # Descend slower when far
                if acc_cmd[2] > max_descent:
                    acc_cmd[2] = max_descent
        else:
            # NEAR: Tight PD tracking for landing
            scale = 1.0 + 1.0 * (1.0 - dist / 8.0)
            K_pos = 4.0 * scale
            K_vel = 6.0 * scale
            acc_cmd = -K_pos * rel_pos - K_vel * rel_vel

        # Limit accelerations
        acc_horiz = np.sqrt(acc_cmd[0]**2 + acc_cmd[1]**2)
        if acc_horiz > 8.0:
            acc_cmd[0] *= 8.0 / acc_horiz
            acc_cmd[1] *= 8.0 / acc_horiz
        acc_cmd[2] = np.clip(acc_cmd[2], -6.0, 6.0)

        return self._acc_to_control(acc_cmd, state, deck_att, height)

    def _acc_to_control(self, acc_cmd: np.ndarray, state: QuadrotorState,
                        deck_att: np.ndarray, height: float) -> np.ndarray:
        """Convert acceleration to thrust/attitude."""
        m = self.params.mass
        yaw = state.yaw

        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        a_forward = cos_yaw * acc_cmd[0] + sin_yaw * acc_cmd[1]
        a_right = -sin_yaw * acc_cmd[0] + cos_yaw * acc_cmd[1]

        pitch_cmd = np.clip(np.arctan2(a_forward, self.g), -0.5, 0.5)
        roll_cmd = np.clip(np.arctan2(-a_right, self.g), -0.5, 0.5)

        if height < 3.0 and height > 0:
            blend = 1 - height / 3.0
            roll_cmd = (1 - blend) * roll_cmd + blend * deck_att[0]
            pitch_cmd = (1 - blend) * pitch_cmd + blend * deck_att[1]

        # NED: positive acc_cmd[2] = descend = need less thrust
        thrust = m * (self.g - acc_cmd[2]) / (np.cos(pitch_cmd) * np.cos(roll_cmd))
        thrust = np.clip(thrust, 0.1 * m * self.g, 2.0 * m * self.g)

        return np.array([thrust, roll_cmd, pitch_cmd])


class ControllerComparisonSim:
    """Simulation comparing multiple controllers."""

    def __init__(self, config: SimConfig = None):
        self.config = config if config is not None else SimConfig()

        # Ship model
        self.ship_params = DDGParams()
        self.sea_state = SeaState.from_state_number(self.config.sea_state, direction=45.0)
        self.ship_sim = DDGMotionSimulator(
            self.ship_params, self.sea_state, self.config.ship_speed_kts
        )

        # Quad params
        self.quad_params = QuadrotorParams()

        # ARMA predictor
        self.arma = ARMAPredictor(ar_order=8, ma_order=4)

        # Controllers
        self.controllers = {
            ControllerType.ZEM_ZEV: ZEMZEVController(self.quad_params),
            ControllerType.TAU: TauController(self.quad_params),
            ControllerType.VH_MPC: VHMPCController(self.quad_params),
            ControllerType.PMP: PMPControllerWrapper(self.quad_params),
            ControllerType.PROP_NAV: PropNavController(self.quad_params),
        }

    def warmup_arma(self):
        """Warm up ARMA predictor."""
        history = []
        for t in np.arange(0, self.config.warmup_time, 0.1):
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
        self.arma.fit(np.array(history), 0.1)

    def run_single(self, controller_type: ControllerType, seed: int = 0) -> Dict:
        """Run single landing with specified controller."""
        np.random.seed(seed)
        cfg = self.config
        dt = cfg.dt

        # Initialize
        self.warmup_arma()

        # Initial quad state
        ship_motion = self.ship_sim.get_motion(cfg.warmup_time)
        deck_pos = ship_motion['deck_position']
        deck_vel = ship_motion['deck_velocity']

        quad_start_pos = deck_pos.copy()
        quad_start_pos[0] -= cfg.initial_distance
        quad_start_pos[2] = deck_pos[2] - cfg.initial_altitude

        state = QuadrotorState(
            pos=quad_start_pos,
            vel=deck_vel.copy(),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            omega=np.zeros(3)
        )

        controller = self.controllers[controller_type]

        # Run simulation
        results = {'time': [], 'pos_error': [], 'vel_error': [], 'thrust': []}
        t = cfg.warmup_time
        total_thrust = 0.0

        while t < cfg.approach_time:
            ship_motion = self.ship_sim.get_motion(t)
            deck_pos = ship_motion['deck_position']
            deck_vel = ship_motion['deck_velocity']
            deck_att = ship_motion['attitude']

            # Update ARMA
            obs = np.array([
                ship_motion['deck_position'][2],
                ship_motion['deck_velocity'][2],
                ship_motion['deck_velocity'][0],
                ship_motion['attitude'][0],
                ship_motion['attitude'][1],
                ship_motion['angular_rate'][0],
            ])
            self.arma.update(obs)

            # Compute control
            t_go = cfg.approach_time - t
            if controller_type == ControllerType.PMP:
                control = controller.compute_control(
                    state, deck_pos, deck_vel, deck_att, t_go, t
                )
            elif controller_type == ControllerType.VH_MPC:
                control = controller.compute_control(
                    state, deck_pos, deck_vel, deck_att, t_go, self.arma.predict
                )
            else:
                control = controller.compute_control(
                    state, deck_pos, deck_vel, deck_att, t_go
                )

            thrust, roll_cmd, pitch_cmd = control[0], control[1], control[2]

            # Simple attitude dynamics
            tau_att = 0.1
            roll_new = state.roll + (roll_cmd - state.roll) * dt / tau_att
            pitch_new = state.pitch + (pitch_cmd - state.pitch) * dt / tau_att

            # Yaw toward deck
            dx = deck_pos[0] - state.pos[0]
            dy = deck_pos[1] - state.pos[1]
            desired_yaw = np.arctan2(dy, dx)
            yaw_error = np.arctan2(np.sin(desired_yaw - state.yaw),
                                   np.cos(desired_yaw - state.yaw))
            yaw_new = state.yaw + np.clip(2.0 * yaw_error, -1.0, 1.0) * dt

            state.quat = euler_to_quat(roll_new, pitch_new, yaw_new)

            # Update velocity and position
            # NED frame: +x=North, +y=East, +z=Down
            # Gravity acts in +z, thrust acts in body -z (up when level)
            g = 9.81
            m = self.quad_params.mass

            # Thrust-induced acceleration (body -z rotated to world frame)
            # For small angles, thrust creates forward accel via pitch, lateral via roll
            ax = thrust / m * np.sin(pitch_new)
            ay = -thrust / m * np.sin(roll_new)
            az = g - thrust / m * np.cos(pitch_new) * np.cos(roll_new)

            state.vel += np.array([ax, ay, az]) * dt
            state.pos += state.vel * dt

            # Record
            pos_error = np.linalg.norm(state.pos - deck_pos)
            vel_error = np.linalg.norm(state.vel - deck_vel)
            results['time'].append(t)
            results['pos_error'].append(pos_error)
            results['vel_error'].append(vel_error)
            results['thrust'].append(thrust)
            total_thrust += thrust * dt

            # Check touchdown
            height = -(state.pos[2] - deck_pos[2])
            if height < 0.5:
                break

            t += dt

        # Final results
        final_pos_error = results['pos_error'][-1] if results['pos_error'] else float('inf')
        final_vel_error = results['vel_error'][-1] if results['vel_error'] else float('inf')

        success = final_pos_error < 3.0 and final_vel_error < 2.0

        return {
            'controller': controller_type.value,
            'pos_error': final_pos_error,
            'vel_error': final_vel_error,
            'success': success,
            'landing_time': t - cfg.warmup_time,
            'fuel': total_thrust,
            'trajectory': results
        }

    def run_comparison(self, n_trials: int = 5) -> Dict:
        """Run comparison across all controllers."""
        print("=" * 70)
        print("CONTROLLER COMPARISON FOR SHIPBOARD LANDING")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Sea State: {self.config.sea_state}")
        print(f"  Ship Speed: {self.config.ship_speed_kts} kts")
        print(f"  Initial Distance: {self.config.initial_distance} m")
        print(f"  Trials per controller: {n_trials}")

        all_results = {}

        for ctrl_type in ControllerType:
            print(f"\n{'─' * 70}")
            print(f"  Testing: {ctrl_type.value}")
            print(f"{'─' * 70}")

            results = {
                'pos_error': [], 'vel_error': [],
                'success': [], 'landing_time': [], 'fuel': []
            }

            for trial in range(n_trials):
                print(f"    Trial {trial + 1}/{n_trials}...", end="", flush=True)
                result = self.run_single(ctrl_type, seed=trial * 100)

                results['pos_error'].append(result['pos_error'])
                results['vel_error'].append(result['vel_error'])
                results['success'].append(result['success'])
                results['landing_time'].append(result['landing_time'])
                results['fuel'].append(result['fuel'])

                status = "OK" if result['success'] else "FAIL"
                print(f" {status} (pos: {result['pos_error']:.2f}m, vel: {result['vel_error']:.2f}m/s)")

            all_results[ctrl_type.value] = results

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("\n┌──────────────┬─────────────┬─────────────┬─────────────┬───────────┐")
        print("│  Controller  │  Pos Error  │  Vel Error  │ Success Rate│    Fuel   │")
        print("├──────────────┼─────────────┼─────────────┼─────────────┼───────────┤")

        for name, res in all_results.items():
            pos_mean = np.mean(res['pos_error'])
            vel_mean = np.mean(res['vel_error'])
            success_rate = 100 * np.mean(res['success'])
            fuel_mean = np.mean(res['fuel'])
            print(f"│ {name:12s} │   {pos_mean:6.2f} m  │  {vel_mean:6.2f} m/s │    {success_rate:5.1f}%   │ {fuel_mean:8.0f} │")

        print("└──────────────┴─────────────┴─────────────┴─────────────┴───────────┘")

        # Best controller
        best_success = max(all_results.items(),
                          key=lambda x: np.mean(x[1]['success']))
        best_pos = min(all_results.items(),
                      key=lambda x: np.mean(x[1]['pos_error']))

        print(f"\nBest success rate: {best_success[0]} ({100*np.mean(best_success[1]['success']):.1f}%)")
        print(f"Best position accuracy: {best_pos[0]} ({np.mean(best_pos[1]['pos_error']):.2f}m)")

        return all_results


def main():
    """Main entry point."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  AUTONOMOUS SHIPBOARD LANDING - CONTROLLER COMPARISON  ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70 + "\n")

    config = SimConfig(
        sea_state=4,
        ship_speed_kts=15.0,
        initial_distance=80.0,
        initial_altitude=25.0
    )

    sim = ControllerComparisonSim(config)
    results = sim.run_comparison(n_trials=5)

    print("\n" + "█" * 70)
    print("█" + "  COMPARISON COMPLETE  ".center(68) + "█")
    print("█" * 70 + "\n")

    return results


if __name__ == "__main__":
    main()
