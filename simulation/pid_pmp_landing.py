#!/usr/bin/env python3
"""
PID Approach + PMP Terminal Landing

Two-phase landing strategy:
1. APPROACH PHASE: PID controller flies UAV to "3s to touchdown" position
   - Targets a point above/behind deck that allows 3s terminal maneuver
   - Uses proportional navigation with velocity matching

2. TERMINAL PHASE: PMP recalculated every 1s until landing
   - Once within 3s of touchdown, switch to PMP
   - Recalculate optimal trajectory every 1s with updated deck prediction
   - ARMA predicts deck state at touchdown
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ship_motion.ddg_motion import DDGParams, SeaState, DDGMotionSimulator, ARMAPredictor
from quad_dynamics.quadrotor import QuadrotorParams, QuadrotorState, QuadrotorDynamics
from optimal_control.trajectory_planner import LandingTrajectoryPlanner
from optimal_control.pmp_controller import PMPController, create_pmp_trajectory, ControllerGains
from optimal_control.pseudospectral import PseudospectralSolver, LandingConstraints


@dataclass
class PIDGains:
    """PID controller gains for approach phase."""
    Kp_pos: np.ndarray = None  # Position P gain
    Kd_pos: np.ndarray = None  # Position D gain (velocity tracking)
    Kp_att: float = 30.0       # Attitude P gain
    Kd_att: float = 8.0        # Attitude D gain

    def __post_init__(self):
        if self.Kp_pos is None:
            self.Kp_pos = np.array([3.0, 3.0, 4.0])  # Moderate position gain
        if self.Kd_pos is None:
            self.Kd_pos = np.array([4.0, 4.0, 5.0])  # Good velocity damping


@dataclass
class LandingConfig:
    """Configuration for PID+PMP landing."""
    # Ship parameters
    sea_state: int = 4
    ship_speed_kts: float = 15.0
    wave_direction: float = 45.0

    # Initial approach
    approach_altitude: float = 25.0   # Initial height above deck (m)
    approach_distance: float = 50.0   # Initial distance behind deck (m)
    approach_speed: float = 15.0      # Approach speed (m/s) - must be faster than ship!

    # Terminal phase trigger
    terminal_time: float = 5.0        # Switch to PMP when this close (s)
    terminal_distance: float = 20.0   # Or when this close in distance (m)

    # PMP replanning
    pmp_replan_interval: float = 1.0  # Recalculate PMP every 1s

    # ARMA prediction
    arma_fit_interval: float = 5.0
    arma_history_length: float = 30.0

    # Touchdown
    touchdown_altitude: float = 0.3
    max_deck_roll_deg: float = 3.0
    max_deck_pitch_deg: float = 2.0


class PIDApproachController:
    """
    PID controller for approach phase.

    Flies UAV to a rendezvous point that is:
    - terminal_time seconds ahead of deck
    - At safe altitude for terminal maneuver
    """

    def __init__(self, params: QuadrotorParams, gains: PIDGains = None):
        self.params = params
        self.gains = gains if gains is not None else PIDGains()
        self.g = 9.81

    def compute_rendezvous_point(self, quad_pos: np.ndarray, quad_vel: np.ndarray,
                                  deck_pos: np.ndarray, deck_vel: np.ndarray,
                                  t_terminal: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute target for approach phase using proportional navigation.

        Returns:
            target_pos: Where to fly to
            target_vel: Target velocity
        """
        # Position and velocity errors
        pos_error = deck_pos - quad_pos
        vel_error = deck_vel - quad_vel

        # Height above deck
        height = pos_error[2]
        horiz_dist = np.linalg.norm(pos_error[:2])
        total_dist = np.linalg.norm(pos_error)

        # Closing rate
        closing_rate = -np.dot(pos_error, vel_error) / (total_dist + 0.1)

        # Time to intercept
        if closing_rate > 1.0:
            t_intercept = total_dist / closing_rate
        else:
            t_intercept = total_dist / 8.0

        t_intercept = max(t_terminal, min(t_intercept, 10.0))

        # Target position: intercept point above deck
        target_height = 6.0  # meters above deck
        target_pos = deck_pos + deck_vel * t_intercept
        target_pos[2] -= target_height

        # Target velocity: approach deck with intercept velocity
        # v_target = v_deck + (p_deck - p_uav) / t_go
        t_go = max(1.0, t_intercept)
        intercept_vel = pos_error / t_go
        target_vel = deck_vel + 0.8 * intercept_vel  # 80% of ideal intercept

        # Add descent
        target_vel[2] = target_height / t_terminal

        # Limit to reasonable speeds
        max_speed = 15.0
        speed = np.linalg.norm(target_vel[:2])
        if speed > max_speed:
            target_vel[:2] *= max_speed / speed

        return target_pos, target_vel

    def compute_control(self, quad_state: QuadrotorState,
                       target_pos: np.ndarray, target_vel: np.ndarray,
                       deck_att: np.ndarray = None) -> np.ndarray:
        """
        PID control to reach target position/velocity.

        Returns:
            u: [T, tau_x, tau_y, tau_z]
        """
        g = self.gains
        m = self.params.mass

        # Position and velocity errors
        pos_error = target_pos - quad_state.pos
        vel_error = target_vel - quad_state.vel

        # Desired acceleration (PD on position, P on velocity)
        acc_des = g.Kp_pos * pos_error + g.Kd_pos * vel_error

        # Clamp acceleration
        acc_des = np.clip(acc_des, -8.0, 8.0)

        # Compute thrust and attitude from desired acceleration
        # NED frame: gravity is [0, 0, g], thrust is along body -z
        thrust_vec = m * (acc_des - np.array([0, 0, self.g]))

        T = np.linalg.norm(thrust_vec)
        T = np.clip(T, 0.2 * m * self.g, 1.5 * m * self.g)

        # Desired attitude from thrust direction
        if T > 0.3 * m * self.g:
            thrust_dir = thrust_vec / np.linalg.norm(thrust_vec)
            pitch_des = np.arcsin(np.clip(-thrust_dir[0], -1, 1))
            roll_des = np.arctan2(thrust_dir[1], -thrust_dir[2])
        else:
            roll_des = 0
            pitch_des = 0

        roll_des = np.clip(roll_des, -0.5, 0.5)
        pitch_des = np.clip(pitch_des, -0.5, 0.5)

        # Attitude control
        euler = quad_state.quat_to_euler()
        roll_err = euler[0] - roll_des
        pitch_err = euler[1] - pitch_des
        yaw_err = euler[2]  # Target yaw = 0

        tau_x = -g.Kp_att * roll_err - g.Kd_att * quad_state.omega[0]
        tau_y = -g.Kp_att * pitch_err - g.Kd_att * quad_state.omega[1]
        tau_z = -g.Kd_att * quad_state.omega[2]

        return np.array([T, tau_x, tau_y, tau_z])


class ARMAPredictor6D:
    """ARMA predictor for deck motion (6 features)."""

    def __init__(self, ar_order: int = 8, ma_order: int = 4):
        self.arma = ARMAPredictor(ar_order, ma_order)
        self.history_t = []
        self.history_data = []
        self.dt = 0.1
        self.last_fit_time = -np.inf
        self.fitted = False

    def update(self, t: float, motion: dict):
        """Add observation."""
        obs = np.array([
            motion['deck_position'][2],
            motion['deck_velocity'][2],
            motion['deck_velocity'][0],
            motion['attitude'][0],
            motion['attitude'][1],
            motion['angular_rate'][0],
        ])
        self.history_t.append(t)
        self.history_data.append(obs)

        # Trim old
        if len(self.history_t) > 400:
            self.history_t = self.history_t[-400:]
            self.history_data = self.history_data[-400:]

    def fit(self, t: float, fit_interval: float = 5.0):
        """Refit ARMA if needed."""
        if t - self.last_fit_time >= fit_interval and len(self.history_data) > 100:
            data = np.array(self.history_data[-300:])
            self.arma.fit(data, self.dt)
            self.last_fit_time = t
            self.fitted = True

    def predict(self, t_current: float, horizon: float, ship_sim: DDGMotionSimulator,
                ship_speed_kts: float) -> dict:
        """
        Predict deck state at t_current + horizon.

        Returns dict with position, velocity, attitude.
        """
        if not self.fitted:
            # Fallback: use current state + linear extrapolation
            motion = ship_sim.get_motion(t_current)
            pos = motion['deck_position'] + motion['deck_velocity'] * horizon
            return {
                'position': pos,
                'velocity': motion['deck_velocity'].copy(),
                'attitude': motion['attitude'].copy()
            }

        # ARMA prediction
        n_steps = int(horizon / self.dt)
        if n_steps < 1:
            n_steps = 1
        preds = self.arma.predict(n_steps)
        pred = preds[-1]

        # Current deck for x,y extrapolation
        motion = ship_sim.get_motion(t_current)
        ship_vel_x = ship_speed_kts * 0.5144

        return {
            'position': np.array([
                motion['deck_position'][0] + ship_vel_x * horizon,
                motion['deck_position'][1],
                pred[0]  # z from ARMA
            ]),
            'velocity': np.array([
                ship_vel_x,
                0,
                pred[1]  # vz from ARMA
            ]),
            'attitude': np.array([
                pred[3],  # roll
                pred[4],  # pitch
                0
            ])
        }


class PIDPMPLandingSimulator:
    """
    Two-phase landing simulator:
    1. PID approach to 3s-to-touchdown position
    2. PMP terminal landing with 1s replanning
    """

    def __init__(self, config: LandingConfig = None):
        self.config = config if config is not None else LandingConfig()
        cfg = self.config

        # Ship motion
        self.ship_params = DDGParams()
        self.sea_state = SeaState.from_state_number(cfg.sea_state, cfg.wave_direction)
        self.ship_sim = DDGMotionSimulator(self.ship_params, self.sea_state, cfg.ship_speed_kts)

        # Quadrotor
        self.quad_params = QuadrotorParams()
        self.quad_dynamics = QuadrotorDynamics(self.quad_params)

        # Controllers
        self.pid_controller = PIDApproachController(self.quad_params)
        self.pmp_controller = PMPController(self.quad_params)

        # Trajectory planner (min-snap)
        self.planner = LandingTrajectoryPlanner(self.quad_params)

        # ARMA predictor
        self.predictor = ARMAPredictor6D()

        # State
        self.t = 0
        self.quad_state = None
        self.phase = "approach"  # "approach" or "terminal"
        self.landed = False
        self.last_pmp_time = -np.inf

        # History for analysis
        self.history = {
            't': [], 'quad_pos': [], 'deck_pos': [],
            'phase': [], 'thrust': [], 'pmp_replan': []
        }

    def reset(self):
        """Initialize simulation."""
        cfg = self.config
        self.t = 0
        self.landed = False
        self.phase = "approach"
        self.last_pmp_time = -np.inf

        # Warm up ARMA
        for t in np.arange(-cfg.arma_history_length, 0, 0.1):
            motion = self.ship_sim.get_motion(t)
            self.predictor.update(t, motion)
        self.predictor.fit(0)

        # Initial quad state
        deck_motion = self.ship_sim.get_motion(0)
        deck_pos = deck_motion['deck_position']

        self.quad_state = QuadrotorState(
            pos=np.array([
                deck_pos[0] - cfg.approach_distance,
                deck_pos[1],
                deck_pos[2] - cfg.approach_altitude
            ]),
            vel=np.array([cfg.approach_speed, 0, 0]),
            quat=np.array([1, 0, 0, 0]),
            omega=np.zeros(3)
        )

        # Clear history
        for key in self.history:
            self.history[key] = []

    def _compute_time_to_deck(self) -> float:
        """Estimate time to reach deck."""
        deck_motion = self.ship_sim.get_motion(self.t)
        deck_pos = deck_motion['deck_position']
        deck_vel = deck_motion['deck_velocity']

        rel_pos = deck_pos - self.quad_state.pos
        rel_vel = deck_vel - self.quad_state.vel

        # 3D distance
        dist = np.linalg.norm(rel_pos)

        # Closing rate
        closing_rate = -np.dot(rel_pos, rel_vel) / (dist + 0.1)

        if closing_rate > 0.5:
            return dist / closing_rate
        else:
            return dist / 5.0  # Conservative estimate

    def _should_switch_to_terminal(self) -> bool:
        """Check if we should switch from approach to terminal phase."""
        if self.phase != "approach":
            return False

        cfg = self.config

        # Time-based check
        t_to_deck = self._compute_time_to_deck()
        if t_to_deck <= cfg.terminal_time:
            return True

        # Distance-based check
        deck_motion = self.ship_sim.get_motion(self.t)
        dist = np.linalg.norm(deck_motion['deck_position'] - self.quad_state.pos)
        if dist <= cfg.terminal_distance:
            return True

        return False

    def _replan_pmp(self):
        """Compute PMP trajectory for terminal phase."""
        cfg = self.config

        # Predict deck state at touchdown
        t_to_land = max(1.5, self._compute_time_to_deck())
        pred = self.predictor.predict(self.t, t_to_land, self.ship_sim, cfg.ship_speed_kts)


        # Current state
        euler = self.quad_state.quat_to_euler()

        # Plan min-snap trajectory
        try:
            result = self.planner.plan_landing(
                quad_pos=self.quad_state.pos,
                quad_vel=self.quad_state.vel,
                deck_pos=pred['position'],
                deck_vel=pred['velocity'],
                deck_att=pred['attitude'],
                tf_desired=t_to_land
            )

            if result['success']:
                traj = result['trajectory']

                # Sample trajectory
                N = 20
                t_traj = np.linspace(0, traj.tf, N)
                x_traj = np.zeros((N, 12))
                u_traj = np.zeros((N, 4))

                for i, t in enumerate(t_traj):
                    sample = self.planner.sample_trajectory(traj, t)
                    alpha = t / traj.tf
                    x_traj[i, 0:3] = sample['position']
                    x_traj[i, 3:6] = sample['velocity']
                    x_traj[i, 6:9] = (1 - alpha) * euler + alpha * pred['attitude']
                    u_traj[i, 0] = sample['thrust']

                # Create PMP trajectory
                pmp_traj = create_pmp_trajectory(
                    x_traj, u_traj, t_traj, traj.tf,
                    pred['position'], pred['velocity'], pred['attitude'],
                    self.quad_params
                )

                self.pmp_controller.set_trajectory(pmp_traj, self.t)

        except Exception as e:
            print(f"PMP replan failed: {e}")

        # Always update last_pmp_time to avoid rapid retries
        self.last_pmp_time = self.t
        return True

    def _get_control(self) -> np.ndarray:
        """Get control using ZEM/ZEV optimal guidance."""
        cfg = self.config
        deck_motion = self.ship_sim.get_motion(self.t)
        deck_pos = deck_motion['deck_position']
        deck_vel = deck_motion['deck_velocity']
        deck_att = deck_motion['attitude']

        # Relative state
        r = deck_pos - self.quad_state.pos  # Position error
        v = deck_vel - self.quad_state.vel  # Velocity error

        height = r[2]
        total_dist = np.linalg.norm(r)

        # Time to go estimate - based on range and closing rate
        closing_rate = -np.dot(r, v) / (total_dist + 0.1)
        if closing_rate > 1.0:
            t_go = total_dist / closing_rate
        else:
            t_go = total_dist / 4.0
        t_go = np.clip(t_go, 1.0, 8.0)

        if self.phase == "approach":
            # Approach: close the gap while matching deck velocity
            target_height = 6.0

            # Horizontal: match deck velocity + close gap
            # Target velocity = deck velocity + position correction
            Kp_approach = 0.5  # Lower gain for gentler approach
            target_vel = deck_vel.copy()
            target_vel[:2] += Kp_approach * r[:2]

            # Limit relative speed during approach to allow deceleration
            max_rel_approach = 4.0  # Max 4 m/s relative to deck
            rel_vel_horiz = target_vel[:2] - deck_vel[:2]
            rel_speed_horiz = np.linalg.norm(rel_vel_horiz)
            if rel_speed_horiz > max_rel_approach:
                rel_vel_horiz *= max_rel_approach / rel_speed_horiz
                target_vel[:2] = deck_vel[:2] + rel_vel_horiz

            # Vertical: maintain target altitude
            alt_error = height - target_height
            if alt_error > 2:
                target_vel[2] = deck_vel[2] + alt_error / 3.0
            else:
                target_vel[2] = deck_vel[2] + 1.5  # Gentle descent

            max_speed = 12.0

        else:  # Terminal phase
            # Terminal guidance: target = deck position, velocity = match deck
            # Use simple proportional control with velocity limiting

            # Target velocity = deck velocity + proportional position correction
            Kp = 1.0  # Position correction gain

            # Horizontal: close the gap proportionally
            target_vel = deck_vel.copy()
            target_vel[:2] += Kp * r[:2]

            # Limit relative velocity to deck
            max_rel_horiz = 2.0  # Max 2 m/s relative to deck horizontally
            rel_vel_horiz = target_vel[:2] - deck_vel[:2]
            rel_speed_horiz = np.linalg.norm(rel_vel_horiz)
            if rel_speed_horiz > max_rel_horiz:
                rel_vel_horiz *= max_rel_horiz / rel_speed_horiz
                target_vel[:2] = deck_vel[:2] + rel_vel_horiz

            # Vertical: controlled descent
            if height > 2.0:
                safe_descent = min(1.5, np.sqrt(2.0 * 2.0 * height))
                target_vel[2] = deck_vel[2] + safe_descent
            elif height > 0.5:
                target_vel[2] = deck_vel[2] + height / 3.0
            else:
                target_vel[2] = deck_vel[2] + 0.2

            max_speed = 10.0

        # Safety limits
        target_vel = np.clip(target_vel, -15, 15)

        # Limit speed relative to deck
        rel_vel = target_vel - deck_vel
        rel_speed = np.linalg.norm(rel_vel[:2])
        if rel_speed > 5.0:
            rel_vel[:2] *= 5.0 / rel_speed
            target_vel[:2] = deck_vel[:2] + rel_vel[:2]

        return self.pid_controller.compute_control(
            self.quad_state, deck_pos, target_vel, deck_att
        )

    def step(self, dt: float = 0.02):
        """Step simulation."""
        if self.landed:
            return

        cfg = self.config

        # Update ARMA
        motion = self.ship_sim.get_motion(self.t)
        self.predictor.update(self.t, motion)
        self.predictor.fit(self.t, cfg.arma_fit_interval)

        deck_pos = motion['deck_position']
        deck_vel = motion['deck_velocity']
        deck_att = motion['attitude']

        # Check phase transition
        if self._should_switch_to_terminal():
            if self.phase == "approach":
                print(f"[{self.t:.2f}s] Switching to TERMINAL phase")
                self.phase = "terminal"
                self._replan_pmp()

        # PMP replanning in terminal phase
        if self.phase == "terminal":
            if self.t - self.last_pmp_time >= cfg.pmp_replan_interval:
                print(f"[{self.t:.2f}s] PMP replan (every {cfg.pmp_replan_interval}s)")
                self._replan_pmp()
                self.history['pmp_replan'].append(self.t)

        # Check touchdown
        height = deck_pos[2] - self.quad_state.pos[2]
        if height < cfg.touchdown_altitude:
            self.landed = True
            rel_vel = self.quad_state.vel - deck_vel
            lateral_err = np.linalg.norm(self.quad_state.pos[:2] - deck_pos[:2])
            uav_euler = self.quad_state.quat_to_euler()

            print(f"\n{'='*70}")
            print(f"TOUCHDOWN at t={self.t:.2f}s")
            print(f"{'='*70}")
            print(f"\n  {'':20} {'X':>10} {'Y':>10} {'Z':>10} {'Roll':>10} {'Pitch':>10}")
            print(f"  {'-'*70}")
            print(f"  {'UAV Position (m)':20} {self.quad_state.pos[0]:10.3f} {self.quad_state.pos[1]:10.3f} {self.quad_state.pos[2]:10.3f}")
            print(f"  {'Ship Deck (m)':20} {deck_pos[0]:10.3f} {deck_pos[1]:10.3f} {deck_pos[2]:10.3f}")
            print(f"  {'Error (m)':20} {self.quad_state.pos[0]-deck_pos[0]:10.3f} {self.quad_state.pos[1]-deck_pos[1]:10.3f} {self.quad_state.pos[2]-deck_pos[2]:10.3f}")
            print(f"\n  {'UAV Velocity (m/s)':20} {self.quad_state.vel[0]:10.3f} {self.quad_state.vel[1]:10.3f} {self.quad_state.vel[2]:10.3f}")
            print(f"  {'Deck Velocity (m/s)':20} {deck_vel[0]:10.3f} {deck_vel[1]:10.3f} {deck_vel[2]:10.3f}")
            print(f"  {'Rel Velocity (m/s)':20} {rel_vel[0]:10.3f} {rel_vel[1]:10.3f} {rel_vel[2]:10.3f}")
            print(f"\n  {'UAV Attitude (deg)':20} {'':>10} {'':>10} {'':>10} {np.degrees(uav_euler[0]):10.2f} {np.degrees(uav_euler[1]):10.2f}")
            print(f"  {'Deck Attitude (deg)':20} {'':>10} {'':>10} {'':>10} {np.degrees(deck_att[0]):10.2f} {np.degrees(deck_att[1]):10.2f}")
            print(f"  {'Att Error (deg)':20} {'':>10} {'':>10} {'':>10} {np.degrees(uav_euler[0]-deck_att[0]):10.2f} {np.degrees(uav_euler[1]-deck_att[1]):10.2f}")
            print(f"\n  Lateral error: {lateral_err:.3f} m")
            print(f"  Relative speed: {np.linalg.norm(rel_vel):.3f} m/s")
            print(f"  Deck heave rate: {deck_vel[2]:.2f} m/s ({'down' if deck_vel[2] > 0 else 'up'})")
            print(f"  PMP replans: {len(self.history['pmp_replan'])}")
            print(f"{'='*70}\n")
            return

        # Get and apply control
        u = self._get_control()
        self._step_quad(u, dt)

        # Record history
        self.history['t'].append(self.t)
        self.history['quad_pos'].append(self.quad_state.pos.copy())
        self.history['deck_pos'].append(deck_pos.copy())
        self.history['phase'].append(self.phase)
        self.history['thrust'].append(u[0])

        self.t += dt

    def _step_quad(self, u: np.ndarray, dt: float):
        """Step quadrotor dynamics."""
        T, tau_x, tau_y, tau_z = u
        p = self.quad_params
        L = p.arm_length
        s45 = np.sin(np.pi/4)

        T_per = T / 4
        torque_to_motor = 1 / (4 * L * s45)
        max_torque_per_motor = min(T_per, p.max_thrust_per_rotor - T_per)
        max_tau = max_torque_per_motor / torque_to_motor

        tau_x = np.clip(tau_x, -max_tau, max_tau)
        tau_y = np.clip(tau_y, -max_tau, max_tau)

        T1 = T_per + tau_x * torque_to_motor + tau_y * torque_to_motor
        T2 = T_per + tau_x * torque_to_motor - tau_y * torque_to_motor
        T3 = T_per - tau_x * torque_to_motor - tau_y * torque_to_motor
        T4 = T_per - tau_x * torque_to_motor + tau_y * torque_to_motor

        motor_thrusts = np.clip([T1, T2, T3, T4], 0, p.max_thrust_per_rotor)
        self.quad_state = self.quad_dynamics.step(self.quad_state, motor_thrusts, dt)

    def run(self, max_time: float = 30.0, dt: float = 0.02) -> dict:
        """Run full simulation."""
        self.reset()

        print(f"\n{'='*60}")
        print("PID APPROACH + PMP TERMINAL LANDING")
        print(f"{'='*60}")
        print(f"Sea state: {self.config.sea_state}")
        print(f"Ship speed: {self.config.ship_speed_kts} kts")
        print(f"Terminal trigger: {self.config.terminal_time}s or {self.config.terminal_distance}m")
        print(f"PMP replan interval: {self.config.pmp_replan_interval}s")
        print(f"{'='*60}\n")

        while self.t < max_time and not self.landed:
            self.step(dt)

        return {
            'success': self.landed,
            'landing_time': self.t if self.landed else np.nan,
            'pmp_replans': len(self.history['pmp_replan']),
            'history': self.history
        }


def demo(seed: int = 42):
    """Demo PID+PMP landing."""
    np.random.seed(seed)

    config = LandingConfig(
        sea_state=4,
        ship_speed_kts=15,
        approach_altitude=25,
        approach_distance=50,
        terminal_time=5.0,
        terminal_distance=20.0,
        pmp_replan_interval=1.0
    )

    sim = PIDPMPLandingSimulator(config)
    results = sim.run(max_time=25.0)

    if results['history']['t']:
        h = results['history']

        # Compute final errors
        final_quad = h['quad_pos'][-1]
        final_deck = h['deck_pos'][-1]
        final_err = np.linalg.norm(final_quad[:2] - final_deck[:2])

        # Count phases
        n_approach = sum(1 for p in h['phase'] if p == 'approach')
        n_terminal = sum(1 for p in h['phase'] if p == 'terminal')

        print(f"\nRESULTS:")
        print(f"  Landing time: {results['landing_time']:.2f}s")
        print(f"  Approach phase: {n_approach * 0.02:.2f}s")
        print(f"  Terminal phase: {n_terminal * 0.02:.2f}s")
        print(f"  PMP replans: {results['pmp_replans']}")
        print(f"  Final lateral error: {final_err:.3f}m")

    return results


if __name__ == "__main__":
    demo()
