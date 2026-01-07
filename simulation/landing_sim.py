"""
Integrated Shipboard Landing Simulation

Combines:
- DDG ship motion model
- ARMA deck motion prediction (fitted on recent past, forecasts future)
- Pseudospectral optimal trajectory to touchdown
- Cone constraint: UAV must stay within approach cone to deck
- Terminal conditions: match position, velocity, attitude, zero thrust

Architecture:
1. Every 5s: Refit ARMA on last 30s of ship motion
2. Every 0.5s: Predict deck state at 0.5s increments up to 5s
3. Solve PMP for trajectory matching touchdown conditions
4. Execute trajectory with feedback control
5. Cone constraint: UAV must approach within cone centered on deck
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ship_motion.ddg_motion import DDGParams, SeaState, DDGMotionSimulator, ARMAPredictor
from quad_dynamics.quadrotor import QuadrotorParams, QuadrotorState, QuadrotorDynamics, RotorShutdownModel
from optimal_control.pseudospectral import PseudospectralSolver, LandingConstraints


@dataclass
class LandingConfig:
    """Configuration for landing simulation."""
    # Ship parameters
    sea_state: int = 4
    ship_speed_kts: float = 15.0
    wave_direction: float = 45.0

    # Approach parameters
    approach_altitude: float = 30.0
    approach_distance: float = 50.0
    approach_speed: float = 10.0

    # ARMA prediction
    arma_fit_interval: float = 5.0      # Refit ARMA every 5s
    arma_history_length: float = 30.0    # Use 30s of past data
    prediction_dt: float = 0.5           # Predict at 0.5s increments
    prediction_horizon: float = 5.0      # Predict up to 5s ahead

    # Trajectory replanning
    replan_interval: float = 0.5        # Replan every 0.5s

    # Cone constraint
    cone_half_angle: float = 30.0       # Cone half-angle (degrees)
    cone_apex_offset: float = 5.0       # Apex above deck (m)

    # Touchdown
    touchdown_altitude: float = 0.3
    max_touchdown_velocity: float = 1.0
    shutdown_advance: float = 0.2

    # Touchdown window constraints
    max_deck_roll_deg: float = 3.0      # Max deck roll at touchdown
    max_deck_pitch_deg: float = 2.0     # Max deck pitch at touchdown
    deck_moving_down: bool = True        # Require deck moving down (heave velocity > 0 in NED)


@dataclass
class DeckPrediction:
    """Predicted deck state at future time."""
    t: float                    # Prediction time
    position: np.ndarray        # [x, y, z] NED
    velocity: np.ndarray        # [vx, vy, vz] NED
    attitude: np.ndarray        # [roll, pitch, yaw]
    angular_rate: np.ndarray    # [p, q, r]
    confidence: float           # Prediction confidence (0-1)


class ApproachCone:
    """
    Approach cone constraint.

    UAV must stay within cone centered on deck, tilted with ship attitude.
    Cone apex is above deck, cone opens downward/aft for approach.
    """

    def __init__(self, half_angle_deg: float = 30.0, apex_offset: float = 5.0):
        self.half_angle = np.radians(half_angle_deg)
        self.apex_offset = apex_offset

    def get_cone_params(self, deck_pos: np.ndarray, deck_att: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cone apex and axis given deck state.

        Returns:
            apex: Cone apex position (above deck)
            axis: Cone axis direction (pointing away from deck into approach zone)
        """
        roll, pitch, yaw = deck_att

        # Rotation matrix for deck attitude
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
            [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
            [-sp,   sr*cp,            cr*cp]
        ])

        # Apex is above deck (in ship body frame, then rotated)
        apex_body = np.array([0, 0, -self.apex_offset])  # Up in NED body
        apex = deck_pos + R @ apex_body

        # Cone axis points aft and up (approach direction)
        # In ship body: [-1, 0, -0.5] normalized (aft and up)
        axis_body = np.array([-1, 0, -0.5])
        axis_body = axis_body / np.linalg.norm(axis_body)
        axis = R @ axis_body

        return apex, axis

    def is_inside_cone(self, uav_pos: np.ndarray, deck_pos: np.ndarray, deck_att: np.ndarray) -> bool:
        """Check if UAV position is inside approach cone."""
        apex, axis = self.get_cone_params(deck_pos, deck_att)

        # Vector from apex to UAV
        to_uav = uav_pos - apex

        # Distance along cone axis
        dist_along_axis = np.dot(to_uav, axis)

        if dist_along_axis <= 0:
            # Behind apex - only valid if very close to deck
            return np.linalg.norm(uav_pos - deck_pos) < 3.0

        # Perpendicular distance from axis
        dist_perp = np.linalg.norm(to_uav - dist_along_axis * axis)

        # Cone radius at this distance
        cone_radius = dist_along_axis * np.tan(self.half_angle)

        return dist_perp <= cone_radius

    def distance_to_cone_surface(self, uav_pos: np.ndarray, deck_pos: np.ndarray, deck_att: np.ndarray) -> float:
        """
        Signed distance to cone surface.

        Negative = inside cone, Positive = outside cone.
        """
        apex, axis = self.get_cone_params(deck_pos, deck_att)
        to_uav = uav_pos - apex
        dist_along_axis = np.dot(to_uav, axis)

        if dist_along_axis <= 0:
            return np.linalg.norm(uav_pos - deck_pos) - 3.0

        dist_perp = np.linalg.norm(to_uav - dist_along_axis * axis)
        cone_radius = dist_along_axis * np.tan(self.half_angle)

        return dist_perp - cone_radius


class ShipMotionPredictor:
    """
    ARMA-based ship motion predictor.

    Fits on recent past, predicts future deck state.
    """

    def __init__(self, ship_sim: DDGMotionSimulator, config: LandingConfig):
        self.ship_sim = ship_sim
        self.config = config

        # ARMA model (6 features: deck_z, deck_vz, deck_vx, roll, pitch, roll_rate)
        self.arma = ARMAPredictor(ar_order=8, ma_order=4)

        # History buffer
        self.history_t = []
        self.history_data = []
        self.dt = 0.1

        self.last_fit_time = -np.inf

    def update(self, t: float):
        """Update predictor with current ship state."""
        motion = self.ship_sim.get_motion(t)

        # Store observation
        obs = np.array([
            motion['deck_position'][2],   # deck z
            motion['deck_velocity'][2],   # deck vz
            motion['deck_velocity'][0],   # deck vx
            motion['attitude'][0],        # roll
            motion['attitude'][1],        # pitch
            motion['angular_rate'][0],    # roll rate
        ])

        self.history_t.append(t)
        self.history_data.append(obs)

        # Trim old history
        max_hist = int(self.config.arma_history_length / self.dt) + 100
        if len(self.history_t) > max_hist:
            self.history_t = self.history_t[-max_hist:]
            self.history_data = self.history_data[-max_hist:]

        # Refit ARMA periodically
        if t - self.last_fit_time >= self.config.arma_fit_interval:
            self._fit_arma()
            self.last_fit_time = t

    def _fit_arma(self):
        """Fit ARMA on recent history."""
        if len(self.history_data) < 100:
            return

        data = np.array(self.history_data[-int(self.config.arma_history_length / self.dt):])
        self.arma.fit(data, self.dt)

    def predict(self, t_current: float, horizon: float = 5.0, dt: float = 0.5) -> List[DeckPrediction]:
        """
        Predict deck state at future times.

        Args:
            t_current: Current time
            horizon: Prediction horizon (s)
            dt: Prediction time step (s)

        Returns:
            List of DeckPredictions
        """
        if not self.arma.fitted:
            # Fallback: use current state extrapolation
            return self._predict_linear(t_current, horizon, dt)

        predictions = []
        n_steps = int(horizon / dt)

        # Get ARMA predictions
        arma_preds = self.arma.predict(n_steps * int(dt / self.dt))

        # Ship forward velocity (constant)
        ship_vel_x = self.config.ship_speed_kts * 0.5144

        # Current deck position (for x, y extrapolation)
        current_motion = self.ship_sim.get_motion(t_current)
        deck_x0 = current_motion['deck_position'][0]
        deck_y0 = current_motion['deck_position'][1]

        for i in range(n_steps):
            t_pred = t_current + (i + 1) * dt
            arma_idx = min((i + 1) * int(dt / self.dt) - 1, len(arma_preds) - 1)
            pred = arma_preds[arma_idx]

            # Confidence decreases with prediction horizon
            confidence = np.exp(-0.3 * (i + 1) * dt)

            predictions.append(DeckPrediction(
                t=t_pred,
                position=np.array([
                    deck_x0 + ship_vel_x * (i + 1) * dt,
                    deck_y0,  # Assume no lateral drift
                    pred[0]   # Predicted z from ARMA
                ]),
                velocity=np.array([
                    ship_vel_x,
                    0,
                    pred[1]   # Predicted vz
                ]),
                attitude=np.array([
                    pred[3],  # Roll
                    pred[4],  # Pitch
                    0         # Yaw constant
                ]),
                angular_rate=np.array([
                    pred[5] if len(pred) > 5 else 0,  # Roll rate
                    0,
                    0
                ]),
                confidence=confidence
            ))

        return predictions

    def _predict_linear(self, t_current: float, horizon: float, dt: float) -> List[DeckPrediction]:
        """Fallback linear prediction."""
        predictions = []
        motion = self.ship_sim.get_motion(t_current)
        ship_vel_x = self.config.ship_speed_kts * 0.5144

        n_steps = int(horizon / dt)
        for i in range(n_steps):
            t_pred = t_current + (i + 1) * dt
            predictions.append(DeckPrediction(
                t=t_pred,
                position=motion['deck_position'] + motion['deck_velocity'] * (i + 1) * dt,
                velocity=motion['deck_velocity'],
                attitude=motion['attitude'],
                angular_rate=motion['angular_rate'],
                confidence=0.5
            ))
        return predictions


class LandingSimulator:
    """Full simulation with ARMA prediction and optimal trajectory."""

    def __init__(self, config: LandingConfig = None):
        self.config = config if config is not None else LandingConfig()

        # Ship motion
        self.ship_params = DDGParams()
        self.sea_state = SeaState.from_state_number(self.config.sea_state, self.config.wave_direction)
        self.ship_sim = DDGMotionSimulator(self.ship_params, self.sea_state, self.config.ship_speed_kts)

        # Ship predictor
        self.predictor = ShipMotionPredictor(self.ship_sim, self.config)

        # Quadrotor
        self.quad_params = QuadrotorParams()
        self.quad_dynamics = QuadrotorDynamics(self.quad_params)

        # Approach cone
        self.cone = ApproachCone(self.config.cone_half_angle, self.config.cone_apex_offset)

        # Trajectory planner
        self.planner = PseudospectralSolver(N=12, params=self.quad_params)

        # State
        self.t = 0
        self.quad_state = None
        self.current_trajectory = None
        self.target_deck_state = None
        self.landed = False
        self.shutdown_commanded = False

        # History
        self.history = {'t': [], 'quad_pos': [], 'deck_pos': [], 'thrust': [],
                       'in_cone': [], 'pred_error': []}

    def reset(self):
        """Reset simulation."""
        cfg = self.config
        self.t = 0
        self.landed = False
        self.shutdown_commanded = False

        # Warm up predictor with initial ship motion
        for t in np.arange(-cfg.arma_history_length, 0, 0.1):
            self.predictor.update(t)

        # Initial quad state
        deck_motion = self.ship_sim.get_motion(0)
        deck_pos = deck_motion['deck_position']

        quad_pos = np.array([
            deck_pos[0] - cfg.approach_distance,
            deck_pos[1],
            deck_pos[2] - cfg.approach_altitude
        ])

        self.quad_state = QuadrotorState(
            pos=quad_pos,
            vel=np.array([cfg.approach_speed, 0, 0]),
            quat=np.array([1, 0, 0, 0]),
            omega=np.zeros(3)
        )

        for key in self.history:
            self.history[key] = []

        # Initial trajectory
        self._replan()

    def _replan(self):
        """Compute optimal trajectory to predicted deck state."""
        # Get deck predictions
        predictions = self.predictor.predict(self.t, self.config.prediction_horizon, self.config.prediction_dt)

        if not predictions:
            return

        # Find best landing time (highest confidence with feasible approach)
        best_pred = None
        deck_motion = self.ship_sim.get_motion(self.t)

        for pred in predictions:
            # Check if cone constraint would be satisfied
            if self.cone.is_inside_cone(self.quad_state.pos, pred.position, pred.attitude):
                # Check if time is reasonable
                dist = np.linalg.norm(pred.position - self.quad_state.pos)
                min_time = dist / 15.0  # Max 15 m/s approach

                if pred.t - self.t >= min_time:
                    best_pred = pred
                    break

        if best_pred is None:
            best_pred = predictions[-1]  # Use furthest prediction

        self.target_deck_state = best_pred

        # Solve optimal trajectory
        euler = self.quad_state.quat_to_euler()
        x_init = np.concatenate([
            self.quad_state.pos,
            self.quad_state.vel,
            euler,
            self.quad_state.omega
        ])

        deck_state = np.concatenate([
            best_pred.position,
            best_pred.velocity,
            best_pred.attitude
        ])

        constraints = LandingConstraints(
            match_position=True,
            match_velocity=True,
            match_roll=True,
            match_pitch=True,
            zero_thrust=True
        )

        tf_guess = best_pred.t - self.t

        try:
            solution = self.planner.solve(x_init, deck_state, constraints, tf_guess, verbose=False)
            if solution['success']:
                self.current_trajectory = solution
        except:
            pass

    def _get_control(self) -> np.ndarray:
        """Get control using trajectory tracking."""
        if self.target_deck_state is None:
            return np.array([self.quad_params.mass * 9.81, 0, 0, 0])

        deck_motion = self.ship_sim.get_motion(self.t)
        deck_pos = deck_motion['deck_position']
        deck_vel = deck_motion['deck_velocity']
        deck_att = deck_motion['attitude']

        # Height above deck
        height = -(self.quad_state.pos[2] - deck_pos[2])

        # Target: predicted deck position with descent profile
        target_pos = self.target_deck_state.position.copy()
        target_vel = self.target_deck_state.velocity.copy()

        # Blend toward current deck as we get closer
        time_to_land = max(self.target_deck_state.t - self.t, 0.5)
        blend = min(1.0, height / 10.0)  # Blend more to prediction when high

        target_pos = blend * target_pos + (1 - blend) * deck_pos
        target_vel = blend * target_vel + (1 - blend) * deck_vel

        # Add descent rate (limit to safe landing speed)
        descent_rate = min(1.5, max(0.5, height / time_to_land))
        target_vel[2] = descent_rate

        # Position and velocity errors
        pos_error = target_pos - self.quad_state.pos
        vel_error = target_vel - self.quad_state.vel

        # Cone constraint: push toward cone center if outside
        cone_dist = self.cone.distance_to_cone_surface(self.quad_state.pos, deck_pos, deck_att)
        if cone_dist > 0:
            # Outside cone - add correction toward cone axis
            apex, axis = self.cone.get_cone_params(deck_pos, deck_att)
            to_axis = apex - self.quad_state.pos
            to_axis = to_axis - np.dot(to_axis, axis) * axis  # Perpendicular component
            pos_error += 0.5 * to_axis  # Push toward cone

        # PD control (high vertical gains for controlled descent)
        Kp = np.array([1.0, 1.0, 3.0])
        Kd = np.array([3.0, 3.0, 6.0])
        acc_cmd = Kp * pos_error + Kd * vel_error

        acc_cmd = np.clip(acc_cmd, -8, 8)

        # Convert to thrust and attitude
        g = 9.81
        m = self.quad_params.mass

        T_z = m * (g + acc_cmd[2])
        T = np.sqrt(T_z**2 + (m*acc_cmd[0])**2 + (m*acc_cmd[1])**2)
        T = np.clip(T, 0.2 * m * g, 1.5 * m * g)

        # Match deck attitude as we approach
        att_blend = max(0, 1 - height / 5.0)
        roll_target = att_blend * deck_att[0] + (1 - att_blend) * np.arctan2(acc_cmd[1], g)
        pitch_target = att_blend * deck_att[1] + (1 - att_blend) * np.arctan2(-acc_cmd[0], g)

        roll_target = np.clip(roll_target, -0.4, 0.4)
        pitch_target = np.clip(pitch_target, -0.4, 0.4)

        # Attitude control
        Kp_att, Kd_att = 20.0, 4.0
        tau_x = Kp_att * (roll_target - self.quad_state.roll) - Kd_att * self.quad_state.omega[0]
        tau_y = Kp_att * (pitch_target - self.quad_state.pitch) - Kd_att * self.quad_state.omega[1]
        tau_z = -Kd_att * self.quad_state.omega[2]

        # Shutdown when close
        if height < 1.0 and not self.shutdown_commanded:
            self.shutdown_commanded = True
            print(f"SHUTDOWN at t={self.t:.2f}s, height={height:.2f}m")

        if self.shutdown_commanded:
            T *= 0.3  # Reduce thrust rapidly

        return np.array([T, tau_x, tau_y, tau_z])

    def step(self, dt: float = 0.02):
        """Step simulation."""
        if self.landed:
            return

        # Update predictor
        self.predictor.update(self.t)

        deck_motion = self.ship_sim.get_motion(self.t)
        deck_pos = deck_motion['deck_position']

        # Check touchdown window conditions
        height = -(self.quad_state.pos[2] - deck_pos[2])
        deck_att = deck_motion['attitude']
        deck_vel_z = deck_motion['deck_velocity'][2]  # NED: positive = deck moving down

        # Check if in touchdown window
        deck_level = (abs(np.degrees(deck_att[0])) < self.config.max_deck_roll_deg and
                     abs(np.degrees(deck_att[1])) < self.config.max_deck_pitch_deg)
        deck_descending = deck_vel_z > 0 if self.config.deck_moving_down else True

        in_window = deck_level and deck_descending

        # Check touchdown
        if height < self.config.touchdown_altitude:
            self.landed = True
            rel_vel = self.quad_state.vel - deck_motion['deck_velocity']
            print(f"TOUCHDOWN t={self.t:.2f}s")
            print(f"  Height: {height:.2f}m")
            print(f"  Relative velocity: [{rel_vel[0]:.2f}, {rel_vel[1]:.2f}, {rel_vel[2]:.2f}] m/s")
            print(f"  Descent rate: {rel_vel[2]:.2f} m/s (target < {self.config.max_touchdown_velocity:.1f})")
            print(f"  Deck roll: {np.degrees(deck_att[0]):.1f}°, pitch: {np.degrees(deck_att[1]):.1f}°")
            print(f"  Deck heave rate: {deck_vel_z:.2f} m/s")
            print(f"  In window: {in_window} (level={deck_level}, descending={deck_descending})")
            self.final_rel_vel = rel_vel
            return

        # Get and apply control
        u = self._get_control()
        self._step_quad(u, dt)

        # Record history
        self.history['t'].append(self.t)
        self.history['quad_pos'].append(self.quad_state.pos.copy())
        self.history['deck_pos'].append(deck_pos.copy())
        self.history['thrust'].append(u[0])
        self.history['in_cone'].append(self.cone.is_inside_cone(
            self.quad_state.pos, deck_pos, deck_motion['attitude']))

        # Prediction error (if we have a prediction)
        if self.target_deck_state and abs(self.target_deck_state.t - self.t) < 0.1:
            pred_error = np.linalg.norm(self.target_deck_state.position - deck_pos)
            self.history['pred_error'].append(pred_error)

        self.t += dt

        # Replan periodically
        if int(self.t / self.config.replan_interval) > int((self.t - dt) / self.config.replan_interval):
            self._replan()

    def _step_quad(self, u: np.ndarray, dt: float):
        """Step quadrotor dynamics."""
        T, tau_x, tau_y, tau_z = u
        p = self.quad_params
        L = p.arm_length
        s45 = np.sin(np.pi/4)

        T_per = T / 4
        T1 = T_per + tau_x/(4*L*s45) + tau_y/(4*L*s45)
        T2 = T_per + tau_x/(4*L*s45) - tau_y/(4*L*s45)
        T3 = T_per - tau_x/(4*L*s45) - tau_y/(4*L*s45)
        T4 = T_per - tau_x/(4*L*s45) + tau_y/(4*L*s45)

        motor_thrusts = np.clip([T1, T2, T3, T4], 0, p.max_thrust_per_rotor)
        self.quad_state = self.quad_dynamics.step(self.quad_state, motor_thrusts, dt)

    def run(self, max_time: float = 30.0, dt: float = 0.02) -> dict:
        """Run simulation."""
        self.reset()

        while self.t < max_time and not self.landed:
            self.step(dt)

        h = self.history
        return {
            'success': self.landed,
            'landing_time': self.t if self.landed else np.nan,
            'in_cone_pct': 100 * sum(h['in_cone']) / max(len(h['in_cone']), 1),
            'avg_pred_error': np.mean(h['pred_error']) if h['pred_error'] else np.nan,
            'history': h
        }


def demo():
    """Demo landing simulation."""
    print("Shipboard Landing Simulation")
    print("=" * 50)

    config = LandingConfig(
        sea_state=4,
        ship_speed_kts=15,
        approach_altitude=25,
        approach_distance=50,
        approach_speed=5.0,  # Slower approach
        cone_half_angle=35
    )

    print(f"Sea state: {config.sea_state}, Ship: {config.ship_speed_kts} kts")
    print(f"Approach: {config.approach_distance}m behind, {config.approach_altitude}m above")
    print(f"Cone: {config.cone_half_angle}° half-angle")
    print()

    sim = LandingSimulator(config)
    results = sim.run(max_time=25.0)

    print(f"\nResults:")
    print(f"  Success: {results['success']}")
    print(f"  Landing time: {results['landing_time']:.2f} s")
    print(f"  In cone: {results['in_cone_pct']:.1f}%")
    print(f"  Avg prediction error: {results['avg_pred_error']:.3f} m")

    if results['history']['t']:
        h = results['history']
        print(f"\n  Max thrust: {max(h['thrust']):.1f} N")
        print(f"  Min thrust: {min(h['thrust']):.1f} N")

    return results


if __name__ == "__main__":
    demo()
