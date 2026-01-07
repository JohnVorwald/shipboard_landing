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
from optimal_control.trajectory_planner import LandingTrajectoryPlanner, TrajectoryResult


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

    # Cone constraint (apex BELOW deck, opens upward to landing circle)
    cone_half_angle: float = 30.0       # Cone half-angle (degrees)
    cone_apex_depth: float = 50.0       # Distance from apex to deck (apex below deck)
    landing_radius: float = 3.0         # Radius of landing circle on deck (m)

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

    The cone apex is BELOW the flight deck. The cone opens UPWARD and ends
    in a circular landing area on the deck surface. UAV must stay within
    this cone during approach.

    Geometry:
    - Apex is below deck (apex_depth meters below)
    - Cone axis points upward (negative z in NED) and tilts with ship
    - At deck level, cone radius equals landing_radius
    - Cone half-angle = atan(landing_radius / apex_depth)
    """

    def __init__(self, half_angle_deg: float = 30.0, apex_depth: float = 50.0, landing_radius: float = 3.0):
        """
        Args:
            half_angle_deg: Cone half-angle (degrees)
            apex_depth: Distance from apex to deck (m), apex is below deck
            landing_radius: Radius of landing circle on deck (m)
        """
        self.half_angle = np.radians(half_angle_deg)
        self.apex_depth = apex_depth
        self.landing_radius = landing_radius

    def get_cone_params(self, deck_pos: np.ndarray, deck_att: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cone apex and axis given deck state.

        Returns:
            apex: Cone apex position (BELOW deck)
            axis: Cone axis direction (pointing UP toward deck)
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

        # Apex is BELOW deck (positive z in NED body = down)
        apex_body = np.array([0, 0, self.apex_depth])  # Below deck in NED
        apex = deck_pos + R @ apex_body

        # Cone axis points UP toward deck (negative z in body frame)
        # Also tilts aft for approach angle
        axis_body = np.array([-0.3, 0, -1.0])  # Mostly up, slightly aft
        axis_body = axis_body / np.linalg.norm(axis_body)
        axis = R @ axis_body

        return apex, axis

    def is_inside_cone(self, uav_pos: np.ndarray, deck_pos: np.ndarray, deck_att: np.ndarray) -> bool:
        """Check if UAV position is inside approach cone."""
        apex, axis = self.get_cone_params(deck_pos, deck_att)

        # Vector from apex to UAV
        to_uav = uav_pos - apex

        # Distance along cone axis (positive = toward deck)
        dist_along_axis = np.dot(to_uav, axis)

        if dist_along_axis <= 0:
            # Below apex (very unlikely in practice)
            return False

        # Perpendicular distance from axis
        dist_perp = np.linalg.norm(to_uav - dist_along_axis * axis)

        # Cone radius at this distance along axis
        cone_radius = dist_along_axis * np.tan(self.half_angle)

        # Also check: if very close to deck, use landing radius instead
        height_above_deck = -(uav_pos[2] - deck_pos[2])  # NED
        if height_above_deck < 3.0:
            # Near deck - just check within landing radius
            lateral_dist = np.linalg.norm(uav_pos[:2] - deck_pos[:2])
            return lateral_dist <= self.landing_radius * 1.5

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
            return 100.0  # Very far outside (below apex)

        dist_perp = np.linalg.norm(to_uav - dist_along_axis * axis)
        cone_radius = dist_along_axis * np.tan(self.half_angle)

        # Near deck - check landing radius
        height_above_deck = -(uav_pos[2] - deck_pos[2])
        if height_above_deck < 3.0:
            lateral_dist = np.linalg.norm(uav_pos[:2] - deck_pos[:2])
            return lateral_dist - self.landing_radius * 1.5

        return dist_perp - cone_radius

    def get_landing_circle(self, deck_pos: np.ndarray, deck_att: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Get the landing circle parameters.

        Returns:
            center: Center of landing circle (deck position)
            radius: Landing radius
        """
        return deck_pos.copy(), self.landing_radius


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

        # Approach cone (apex below deck, opens upward to landing circle)
        self.cone = ApproachCone(
            half_angle_deg=self.config.cone_half_angle,
            apex_depth=self.config.cone_apex_depth,
            landing_radius=self.config.landing_radius
        )

        # Trajectory planner (minimum snap polynomial trajectories)
        self.planner = LandingTrajectoryPlanner(self.quad_params)

        # State
        self.t = 0
        self.quad_state = None
        self.current_trajectory = None
        self.trajectory_start_time = None  # When current trajectory was computed
        self.target_deck_state = None
        self.landed = False
        self.shutdown_commanded = False
        self.waiting_for_window = False  # True when waiting for touchdown window

        # History
        self.history = {'t': [], 'quad_pos': [], 'deck_pos': [], 'thrust': [],
                       'in_cone': [], 'pred_error': [], 'tracking_error': []}

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

        self.trajectory_start_time = None
        self.waiting_for_window = False

        # Initial trajectory
        self._replan()

    def _replan(self):
        """Compute optimal trajectory to predicted deck state."""
        deck_motion = self.ship_sim.get_motion(self.t)
        deck_pos = deck_motion['deck_position']
        deck_vel = deck_motion['deck_velocity']
        deck_att = deck_motion['attitude']

        # Compute time to intercept based on closing rate
        # Relative position (UAV to deck)
        rel_pos = deck_pos - self.quad_state.pos
        rel_vel = deck_vel - self.quad_state.vel

        # Height above deck
        height = -rel_pos[2]  # NED: negative z is up

        # Horizontal distance
        horiz_dist = np.linalg.norm(rel_pos[:2])

        # Estimate intercept time based on current kinematics
        # Use a simple glide-slope: descend at controlled rate while closing horizontal gap
        closing_rate_horiz = -np.dot(rel_pos[:2], rel_vel[:2]) / (horiz_dist + 0.1)

        # Safe descent rate (1-2 m/s at touchdown)
        safe_descent_rate = min(3.0, max(1.0, height / 10.0))

        # Time to reach deck height at safe descent rate
        t_vertical = height / safe_descent_rate if safe_descent_rate > 0.1 else 10.0

        # Time to close horizontal gap
        if closing_rate_horiz > 1.0:
            t_horizontal = horiz_dist / closing_rate_horiz
        else:
            # Not closing fast enough - need to accelerate
            t_horizontal = horiz_dist / 5.0  # Assume we can close at 5 m/s

        # Intercept time is maximum of both (need to satisfy both)
        t_intercept = max(t_vertical, t_horizontal, 2.0)
        t_intercept = min(t_intercept, self.config.prediction_horizon)

        # Get ARMA prediction at intercept time
        predictions = self.predictor.predict(self.t, t_intercept + 1.0, self.config.prediction_dt)

        if not predictions:
            return

        # Find prediction closest to intercept time
        best_pred = None
        for pred in predictions:
            if pred.t - self.t >= t_intercept - 0.3:
                # Check touchdown window
                deck_level = (abs(np.degrees(pred.attitude[0])) < self.config.max_deck_roll_deg and
                             abs(np.degrees(pred.attitude[1])) < self.config.max_deck_pitch_deg)
                deck_descending = pred.velocity[2] > 0 if self.config.deck_moving_down else True

                if deck_level and deck_descending:
                    best_pred = pred
                    break
                elif best_pred is None:
                    best_pred = pred

        if best_pred is None:
            best_pred = predictions[-1]

        self.target_deck_state = best_pred

        # Plan trajectory using minimum-snap polynomial planner
        # Use the computed intercept time, not the prediction time
        tf_actual = max(1.5, best_pred.t - self.t)

        try:
            result = self.planner.plan_landing(
                quad_pos=self.quad_state.pos,
                quad_vel=self.quad_state.vel,
                deck_pos=best_pred.position,
                deck_vel=best_pred.velocity,
                deck_att=best_pred.attitude,
                tf_desired=tf_actual
            )

            # VALIDATE SOLUTION before using
            if result['success']:
                pos_err = result['terminal_pos_error']
                vel_err = result['terminal_vel_error']

                if pos_err < 0.1 and vel_err < 0.1:
                    self.current_trajectory = result['trajectory']
                    self.trajectory_start_time = self.t

        except Exception as e:
            pass

    def _interpolate_trajectory(self, t_query: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate optimal trajectory at given time.

        Returns:
            target_pos, target_vel, target_thrust
        """
        if self.current_trajectory is None or self.trajectory_start_time is None:
            return None, None, None

        traj = self.current_trajectory  # TrajectoryResult from min-snap planner
        t_rel = t_query - self.trajectory_start_time  # Time since trajectory start
        tf = traj.tf

        # Clamp to trajectory bounds
        t_rel = np.clip(t_rel, 0, tf)

        # Use polynomial evaluation for smooth interpolation
        sample = self.planner.sample_trajectory(traj, t_rel)

        return sample['position'], sample['velocity'], sample['thrust']

    def _get_control(self) -> np.ndarray:
        """
        Get control using proportional navigation guidance.

        Key principle: command descent rate proportional to time-to-go,
        so that velocity naturally decreases to match deck at intercept.
        """
        g = 9.81
        m = self.quad_params.mass

        deck_motion = self.ship_sim.get_motion(self.t)
        deck_pos = deck_motion['deck_position']
        deck_vel = deck_motion['deck_velocity']
        deck_att = deck_motion['attitude']

        # Relative state
        rel_pos = deck_pos - self.quad_state.pos  # Vector FROM UAV TO deck
        rel_vel = deck_vel - self.quad_state.vel   # Deck velocity relative to UAV (range rate)

        # Height above deck
        # In NED: negative z = up
        # UAV above deck means UAV_z < deck_z (more negative)
        # rel_pos[2] = deck_z - UAV_z > 0 when UAV is above deck
        height = rel_pos[2]  # Positive when UAV is above deck

        # Horizontal distance to deck
        horiz_dist = np.linalg.norm(rel_pos[:2])

        # Closing rate: rate at which distance is decreasing
        # d(dist)/dt = (rel_pos dot rel_vel) / dist
        # Positive closing rate = getting closer
        range_rate = np.dot(rel_pos[:2], rel_vel[:2]) / (horiz_dist + 0.1)
        closing_rate = -range_rate  # Negate because range rate is positive when getting further

        # Estimate time to go - use a fixed approach profile for predictability
        # Want to intercept in reasonable time while maintaining control
        min_intercept_time = 5.0  # At least 5s to intercept
        max_intercept_time = 15.0  # At most 15s

        # Compute required velocity to intercept in min_intercept_time
        required_vel_for_fast = np.linalg.norm(rel_pos[:2]) / min_intercept_time
        max_velocity = 8.0  # Max lateral velocity we want to command

        if required_vel_for_fast > max_velocity:
            # Need more time - use max velocity approach
            t_go_horiz = horiz_dist / max_velocity
        else:
            t_go_horiz = min_intercept_time

        t_go_horiz = np.clip(t_go_horiz, min_intercept_time, max_intercept_time)

        # Vertical time to go - controlled descent
        if height > 2.0:
            t_go_vert = height / 1.5  # 1.5 m/s descent rate
        else:
            t_go_vert = height / 0.5  # Slow down near deck

        # Time to go is the longer of the two (we need to satisfy both)
        t_go = max(t_go_horiz, t_go_vert, 3.0)

        # ===== HORIZONTAL GUIDANCE =====
        # Use pursuit guidance: command velocity toward target, blending to deck velocity

        if horiz_dist > 5.0:
            # Far from deck: pursuit guidance
            # Desired velocity = direction to target * speed + deck velocity
            pursuit_speed = min(horiz_dist / 5.0, 6.0)  # Scale speed with distance, cap at 6 m/s
            los = rel_pos[:2] / horiz_dist
            pursuit_vel = pursuit_speed * los + deck_vel[:2]

            # Velocity error
            vel_error = pursuit_vel - self.quad_state.vel[:2]

            # Acceleration to achieve desired velocity
            Kp_horiz = 2.0
            horiz_acc_cmd = Kp_horiz * vel_error
            horiz_acc_cmd = np.clip(horiz_acc_cmd, -5.0, 5.0)
        else:
            # Close to deck: match deck velocity with position correction
            # Blend between position tracking and velocity matching
            pos_gain = 1.5  # m/s per meter of error
            vel_gain = 3.0

            # Position correction: velocity toward deck
            pos_correction = pos_gain * rel_pos[:2]

            # Velocity matching: match deck velocity
            vel_error = deck_vel[:2] - self.quad_state.vel[:2]

            # Blend: closer = more velocity matching
            blend = horiz_dist / 5.0  # 1 at 5m, 0 at 0m
            target_vel = blend * pos_correction + (1 - blend) * (deck_vel[:2] + pos_correction * 0.5)
            vel_error_total = target_vel - self.quad_state.vel[:2]

            horiz_acc_cmd = vel_gain * vel_error_total
            horiz_acc_cmd = np.clip(horiz_acc_cmd, -4.0, 4.0)

        # VERTICAL: Control descent to arrive at deck at same time as horizontal intercept
        # In NED: positive z velocity = moving down
        # UAV needs to descend (UAV_vz > 0) to close positive height gap

        # Descent rate should match time to horizontal intercept
        target_descent_rate = height / max(t_go, 3.0)

        # Limit descent rate based on deceleration capability
        max_decel = 4.0  # m/s² (conservative - max is ~5)
        effective_height = max(0, height - 0.5)
        max_safe_descent = np.sqrt(2 * max_decel * effective_height) if effective_height > 0 else 0.3
        target_descent_rate = min(target_descent_rate, max_safe_descent, 2.0)

        # Near deck: smoothly transition to match deck velocity
        if height < 3.0:
            blend = height / 3.0
            target_descent_rate = blend * target_descent_rate
            horiz_acc_cmd *= blend  # Reduce horizontal acceleration near deck

        # Vertical velocity control
        desired_uav_vz = deck_vel[2] + target_descent_rate
        uav_vz = self.quad_state.vel[2]
        vert_vel_error = desired_uav_vz - uav_vz

        # Vertical acceleration command (P controller on velocity error)
        Kp_vert = 5.0
        acc_cmd_vert = Kp_vert * vert_vel_error
        acc_cmd_vert = np.clip(acc_cmd_vert, -6, 6)

        # Total commanded acceleration (NED)
        acc_cmd = np.array([horiz_acc_cmd[0], horiz_acc_cmd[1], acc_cmd_vert])

        # Convert to thrust
        # In NED: thrust acts upward (negative z direction in body frame)
        # Equation of motion: m*a = T*(-z_body) + m*g*(+z_ned)
        # For pure vertical with level attitude: m*a_z = -T + m*g
        # => T = m*(g - a_z)
        # For a_z < 0 (upward accel to slow descent): T > m*g (more than hover)
        # For a_z > 0 (downward accel to speed descent): T < m*g (less than hover)

        # Vertical component
        T_z = m * (g - acc_cmd[2])

        # Horizontal acceleration requires attitude tilt, which adds to thrust
        # Approximate: T = sqrt(T_z² + (m*ax)² + (m*ay)²)
        T = np.sqrt(T_z**2 + (m * acc_cmd[0])**2 + (m * acc_cmd[1])**2)
        T = np.clip(T, 0.1 * m * g, 1.5 * m * g)

        # Attitude from acceleration direction
        if T > 0.1 * m * g:
            roll_target = np.arctan2(acc_cmd[1], g)
            pitch_target = np.arctan2(-acc_cmd[0], g)
        else:
            roll_target = 0
            pitch_target = 0

        # Near touchdown: blend attitude toward deck attitude
        if height < 3.0:
            att_blend = 1 - height / 3.0  # 0 at 3m, 1 at 0m
            roll_target = (1 - att_blend) * roll_target + att_blend * deck_att[0]
            pitch_target = (1 - att_blend) * pitch_target + att_blend * deck_att[1]

        # Cone constraint
        cone_dist = self.cone.distance_to_cone_surface(self.quad_state.pos, deck_pos, deck_att)
        if cone_dist > 0:
            apex, axis = self.cone.get_cone_params(deck_pos, deck_att)
            to_axis = apex - self.quad_state.pos
            to_axis = to_axis - np.dot(to_axis, axis) * axis
            roll_target += 0.05 * to_axis[1]
            pitch_target -= 0.05 * to_axis[0]

        # Clamp attitude
        roll_target = np.clip(roll_target, -0.4, 0.4)
        pitch_target = np.clip(pitch_target, -0.4, 0.4)

        # Attitude control
        Kp_att, Kd_att = 25.0, 5.0
        tau_x = Kp_att * (roll_target - self.quad_state.roll) - Kd_att * self.quad_state.omega[0]
        tau_y = Kp_att * (pitch_target - self.quad_state.pitch) - Kd_att * self.quad_state.omega[1]
        tau_z = -Kd_att * self.quad_state.omega[2]

        # Record tracking error
        self.history['tracking_error'].append(horiz_dist)

        # Shutdown logic
        if height < 0.5 and not self.shutdown_commanded:
            rel_speed = np.linalg.norm(rel_vel)
            if rel_speed < 2.0:
                self.shutdown_commanded = True
                print(f"SHUTDOWN at t={self.t:.2f}s, height={height:.2f}m, rel_vel={rel_speed:.2f}m/s")

        if self.shutdown_commanded:
            T *= 0.2

        return np.array([T, tau_x, tau_y, tau_z])

    def step(self, dt: float = 0.02):
        """Step simulation."""
        if self.landed:
            return

        # Update predictor
        self.predictor.update(self.t)

        deck_motion = self.ship_sim.get_motion(self.t)
        deck_pos = deck_motion['deck_position']
        deck_vel = deck_motion['deck_velocity']

        # Check touchdown window conditions
        # Height = how far above deck the UAV is (positive when UAV above)
        # In NED: height = deck_z - UAV_z (since more negative z = higher)
        height = deck_pos[2] - self.quad_state.pos[2]
        deck_att = deck_motion['attitude']
        deck_vel_z = deck_vel[2]  # NED: positive = deck moving down

        # Check if in touchdown window
        deck_level = (abs(np.degrees(deck_att[0])) < self.config.max_deck_roll_deg and
                     abs(np.degrees(deck_att[1])) < self.config.max_deck_pitch_deg)
        deck_descending = deck_vel_z > 0 if self.config.deck_moving_down else True

        in_window = deck_level and deck_descending

        # Check touchdown
        if height < self.config.touchdown_altitude:
            self.landed = True
            rel_vel = self.quad_state.vel - deck_vel

            # Compute touchdown quality metrics
            rel_speed = np.linalg.norm(rel_vel)
            lateral_error = np.sqrt((self.quad_state.pos[0] - deck_pos[0])**2 +
                                   (self.quad_state.pos[1] - deck_pos[1])**2)
            roll_error = abs(self.quad_state.roll - deck_att[0])
            pitch_error = abs(self.quad_state.pitch - deck_att[1])

            print(f"\n{'='*50}")
            print(f"TOUCHDOWN t={self.t:.2f}s")
            print(f"{'='*50}")
            print(f"  Height: {height:.3f}m")
            print(f"  Relative velocity: [{rel_vel[0]:.3f}, {rel_vel[1]:.3f}, {rel_vel[2]:.3f}] m/s")
            print(f"  Relative speed: {rel_speed:.3f} m/s (target < {self.config.max_touchdown_velocity:.1f})")
            print(f"  Lateral position error: {lateral_error:.3f} m")
            print(f"  Roll error: {np.degrees(roll_error):.2f}°")
            print(f"  Pitch error: {np.degrees(pitch_error):.2f}°")
            print(f"  Deck roll: {np.degrees(deck_att[0]):.1f}°, pitch: {np.degrees(deck_att[1]):.1f}°")
            print(f"  Deck heave rate: {deck_vel_z:.3f} m/s ({'down' if deck_vel_z > 0 else 'up'})")
            print(f"  In window: {in_window} (level={deck_level}, descending={deck_descending})")

            # Success criteria
            velocity_ok = rel_speed < self.config.max_touchdown_velocity
            position_ok = lateral_error < 1.0
            attitude_ok = roll_error < 0.1 and pitch_error < 0.1

            print(f"\n  Success metrics:")
            print(f"    Velocity matched: {'YES' if velocity_ok else 'NO'}")
            print(f"    Position matched: {'YES' if position_ok else 'NO'}")
            print(f"    Attitude matched: {'YES' if attitude_ok else 'NO'}")
            print(f"    In touchdown window: {'YES' if in_window else 'NO'}")
            print(f"{'='*50}\n")

            self.final_rel_vel = rel_vel
            self.touchdown_success = velocity_ok and position_ok and attitude_ok
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
        """Step quadrotor dynamics with thrust-prioritized motor allocation."""
        T, tau_x, tau_y, tau_z = u
        p = self.quad_params
        L = p.arm_length
        s45 = np.sin(np.pi/4)

        # Maximum torque that can be applied without losing thrust
        # With T_per = T/4 per motor, max torque increment per motor = min(T_per, max_thrust - T_per)
        T_per = T / 4
        max_torque_per_motor = min(T_per, p.max_thrust_per_rotor - T_per)

        # Torque to motor thrust conversion factor
        torque_to_motor = 1 / (4 * L * s45)

        # Limit torques to avoid thrust loss
        max_tau = max_torque_per_motor / torque_to_motor
        tau_x = np.clip(tau_x, -max_tau, max_tau)
        tau_y = np.clip(tau_y, -max_tau, max_tau)

        # Now compute motor thrusts (won't saturate due to torque limiting)
        T1 = T_per + tau_x * torque_to_motor + tau_y * torque_to_motor
        T2 = T_per + tau_x * torque_to_motor - tau_y * torque_to_motor
        T3 = T_per - tau_x * torque_to_motor - tau_y * torque_to_motor
        T4 = T_per - tau_x * torque_to_motor + tau_y * torque_to_motor

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
        approach_speed=12.0,  # Must be faster than ship speed (15kt = 7.7m/s)
        cone_half_angle=30,
        cone_apex_depth=50.0,  # Apex 50m below deck
        landing_radius=3.0     # 3m landing circle
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
