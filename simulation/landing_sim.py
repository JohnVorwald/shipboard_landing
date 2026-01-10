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
from optimal_control.pmp_controller import PMPController, PMPTrajectory, create_pmp_trajectory, ControllerGains
from optimal_control.pseudospectral import PseudospectralSolver, LandingConstraints


@dataclass
class LandingConfig:
    """Configuration for landing simulation."""
    # Ship parameters
    sea_state: int = 4
    ship_speed_kts: float = 15.0
    wave_direction: float = 45.0

    # Approach parameters
    approach_altitude: float = 25.0
    approach_distance: float = 40.0
    approach_speed: float = 6.0          # Slower approach for prediction accuracy

    # ARMA prediction
    arma_fit_interval: float = 5.0      # Refit ARMA every 5s
    arma_history_length: float = 30.0    # Use 30s of past data
    prediction_dt: float = 0.5           # Predict at 0.5s increments
    prediction_horizon: float = 10.0     # Predict up to 10s ahead for longer approaches

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
    """Full simulation with ARMA prediction and PMP-based optimal control."""

    def __init__(self, config: LandingConfig = None, use_pmp: bool = True):
        self.config = config if config is not None else LandingConfig()
        self.use_pmp = use_pmp

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

        # PMP Controller with costate feedback
        self.pmp_controller = PMPController(self.quad_params)

        # Pseudospectral solver for optimal trajectories (N=10 for faster solve)
        self.ps_solver = PseudospectralSolver(N=10, params=self.quad_params)

        # Track solver performance
        self.solver_success_count = 0
        self.solver_fail_count = 0

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
                       'in_cone': [], 'pred_error': [], 'tracking_error': [],
                       'costate_norm': []}

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
        """Compute optimal trajectory using PMP/pseudospectral solver with ARMA predictions."""
        deck_motion = self.ship_sim.get_motion(self.t)
        deck_pos = deck_motion['deck_position']
        deck_vel = deck_motion['deck_velocity']
        deck_att = deck_motion['attitude']

        # Compute time to intercept based on closing rate
        # Relative position (UAV to deck)
        rel_pos = deck_pos - self.quad_state.pos
        rel_vel = deck_vel - self.quad_state.vel

        # Height above deck (positive when UAV above deck in NED)
        height = rel_pos[2]

        # Horizontal distance
        horiz_dist = np.linalg.norm(rel_pos[:2])

        # Estimate intercept time based on current kinematics
        closing_rate_horiz = -np.dot(rel_pos[:2], rel_vel[:2]) / (horiz_dist + 0.1)

        # Safe descent rate
        safe_descent_rate = min(3.0, max(1.0, abs(height) / 10.0))

        # Time to reach deck height
        t_vertical = abs(height) / safe_descent_rate if safe_descent_rate > 0.1 else 10.0

        # Time to close horizontal gap
        if closing_rate_horiz > 1.0:
            t_horizontal = horiz_dist / closing_rate_horiz
        else:
            t_horizontal = horiz_dist / 5.0

        # Intercept time estimate - ensure enough time for smooth approach
        # Calculate based on 3D distance and comfortable approach speed
        dist_3d = np.sqrt(horiz_dist**2 + height**2)

        # Use consistent approach speed for intercept calculation
        approach_speed = 6.0  # m/s net closing rate
        t_direct = dist_3d / approach_speed

        # Clamp intercept time to reasonable range
        t_intercept = np.clip(t_direct, 3.0, 8.0)

        # Get ARMA predictions covering the intercept window
        predictions = self.predictor.predict(self.t, t_intercept + 1.0, self.config.prediction_dt)

        if not predictions:
            return

        # Find prediction closest to intercept time with good touchdown conditions
        best_pred = None
        best_time_diff = float('inf')

        for pred in predictions:
            time_to_pred = pred.t - self.t

            # Check touchdown window conditions
            deck_level = (abs(np.degrees(pred.attitude[0])) < self.config.max_deck_roll_deg and
                         abs(np.degrees(pred.attitude[1])) < self.config.max_deck_pitch_deg)
            deck_descending = pred.velocity[2] > 0 if self.config.deck_moving_down else True

            # Prefer predictions close to intercept time
            time_diff = abs(time_to_pred - t_intercept)

            if time_diff < best_time_diff:
                # Accept if in good window, or if no good window found yet
                if (deck_level and deck_descending) or best_pred is None:
                    best_pred = pred
                    best_time_diff = time_diff

                    # If this is a good window close to intercept, use it
                    if deck_level and deck_descending and time_diff < 0.5:
                        break

        if best_pred is None:
            best_pred = predictions[min(int(t_intercept / self.config.prediction_dt), len(predictions)-1)]

        # Ensure minimum trajectory time for smooth approach
        time_to_pred = best_pred.t - self.t
        tf_actual = max(3.0, time_to_pred)

        # If we extended the time, find a prediction at that time instead
        if tf_actual > time_to_pred + 0.3:
            # Find prediction closest to tf_actual from now
            target_time = self.t + tf_actual
            for pred in predictions:
                if abs(pred.t - target_time) < abs(best_pred.t - target_time):
                    best_pred = pred

        self.target_deck_state = best_pred
        # Recalculate tf_actual to match the prediction we're actually using
        tf_actual = max(2.5, best_pred.t - self.t)

        if self.use_pmp:
            # Use min-snap for speed, but create PMPTrajectory with costates
            self._replan_pmp_minsnap(best_pred, tf_actual)
        else:
            # Fallback to min-snap planner without PMP controller
            self._replan_minsnap(best_pred, tf_actual)

    def _replan_pmp_minsnap(self, target: DeckPrediction, tf: float):
        """
        Fast trajectory planning using min-snap with PMP costate estimation.

        Uses polynomial trajectory for speed, but estimates costates
        to enable optimal feedback corrections in the controller.
        """
        try:
            result = self.planner.plan_landing(
                quad_pos=self.quad_state.pos,
                quad_vel=self.quad_state.vel,
                deck_pos=target.position,
                deck_vel=target.velocity,
                deck_att=target.attitude,
                tf_desired=tf
            )

            if result['success']:
                traj = result['trajectory']

                # Sample trajectory to create state/control arrays
                N = 20
                t_traj = np.linspace(0, traj.tf, N)
                x_traj = np.zeros((N, 12))
                u_traj = np.zeros((N, 4))

                euler = self.quad_state.quat_to_euler()

                for i, t in enumerate(t_traj):
                    sample = self.planner.sample_trajectory(traj, t)
                    # State: pos, vel, att, omega
                    alpha = t / traj.tf
                    x_traj[i, 0:3] = sample['position']
                    x_traj[i, 3:6] = sample['velocity']
                    # Interpolate attitude from current to target
                    x_traj[i, 6:9] = (1 - alpha) * euler + alpha * target.attitude
                    # Angular rates approximately zero for smooth trajectory
                    x_traj[i, 9:12] = 0

                    # Control: thrust + zero torques (approximate)
                    u_traj[i, 0] = sample['thrust']

                # Create PMPTrajectory with costate estimation
                pmp_traj = create_pmp_trajectory(
                    x_traj=x_traj,
                    u_traj=u_traj,
                    t_traj=t_traj,
                    tf=traj.tf,
                    deck_pos=target.position,
                    deck_vel=target.velocity,
                    deck_att=target.attitude,
                    params=self.quad_params
                )

                self.pmp_controller.set_trajectory(pmp_traj, self.t)
                self.current_trajectory = pmp_traj
                self.trajectory_start_time = self.t
                self.solver_success_count += 1
            else:
                self.solver_fail_count += 1

        except Exception as e:
            self.solver_fail_count += 1

    def _replan_pseudospectral(self, target: DeckPrediction, tf: float):
        """
        Solve optimal trajectory using pseudospectral method.

        Creates PMPTrajectory with estimated costates for feedback control.
        """
        try:
            # Current state
            euler = self.quad_state.quat_to_euler()
            x_init = np.concatenate([
                self.quad_state.pos,
                self.quad_state.vel,
                euler,
                self.quad_state.omega
            ])

            # Target deck state
            deck_state = np.concatenate([
                target.position,
                target.velocity,
                target.attitude
            ])

            # Solve using pseudospectral
            constraints = LandingConstraints(
                match_position=True,
                match_velocity=True,
                match_roll=True,
                match_pitch=True,
                zero_thrust=True
            )

            solution = self.ps_solver.solve(
                x_init=x_init,
                deck_state=deck_state,
                constraints=constraints,
                tf_guess=tf,
                verbose=False
            )

            if solution['success'] or solution['constraint_violation'] < 1.0:
                # Create PMPTrajectory from solution
                pmp_traj = create_pmp_trajectory(
                    x_traj=solution['X'],
                    u_traj=solution['U'],
                    t_traj=solution['t'],
                    tf=solution['tf'],
                    deck_pos=target.position,
                    deck_vel=target.velocity,
                    deck_att=target.attitude,
                    params=self.quad_params
                )

                # Set trajectory in controller
                self.pmp_controller.set_trajectory(pmp_traj, self.t)
                self.current_trajectory = pmp_traj
                self.trajectory_start_time = self.t

        except Exception as e:
            # Fallback to min-snap
            self._replan_minsnap(target, tf)

    def _replan_minsnap(self, target: DeckPrediction, tf: float):
        """Fallback trajectory planning using minimum-snap polynomials."""
        try:
            result = self.planner.plan_landing(
                quad_pos=self.quad_state.pos,
                quad_vel=self.quad_state.vel,
                deck_pos=target.position,
                deck_vel=target.velocity,
                deck_att=target.attitude,
                tf_desired=tf
            )

            if result['success']:
                pos_err = result['terminal_pos_error']
                vel_err = result['terminal_vel_error']

                if pos_err < 0.1 and vel_err < 0.1:
                    self.current_trajectory = result['trajectory']
                    self.trajectory_start_time = self.t
        except Exception:
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
        Get control using PMP-based trajectory tracking or fallback guidance.

        When use_pmp=True:
        - Uses costate feedback for near-optimal control
        - Tracks pseudospectral optimal trajectory

        Fallback (use_pmp=False):
        - Uses proportional navigation guidance
        """
        deck_motion = self.ship_sim.get_motion(self.t)
        deck_pos = deck_motion['deck_position']
        deck_vel = deck_motion['deck_velocity']
        deck_att = deck_motion['attitude']

        # Relative state for metrics
        rel_pos = deck_pos - self.quad_state.pos
        rel_vel = deck_vel - self.quad_state.vel
        height = rel_pos[2]
        horiz_dist = np.linalg.norm(rel_pos[:2])

        # Record tracking error
        self.history['tracking_error'].append(horiz_dist)

        if self.use_pmp and self.pmp_controller.trajectory_valid(self.t):
            # Use PMP controller with costate feedback
            x_current = np.concatenate([
                self.quad_state.pos,
                self.quad_state.vel,
                self.quad_state.quat_to_euler(),
                self.quad_state.omega
            ])

            u = self.pmp_controller.compute_control(
                self.t, x_current, deck_pos, deck_vel, deck_att
            )

            # Record costate norm for debugging
            if self.pmp_controller.trajectory is not None:
                _, _, lam = self.pmp_controller._interpolate_trajectory(self.t)
                if lam is not None:
                    self.history['costate_norm'].append(np.linalg.norm(lam))

            # Apply cone constraint correction
            u = self._apply_cone_constraint(u, deck_pos, deck_att)

            # Shutdown logic
            u = self._apply_shutdown_logic(u, height, rel_vel)

            return u
        else:
            # Fallback to proportional navigation
            return self._get_control_fallback(deck_pos, deck_vel, deck_att,
                                               rel_pos, rel_vel, height, horiz_dist)

    def _apply_cone_constraint(self, u: np.ndarray, deck_pos: np.ndarray,
                                deck_att: np.ndarray) -> np.ndarray:
        """Apply cone constraint correction to control."""
        T, tau_x, tau_y, tau_z = u

        cone_dist = self.cone.distance_to_cone_surface(self.quad_state.pos, deck_pos, deck_att)
        if cone_dist > 0:
            # Outside cone - add correction torque to steer back
            apex, axis = self.cone.get_cone_params(deck_pos, deck_att)
            to_axis = apex - self.quad_state.pos
            to_axis = to_axis - np.dot(to_axis, axis) * axis

            # Proportional correction
            correction_gain = 0.1 * min(cone_dist, 5.0)
            tau_x += correction_gain * to_axis[1]
            tau_y -= correction_gain * to_axis[0]

        return np.array([T, tau_x, tau_y, tau_z])

    def _apply_shutdown_logic(self, u: np.ndarray, height: float,
                               rel_vel: np.ndarray) -> np.ndarray:
        """Apply rotor shutdown logic near touchdown."""
        T, tau_x, tau_y, tau_z = u

        if height < 0.5 and not self.shutdown_commanded:
            rel_speed = np.linalg.norm(rel_vel)
            if rel_speed < 2.0:
                self.shutdown_commanded = True
                print(f"SHUTDOWN at t={self.t:.2f}s, height={height:.2f}m, rel_vel={rel_speed:.2f}m/s")

        if self.shutdown_commanded:
            T *= 0.2

        return np.array([T, tau_x, tau_y, tau_z])

    def _get_control_fallback(self, deck_pos: np.ndarray, deck_vel: np.ndarray,
                               deck_att: np.ndarray, rel_pos: np.ndarray,
                               rel_vel: np.ndarray, height: float,
                               horiz_dist: float) -> np.ndarray:
        """
        Fallback control using proportional navigation guidance.
        """
        g = 9.81
        m = self.quad_params.mass

        # Time to go estimate
        min_intercept_time = 5.0
        max_intercept_time = 15.0
        max_velocity = 8.0

        required_vel_for_fast = horiz_dist / min_intercept_time
        if required_vel_for_fast > max_velocity:
            t_go_horiz = horiz_dist / max_velocity
        else:
            t_go_horiz = min_intercept_time

        t_go_horiz = np.clip(t_go_horiz, min_intercept_time, max_intercept_time)

        if height > 2.0:
            t_go_vert = height / 1.5
        else:
            t_go_vert = height / 0.5

        t_go = max(t_go_horiz, t_go_vert, 3.0)

        # Horizontal guidance
        if horiz_dist > 5.0:
            pursuit_speed = min(horiz_dist / 5.0, 6.0)
            los = rel_pos[:2] / horiz_dist
            pursuit_vel = pursuit_speed * los + deck_vel[:2]
            vel_error = pursuit_vel - self.quad_state.vel[:2]
            horiz_acc_cmd = 2.0 * vel_error
            horiz_acc_cmd = np.clip(horiz_acc_cmd, -5.0, 5.0)
        else:
            pos_correction = 1.5 * rel_pos[:2]
            vel_error = deck_vel[:2] - self.quad_state.vel[:2]
            blend = horiz_dist / 5.0
            target_vel = blend * pos_correction + (1 - blend) * (deck_vel[:2] + pos_correction * 0.5)
            vel_error_total = target_vel - self.quad_state.vel[:2]
            horiz_acc_cmd = 3.0 * vel_error_total
            horiz_acc_cmd = np.clip(horiz_acc_cmd, -4.0, 4.0)

        # Vertical control
        target_descent_rate = height / max(t_go, 3.0)
        max_decel = 4.0
        effective_height = max(0, height - 0.5)
        max_safe_descent = np.sqrt(2 * max_decel * effective_height) if effective_height > 0 else 0.3
        target_descent_rate = min(target_descent_rate, max_safe_descent, 2.0)

        if height < 3.0:
            blend = height / 3.0
            target_descent_rate = blend * target_descent_rate
            horiz_acc_cmd *= blend

        desired_uav_vz = deck_vel[2] + target_descent_rate
        vert_vel_error = desired_uav_vz - self.quad_state.vel[2]
        acc_cmd_vert = np.clip(5.0 * vert_vel_error, -6, 6)

        acc_cmd = np.array([horiz_acc_cmd[0], horiz_acc_cmd[1], acc_cmd_vert])

        # Convert to thrust
        T_z = m * (g - acc_cmd[2])
        T = np.sqrt(T_z**2 + (m * acc_cmd[0])**2 + (m * acc_cmd[1])**2)
        T = np.clip(T, 0.1 * m * g, 1.5 * m * g)

        # Attitude from acceleration
        if T > 0.1 * m * g:
            roll_target = np.arctan2(acc_cmd[1], g)
            pitch_target = np.arctan2(-acc_cmd[0], g)
        else:
            roll_target = 0
            pitch_target = 0

        # Near touchdown: blend toward deck attitude
        if height < 3.0:
            att_blend = 1 - height / 3.0
            roll_target = (1 - att_blend) * roll_target + att_blend * deck_att[0]
            pitch_target = (1 - att_blend) * pitch_target + att_blend * deck_att[1]

        roll_target = np.clip(roll_target, -0.4, 0.4)
        pitch_target = np.clip(pitch_target, -0.4, 0.4)

        # Attitude control
        Kp_att, Kd_att = 25.0, 5.0
        tau_x = Kp_att * (roll_target - self.quad_state.roll) - Kd_att * self.quad_state.omega[0]
        tau_y = Kp_att * (pitch_target - self.quad_state.pitch) - Kd_att * self.quad_state.omega[1]
        tau_z = -Kd_att * self.quad_state.omega[2]

        u = np.array([T, tau_x, tau_y, tau_z])

        # Apply cone and shutdown
        u = self._apply_cone_constraint(u, deck_pos, deck_att)
        u = self._apply_shutdown_logic(u, height, rel_vel)

        return u

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

        # Replan at fixed interval
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
