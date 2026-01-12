"""
PMP-Based Trajectory Tracking Controller for Shipboard Landing

Combines pseudospectral optimal trajectory generation with
costate-based feedback control for tracking.

Key features:
1. Uses pseudospectral solver for fast trajectory optimization
2. Estimates costates from trajectory for optimal feedback
3. Tracks optimal trajectory with LQR-like feedback gains
4. Handles replanning when ARMA predictions update

Theory:
- Optimal control satisfies: u* = argmin H(x, u, λ)
- For quadratic cost: u* = -R^{-1} B'λ
- Costate provides gradient information for corrections
- Near-optimal feedback: δu = -R^{-1} B'(λ + K δx)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Callable
from scipy.linalg import solve_continuous_are
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quad_dynamics.quadrotor import QuadrotorParams, QuadrotorState


@dataclass
class PMPTrajectory:
    """Optimal trajectory from PMP/pseudospectral solver."""
    t: np.ndarray           # Time points [N]
    x: np.ndarray           # States [N, 12] - pos, vel, att, omega
    u: np.ndarray           # Controls [N, 4] - T, tau_x, tau_y, tau_z
    lam: np.ndarray         # Costates [N, 12]
    tf: float               # Final time

    # Target deck state at tf
    deck_pos: np.ndarray
    deck_vel: np.ndarray
    deck_att: np.ndarray


@dataclass
class ControllerGains:
    """Feedback gains for trajectory tracking."""
    # Position tracking
    Kp_pos: np.ndarray = None  # [3,] position gains
    Kd_pos: np.ndarray = None  # [3,] velocity gains

    # Attitude tracking
    Kp_att: np.ndarray = None  # [3,] attitude gains
    Kd_att: np.ndarray = None  # [3,] angular rate gains

    def __post_init__(self):
        if self.Kp_pos is None:
            self.Kp_pos = np.array([3.0, 3.0, 5.0])
        if self.Kd_pos is None:
            self.Kd_pos = np.array([4.0, 4.0, 6.0])
        if self.Kp_att is None:
            self.Kp_att = np.array([30.0, 30.0, 15.0])
        if self.Kd_att is None:
            self.Kd_att = np.array([8.0, 8.0, 5.0])


class PMPController:
    """
    PMP-based trajectory tracking controller.

    Uses optimal trajectory + costate feedback for near-optimal control.
    """

    def __init__(self, params: QuadrotorParams = None, gains: ControllerGains = None):
        self.params = params if params is not None else QuadrotorParams()
        self.gains = gains if gains is not None else ControllerGains()
        self.g = 9.81

        # Current trajectory
        self.trajectory: Optional[PMPTrajectory] = None
        self.trajectory_start_time: float = 0.0

        # State dimension
        self.nx = 12
        self.nu = 4

        # Cost weights (should match solver)
        self.R = np.diag([0.001, 0.01, 0.01, 0.01])
        self.R_inv = np.linalg.inv(self.R)

    def set_trajectory(self, traj: PMPTrajectory, start_time: float):
        """Set new optimal trajectory to track."""
        self.trajectory = traj
        self.trajectory_start_time = start_time

    def _interpolate_trajectory(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate trajectory at time t.

        Returns:
            x_ref: Reference state [12]
            u_ref: Reference control [4]
            lam_ref: Reference costate [12]
        """
        if self.trajectory is None:
            return None, None, None

        traj = self.trajectory
        t_rel = t - self.trajectory_start_time

        # Clamp to trajectory bounds
        t_rel = np.clip(t_rel, 0, traj.tf)

        # Linear interpolation
        idx = np.searchsorted(traj.t, t_rel)
        if idx == 0:
            return traj.x[0], traj.u[0], traj.lam[0]
        if idx >= len(traj.t):
            return traj.x[-1], traj.u[-1], traj.lam[-1]

        # Interpolation factor
        t0, t1 = traj.t[idx-1], traj.t[idx]
        alpha = (t_rel - t0) / (t1 - t0) if t1 > t0 else 0

        x_ref = (1 - alpha) * traj.x[idx-1] + alpha * traj.x[idx]
        u_ref = (1 - alpha) * traj.u[idx-1] + alpha * traj.u[idx]
        lam_ref = (1 - alpha) * traj.lam[idx-1] + alpha * traj.lam[idx]

        return x_ref, u_ref, lam_ref

    def _compute_B_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        Compute control influence matrix B = ∂f/∂u.

        For quadrotor:
        - Thrust affects velocity (through rotation)
        - Torques affect angular acceleration
        """
        p = self.params
        B = np.zeros((self.nx, self.nu))

        # Extract attitude
        phi, theta, psi = x[6:9]
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cy, sy = np.cos(psi), np.sin(psi)

        # Rotation matrix (body to NED)
        R = np.array([
            [ct*cy, sp*st*cy - cp*sy, cp*st*cy + sp*sy],
            [ct*sy, sp*st*sy + cp*cy, cp*st*sy - sp*cy],
            [-st,   sp*ct,            cp*ct]
        ])

        # Thrust affects velocity: a = R @ [0, 0, -T/m]'
        # ∂v_dot/∂T = R @ [0, 0, -1/m]'
        B[3:6, 0] = R @ np.array([0, 0, -1/p.mass])

        # Torques affect angular acceleration: ω_dot = J^{-1} @ tau
        # ∂ω_dot/∂tau = J^{-1}
        B[9:12, 1:4] = p.inertia_inv

        return B

    def compute_control(self, t: float, x_current: np.ndarray,
                       deck_pos: np.ndarray, deck_vel: np.ndarray,
                       deck_att: np.ndarray) -> np.ndarray:
        """
        Compute optimal control using trajectory tracking + costate feedback.

        Uses cascaded control:
        1. Outer loop: position/velocity tracking -> desired acceleration
        2. Inner loop: attitude control to achieve desired acceleration
        3. Costate correction: optimal adjustment based on costate gradient

        Args:
            t: Current time
            x_current: Current state [pos, vel, att, omega]
            deck_pos: Current deck position (for safety fallback)
            deck_vel: Current deck velocity
            deck_att: Current deck attitude

        Returns:
            u: Control [T, tau_x, tau_y, tau_z]
        """
        p = self.params
        g = self.gains
        m = p.mass

        # Get reference from trajectory
        x_ref, u_ref, lam_ref = self._interpolate_trajectory(t)

        if x_ref is None:
            return self._fallback_control(x_current, deck_pos, deck_vel, deck_att)

        # === POSITION/VELOCITY TRACKING ===
        # Blend trajectory reference with current deck prediction
        # As we get closer to landing, weight current deck more heavily

        pos_ref = x_ref[0:3]
        vel_ref = x_ref[3:6]
        pos_cur = x_current[0:3]
        vel_cur = x_current[3:6]

        # Calculate distance to deck for blending
        dist_to_deck = np.linalg.norm(deck_pos - pos_cur)

        # Time remaining in trajectory
        t_rel = t - self.trajectory_start_time
        t_remain = max(0.5, self.trajectory.tf - t_rel)

        # Use trajectory reference directly
        # The trajectory already targets the predicted deck position

        pos_error = pos_cur - pos_ref
        vel_error = vel_cur - vel_ref

        # Desired acceleration = feedforward + feedback
        # Feedforward: from trajectory (assume smooth trajectory has ~0 accel)
        # Feedback: PD control on error
        acc_des = -g.Kp_pos * pos_error - g.Kd_pos * vel_error

        # Add costate-based correction
        # Costate gives gradient of value function - points toward lower cost
        if lam_ref is not None:
            # Position costates give direction of improvement
            costate_correction = -0.1 * lam_ref[0:3] / (np.linalg.norm(lam_ref[0:3]) + 1e-6)
            acc_des += costate_correction

        # Clamp acceleration
        acc_des = np.clip(acc_des, -6, 6)

        # === THRUST AND ATTITUDE FROM DESIRED ACCELERATION ===
        # In NED frame with gravity along +z:
        # m*a = R*[0; 0; -T] + m*g*[0; 0; 1]
        # a = R*[0; 0; -T/m] + [0; 0; g]

        # Required thrust vector (what R*[0;0;-T/m] needs to be)
        thrust_vec_des = acc_des - np.array([0, 0, self.g])

        # Thrust magnitude
        T = m * np.linalg.norm(thrust_vec_des)
        T = np.clip(T, 0.1 * m * self.g, p.max_total_thrust)

        # Desired attitude from thrust direction
        if T > 0.2 * m * self.g:
            # Normalize thrust direction (should point along body -z)
            thrust_dir = thrust_vec_des / np.linalg.norm(thrust_vec_des)

            # Roll and pitch from thrust direction
            # thrust_dir = R @ [0, 0, -1]
            # For small angles: thrust_dir ≈ [pitch, -roll, -1]
            pitch_des = np.arcsin(np.clip(-thrust_dir[0], -1, 1))
            roll_des = np.arctan2(thrust_dir[1], -thrust_dir[2])
        else:
            roll_des = x_ref[6]
            pitch_des = x_ref[7]

        yaw_des = x_ref[8]  # Track reference yaw

        # Height above deck for blending
        height = deck_pos[2] - pos_cur[2]

        # Near touchdown: blend toward deck attitude
        if height < 3.0 and height > 0:
            blend = 1 - height / 3.0
            roll_des = (1 - blend) * roll_des + blend * deck_att[0]
            pitch_des = (1 - blend) * pitch_des + blend * deck_att[1]

        # Clamp desired attitude
        roll_des = np.clip(roll_des, -0.5, 0.5)
        pitch_des = np.clip(pitch_des, -0.5, 0.5)

        # === ATTITUDE CONTROL ===
        roll_cur = x_current[6]
        pitch_cur = x_current[7]
        yaw_cur = x_current[8]
        p_cur = x_current[9]
        q_cur = x_current[10]
        r_cur = x_current[11]

        # Reference angular rates (from trajectory or zero)
        p_ref = x_ref[9] if x_ref is not None else 0
        q_ref = x_ref[10] if x_ref is not None else 0
        r_ref = x_ref[11] if x_ref is not None else 0

        # Attitude errors
        roll_err = roll_cur - roll_des
        pitch_err = pitch_cur - pitch_des
        yaw_err = yaw_cur - yaw_des

        # Wrap yaw error to [-pi, pi]
        yaw_err = np.arctan2(np.sin(yaw_err), np.cos(yaw_err))

        # PD control
        tau_x = -g.Kp_att[0] * roll_err - g.Kd_att[0] * (p_cur - p_ref)
        tau_y = -g.Kp_att[1] * pitch_err - g.Kd_att[1] * (q_cur - q_ref)
        tau_z = -g.Kp_att[2] * yaw_err - g.Kd_att[2] * (r_cur - r_ref)

        # Clamp torques
        tau_max = p.max_total_thrust * p.arm_length / 2
        tau_x = np.clip(tau_x, -tau_max, tau_max)
        tau_y = np.clip(tau_y, -tau_max, tau_max)
        tau_z = np.clip(tau_z, -tau_max, tau_max)

        return np.array([T, tau_x, tau_y, tau_z])

    def _fallback_control(self, x: np.ndarray, deck_pos: np.ndarray,
                          deck_vel: np.ndarray, deck_att: np.ndarray) -> np.ndarray:
        """
        Fallback controller when no trajectory available.

        Uses proportional navigation toward deck.
        """
        p = self.params
        g = self.gains
        m = p.mass

        pos = x[0:3]
        vel = x[3:6]
        att = x[6:9]
        omega = x[9:12]

        # Relative state
        rel_pos = deck_pos - pos
        rel_vel = deck_vel - vel
        height = rel_pos[2]  # Positive when UAV above deck

        # Time to go estimate
        horiz_dist = np.linalg.norm(rel_pos[:2])
        t_go = max(horiz_dist / 5.0, height / 1.5, 3.0)

        # Desired velocity: position error / time-to-go + deck velocity
        vel_des = rel_pos / t_go + deck_vel
        vel_error = vel_des - vel

        # Acceleration command
        acc_cmd = g.Kd_pos * vel_error
        acc_cmd = np.clip(acc_cmd, -5, 5)

        # Thrust
        T_z = m * (self.g - acc_cmd[2])
        T = np.sqrt(T_z**2 + (m * acc_cmd[0])**2 + (m * acc_cmd[1])**2)
        T = np.clip(T, 0.3 * m * self.g, 1.3 * m * self.g)

        # Attitude from acceleration
        roll_des = np.arctan2(acc_cmd[1], self.g)
        pitch_des = np.arctan2(-acc_cmd[0], self.g)

        # Near deck: match deck attitude
        if height < 3.0 and height > 0:
            blend = 1 - height / 3.0
            roll_des = (1 - blend) * roll_des + blend * deck_att[0]
            pitch_des = (1 - blend) * pitch_des + blend * deck_att[1]

        # Attitude control
        tau_x = -g.Kp_att[0] * (att[0] - roll_des) - g.Kd_att[0] * omega[0]
        tau_y = -g.Kp_att[1] * (att[1] - pitch_des) - g.Kd_att[1] * omega[1]
        tau_z = -g.Kd_att[2] * omega[2]

        tau_max = p.max_total_thrust * p.arm_length / 2
        tau_x = np.clip(tau_x, -tau_max, tau_max)
        tau_y = np.clip(tau_y, -tau_max, tau_max)
        tau_z = np.clip(tau_z, -tau_max, tau_max)

        return np.array([T, tau_x, tau_y, tau_z])

    def time_remaining(self, t: float) -> float:
        """Get time remaining on current trajectory."""
        if self.trajectory is None:
            return float('inf')
        t_rel = t - self.trajectory_start_time
        return max(0, self.trajectory.tf - t_rel)

    def trajectory_valid(self, t: float) -> bool:
        """Check if trajectory is still valid (not expired)."""
        return self.trajectory is not None and self.time_remaining(t) > 0


class CostateEstimator:
    """
    Estimate costates from optimal trajectory for feedback.

    Uses the relationship:
    - λ_pos ≈ gradient of value function w.r.t. position
    - λ_vel = m * (v_des - v) for velocity matching
    - λ_att from terminal attitude matching
    """

    def __init__(self, params: QuadrotorParams = None):
        self.params = params if params is not None else QuadrotorParams()

        # Terminal cost weights (Qf from PMP formulation)
        self.Qf = np.diag([
            100, 100, 100,   # position
            50, 50, 50,      # velocity
            20, 20, 5,       # attitude (roll, pitch more important than yaw)
            5, 5, 2          # angular rates
        ])

    def estimate_costates(self, x_traj: np.ndarray, u_traj: np.ndarray,
                          t_traj: np.ndarray, deck_state: np.ndarray) -> np.ndarray:
        """
        Estimate costates along trajectory.

        Uses backward sweep approximation:
        λ(tf) = Qf @ (x(tf) - x_des)
        λ(t) ≈ backward integration of costate dynamics

        For simplicity, use linear interpolation from terminal condition.

        Args:
            x_traj: State trajectory [N, 12]
            u_traj: Control trajectory [N, 4]
            t_traj: Time points [N]
            deck_state: Target deck state [12]

        Returns:
            lam_traj: Costate trajectory [N, 12]
        """
        N = len(t_traj)
        lam_traj = np.zeros((N, 12))

        # Terminal costate from terminal cost gradient
        x_error_tf = x_traj[-1] - deck_state
        lam_tf = self.Qf @ x_error_tf
        lam_traj[-1] = lam_tf

        # Backward approximation
        # Costate tends to decay backward from terminal condition
        # Use exponential decay as approximation
        tf = t_traj[-1]
        for i in range(N - 2, -1, -1):
            tau = tf - t_traj[i]  # Time to go
            decay = np.exp(-0.5 * tau)  # Decay factor

            # Interpolate between zero and terminal costate
            lam_traj[i] = decay * lam_tf

            # Add contribution from state error at this point
            x_error = x_traj[i] - (deck_state - (deck_state - x_traj[-1]) * tau / tf)
            lam_traj[i] += 0.3 * self.Qf @ x_error * (1 - decay)

        return lam_traj


def create_pmp_trajectory(x_traj: np.ndarray, u_traj: np.ndarray,
                          t_traj: np.ndarray, tf: float,
                          deck_pos: np.ndarray, deck_vel: np.ndarray,
                          deck_att: np.ndarray,
                          params: QuadrotorParams = None) -> PMPTrajectory:
    """
    Create PMPTrajectory from pseudospectral solution.

    Estimates costates for the trajectory.
    """
    params = params if params is not None else QuadrotorParams()
    estimator = CostateEstimator(params)

    # Deck state for costate estimation
    deck_state = np.concatenate([deck_pos, deck_vel, deck_att, np.zeros(3)])

    # Estimate costates
    lam_traj = estimator.estimate_costates(x_traj, u_traj, t_traj, deck_state)

    return PMPTrajectory(
        t=t_traj,
        x=x_traj,
        u=u_traj,
        lam=lam_traj,
        tf=tf,
        deck_pos=deck_pos,
        deck_vel=deck_vel,
        deck_att=deck_att
    )


def extract_waypoints(trajectory: PMPTrajectory,
                      interval: float = 0.5,
                      frame: str = 'NED') -> List[dict]:
    """
    Extract waypoints from PMP trajectory for ArduPilot Guided mode.

    Args:
        trajectory: PMPTrajectory object
        interval: Time interval between waypoints (seconds)
        frame: Coordinate frame ('NED' or 'ENU')

    Returns:
        List of waypoint dictionaries with:
            - time: Time along trajectory
            - position: [x, y, z] in specified frame
            - velocity: [vx, vy, vz] in specified frame
            - yaw: Heading angle (rad)
    """
    waypoints = []
    t_points = np.arange(0, trajectory.tf, interval)
    if t_points[-1] < trajectory.tf:
        t_points = np.append(t_points, trajectory.tf)

    for t in t_points:
        # Interpolate trajectory at this time
        idx = np.searchsorted(trajectory.t, t)
        if idx == 0:
            x = trajectory.x[0]
        elif idx >= len(trajectory.t):
            x = trajectory.x[-1]
        else:
            t0, t1 = trajectory.t[idx-1], trajectory.t[idx]
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0
            x = (1 - alpha) * trajectory.x[idx-1] + alpha * trajectory.x[idx]

        pos = x[:3]
        vel = x[3:6]
        yaw = x[8]  # Attitude index 8 is yaw

        # Convert frame if needed
        if frame == 'ENU':
            # NED to ENU: swap x<->y, negate z
            pos = np.array([pos[1], pos[0], -pos[2]])
            vel = np.array([vel[1], vel[0], -vel[2]])
            yaw = np.pi/2 - yaw  # NED yaw to ENU heading

        waypoints.append({
            'time': t,
            'position': pos.tolist(),
            'velocity': vel.tolist(),
            'yaw': float(yaw)
        })

    return waypoints


def trajectory_to_mavlink_messages(trajectory: PMPTrajectory,
                                   interval: float = 0.5,
                                   coordinate_frame: int = 1,
                                   origin_lat: float = 0.0,
                                   origin_lon: float = 0.0,
                                   origin_alt: float = 0.0) -> List[dict]:
    """
    Convert PMP trajectory to MAVLink SET_POSITION_TARGET_LOCAL_NED format.

    This generates the message parameters that can be sent via pymavlink to
    ArduPilot's Guided mode for trajectory following.

    Args:
        trajectory: PMPTrajectory object
        interval: Time interval between commands (seconds)
        coordinate_frame: MAVLink coordinate frame (1=LOCAL_NED, 7=LOCAL_ENU)
        origin_lat: Origin latitude for global conversion (optional)
        origin_lon: Origin longitude for global conversion (optional)
        origin_alt: Origin altitude for global conversion (optional)

    Returns:
        List of MAVLink message parameter dictionaries:
            - time_boot_ms: Timestamp (ms)
            - coordinate_frame: MAV_FRAME
            - type_mask: Bitmask for which fields to use
            - x, y, z: Position (m)
            - vx, vy, vz: Velocity (m/s)
            - afx, afy, afz: Acceleration (m/s²) - set to 0
            - yaw: Heading (rad)
            - yaw_rate: Yaw rate (rad/s)
    """
    messages = []
    t_points = np.arange(0, trajectory.tf, interval)
    if t_points[-1] < trajectory.tf:
        t_points = np.append(t_points, trajectory.tf)

    # Type mask: use position (0x7) + velocity (0x38) + yaw (0x400)
    # Ignore acceleration (0x1C0) and yaw_rate (0x800)
    type_mask = 0x1C0 | 0x800  # Ignore accel and yaw_rate = use pos, vel, yaw

    for i, t in enumerate(t_points):
        # Interpolate trajectory
        idx = np.searchsorted(trajectory.t, t)
        if idx == 0:
            x = trajectory.x[0]
        elif idx >= len(trajectory.t):
            x = trajectory.x[-1]
        else:
            t0, t1 = trajectory.t[idx-1], trajectory.t[idx]
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0
            x = (1 - alpha) * trajectory.x[idx-1] + alpha * trajectory.x[idx]

        msg = {
            'time_boot_ms': int(t * 1000),
            'coordinate_frame': coordinate_frame,
            'type_mask': type_mask,
            'x': float(x[0]),      # North position
            'y': float(x[1]),      # East position
            'z': float(x[2]),      # Down position (negative = up)
            'vx': float(x[3]),     # North velocity
            'vy': float(x[4]),     # East velocity
            'vz': float(x[5]),     # Down velocity
            'afx': 0.0,
            'afy': 0.0,
            'afz': 0.0,
            'yaw': float(x[8]),    # Yaw angle
            'yaw_rate': 0.0
        }
        messages.append(msg)

    return messages


def trajectory_to_mission_items(trajectory: PMPTrajectory,
                                n_waypoints: int = 10,
                                home_lat: float = 0.0,
                                home_lon: float = 0.0,
                                home_alt: float = 0.0) -> List[dict]:
    """
    Convert PMP trajectory to ArduPilot mission waypoint items.

    Creates a mission that can be uploaded via MAVLink for AUTO mode.

    Args:
        trajectory: PMPTrajectory object
        n_waypoints: Number of waypoints to generate
        home_lat, home_lon, home_alt: Home position for coordinate conversion

    Returns:
        List of mission item dictionaries compatible with pymavlink:
            - seq: Sequence number
            - command: MAV_CMD (16 = NAV_WAYPOINT)
            - frame: MAV_FRAME (3 = GLOBAL_RELATIVE_ALT)
            - param1-4: Command parameters
            - x, y, z: Position (lat, lon, alt)
    """
    items = []

    # Sample trajectory at regular intervals
    t_samples = np.linspace(0, trajectory.tf, n_waypoints)

    for seq, t in enumerate(t_samples):
        # Interpolate trajectory
        idx = np.searchsorted(trajectory.t, t)
        if idx == 0:
            x = trajectory.x[0]
        elif idx >= len(trajectory.t):
            x = trajectory.x[-1]
        else:
            t0, t1 = trajectory.t[idx-1], trajectory.t[idx]
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0
            x = (1 - alpha) * trajectory.x[idx-1] + alpha * trajectory.x[idx]

        # Convert NED position to lat/lon (simple flat-earth approximation)
        # More accurate conversion would use pyproj or navpy
        METERS_PER_DEG_LAT = 111320.0
        meters_per_deg_lon = 111320.0 * np.cos(np.radians(home_lat))

        lat = home_lat + x[0] / METERS_PER_DEG_LAT
        lon = home_lon + x[1] / max(meters_per_deg_lon, 1.0)
        alt = home_alt - x[2]  # NED z is down, altitude is up

        item = {
            'seq': seq,
            'command': 16,  # MAV_CMD_NAV_WAYPOINT
            'frame': 3,     # MAV_FRAME_GLOBAL_RELATIVE_ALT
            'current': 1 if seq == 0 else 0,
            'autocontinue': 1,
            'param1': 0.0,  # Hold time (s)
            'param2': 1.0,  # Acceptance radius (m)
            'param3': 0.0,  # Pass through (0 = stop at waypoint)
            'param4': float(np.degrees(x[8])),  # Yaw angle (deg)
            'x': lat,
            'y': lon,
            'z': alt
        }
        items.append(item)

    return items


def demo():
    """Demonstrate PMP controller."""
    print("PMP Controller Demo")
    print("=" * 50)

    params = QuadrotorParams()
    controller = PMPController(params)

    # Create a simple test trajectory
    tf = 5.0
    N = 50
    t_traj = np.linspace(0, tf, N)

    # Straight line descent
    x_traj = np.zeros((N, 12))
    for i, t in enumerate(t_traj):
        alpha = t / tf
        x_traj[i, 0:3] = np.array([-30 + 30*alpha, 0, -25 + 17*alpha])  # pos
        x_traj[i, 3:6] = np.array([6, 0, 3.4])  # vel

    # Hover thrust trajectory
    u_traj = np.zeros((N, 4))
    u_traj[:, 0] = params.mass * 9.81

    # Deck state
    deck_pos = np.array([0, 0, -8])
    deck_vel = np.array([7.7, 0, 0])
    deck_att = np.array([0.05, 0.02, 0])

    # Create trajectory
    traj = create_pmp_trajectory(x_traj, u_traj, t_traj, tf,
                                  deck_pos, deck_vel, deck_att, params)
    controller.set_trajectory(traj, 0)

    print(f"Trajectory: {N} points, tf={tf}s")
    print(f"Costate at t=0: {traj.lam[0, :3]} (position)")
    print(f"Costate at tf: {traj.lam[-1, :3]} (position)")

    # Test control computation
    x_test = np.array([-25, 1, -22, 5, 0.5, 2, 0, 0, 0, 0, 0, 0])
    t_test = 1.0

    u = controller.compute_control(t_test, x_test, deck_pos, deck_vel, deck_att)
    print(f"\nControl at t={t_test}s:")
    print(f"  Thrust: {u[0]:.1f} N (hover={params.mass*9.81:.1f})")
    print(f"  Torques: [{u[1]:.3f}, {u[2]:.3f}, {u[3]:.3f}] Nm")

    print("\nDemo complete.")


if __name__ == "__main__":
    demo()
