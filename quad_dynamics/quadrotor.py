"""
6-DOF Quadrotor Dynamics for Shipboard Landing

Full nonlinear dynamics with rotor dynamics for optimal control.
Includes rotor spindown model for touchdown timing.

Reference frames:
- NED: North-East-Down inertial frame
- Body: Quadrotor body frame (x forward, y right, z down)

State vector (13 states):
- Position: [x, y, z] in NED (m)
- Velocity: [vx, vy, vz] in NED (m/s)
- Quaternion: [qw, qx, qy, qz] (attitude)
- Angular rate: [p, q, r] in body frame (rad/s)

Control inputs (4):
- [u1, u2, u3, u4] = normalized motor commands (0-1)
  or [T1, T2, T3, T4] = individual rotor thrusts (N)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Callable
from enum import Enum


class RotorConfig(Enum):
    """Rotor configuration."""
    X_CONFIG = "x"      # X configuration (rotors at 45°)
    PLUS_CONFIG = "+"   # Plus configuration (rotors at 0°/90°)


@dataclass
class QuadrotorParams:
    """Quadrotor physical parameters."""
    # Mass and inertia
    mass: float = 2.5           # kg (typical small quad)
    Ixx: float = 0.0142         # kg·m² (roll inertia)
    Iyy: float = 0.0142         # kg·m² (pitch inertia)
    Izz: float = 0.0267         # kg·m² (yaw inertia)

    # Geometry
    arm_length: float = 0.23    # m (motor to center)
    rotor_config: RotorConfig = RotorConfig.X_CONFIG

    # Rotor parameters
    k_thrust: float = 8.54e-6   # Thrust coefficient (N/(rad/s)²)
    k_torque: float = 1.36e-7   # Torque coefficient (N·m/(rad/s)²)
    max_rpm: float = 10000.0    # Maximum rotor RPM
    min_rpm: float = 1000.0     # Minimum rotor RPM (idle)
    rotor_inertia: float = 3.5e-5  # kg·m² per rotor

    # Rotor dynamics
    tau_rotor: float = 0.05     # Rotor time constant (s)

    # Aerodynamics
    Cd: float = 0.1             # Drag coefficient
    A_ref: float = 0.1          # Reference area (m²)

    # Motor direction (1 = CCW, -1 = CW looking from above)
    # Standard: motors 1,3 CCW, motors 2,4 CW
    motor_dirs: Tuple[int, ...] = (1, -1, 1, -1)

    @property
    def max_thrust_per_rotor(self) -> float:
        """Maximum thrust per rotor (N)."""
        omega_max = self.max_rpm * 2 * np.pi / 60
        return self.k_thrust * omega_max**2

    @property
    def max_total_thrust(self) -> float:
        """Maximum total thrust (N)."""
        return 4 * self.max_thrust_per_rotor

    @property
    def hover_throttle(self) -> float:
        """Throttle for hover (0-1)."""
        g = 9.81
        thrust_needed = self.mass * g
        return np.sqrt(thrust_needed / self.max_total_thrust)

    @property
    def inertia_matrix(self) -> np.ndarray:
        """Inertia matrix J."""
        return np.diag([self.Ixx, self.Iyy, self.Izz])

    @property
    def inertia_inv(self) -> np.ndarray:
        """Inverse inertia matrix."""
        return np.diag([1/self.Ixx, 1/self.Iyy, 1/self.Izz])


class QuadrotorState:
    """Quadrotor state representation."""

    def __init__(self, pos: np.ndarray = None, vel: np.ndarray = None,
                 quat: np.ndarray = None, omega: np.ndarray = None,
                 rotor_speeds: np.ndarray = None):
        """
        Initialize state.

        Args:
            pos: [x, y, z] NED position (m)
            vel: [vx, vy, vz] NED velocity (m/s)
            quat: [qw, qx, qy, qz] quaternion (scalar first)
            omega: [p, q, r] body angular rates (rad/s)
            rotor_speeds: [ω1, ω2, ω3, ω4] rotor speeds (rad/s)
        """
        self.pos = pos if pos is not None else np.zeros(3)
        self.vel = vel if vel is not None else np.zeros(3)
        self.quat = quat if quat is not None else np.array([1, 0, 0, 0])
        self.omega = omega if omega is not None else np.zeros(3)
        self.rotor_speeds = rotor_speeds if rotor_speeds is not None else np.zeros(4)

    def to_array(self) -> np.ndarray:
        """Convert to state vector [pos, vel, quat, omega, rotor_speeds]."""
        return np.concatenate([self.pos, self.vel, self.quat, self.omega, self.rotor_speeds])

    @classmethod
    def from_array(cls, x: np.ndarray) -> 'QuadrotorState':
        """Create state from array."""
        return cls(
            pos=x[0:3],
            vel=x[3:6],
            quat=x[6:10],
            omega=x[10:13],
            rotor_speeds=x[13:17] if len(x) > 13 else np.zeros(4)
        )

    @property
    def roll(self) -> float:
        """Roll angle (rad)."""
        return self.quat_to_euler()[0]

    @property
    def pitch(self) -> float:
        """Pitch angle (rad)."""
        return self.quat_to_euler()[1]

    @property
    def yaw(self) -> float:
        """Yaw angle (rad)."""
        return self.quat_to_euler()[2]

    def quat_to_euler(self) -> np.ndarray:
        """Convert quaternion to Euler angles [roll, pitch, yaw]."""
        qw, qx, qy, qz = self.quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx**2 + qy**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        sinp = np.clip(sinp, -1, 1)
        pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy**2 + qz**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def rotation_matrix(self) -> np.ndarray:
        """Rotation matrix from body to NED frame."""
        qw, qx, qy, qz = self.quat
        return np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])

    def normalize_quaternion(self):
        """Normalize quaternion to unit length."""
        norm = np.linalg.norm(self.quat)
        if norm > 1e-6:
            self.quat = self.quat / norm


class QuadrotorDynamics:
    """
    6-DOF quadrotor dynamics model.

    Includes:
    - Full nonlinear rigid body dynamics
    - Rotor dynamics (first-order lag)
    - Gyroscopic effects from rotors
    - Aerodynamic drag
    """

    def __init__(self, params: QuadrotorParams = None):
        self.params = params if params is not None else QuadrotorParams()
        self.g = 9.81  # Gravity (m/s²)

        # Compute allocation matrix (maps rotor thrusts to body forces/torques)
        self._compute_allocation_matrix()

    def _compute_allocation_matrix(self):
        """
        Compute control allocation matrix.

        Maps [T1, T2, T3, T4] to [Fz, τx, τy, τz]
        """
        p = self.params
        L = p.arm_length
        k_t = p.k_thrust
        k_q = p.k_torque

        if p.rotor_config == RotorConfig.X_CONFIG:
            # X configuration: rotors at 45° angles
            # Motor positions (looking from above):
            #   1 (front-right) at 45°
            #   2 (rear-right) at 135°
            #   3 (rear-left) at 225°
            #   4 (front-left) at 315°
            angle = np.pi / 4
            c, s = np.cos(angle), np.sin(angle)

            # Thrust contributes to Fz (down in body frame)
            # Roll (τx) from y-moment arm
            # Pitch (τy) from x-moment arm
            # Yaw (τz) from rotor torque reaction
            self.alloc_matrix = np.array([
                [1, 1, 1, 1],                           # Fz = sum of thrusts
                [L*s, L*s, -L*s, -L*s],                 # τx (roll)
                [L*c, -L*c, -L*c, L*c],                 # τy (pitch)
                [-k_q/k_t * p.motor_dirs[0],
                 -k_q/k_t * p.motor_dirs[1],
                 -k_q/k_t * p.motor_dirs[2],
                 -k_q/k_t * p.motor_dirs[3]]           # τz (yaw)
            ])
        else:
            # Plus configuration
            self.alloc_matrix = np.array([
                [1, 1, 1, 1],                           # Fz
                [0, L, 0, -L],                          # τx (roll)
                [L, 0, -L, 0],                          # τy (pitch)
                [-k_q/k_t * p.motor_dirs[0],
                 -k_q/k_t * p.motor_dirs[1],
                 -k_q/k_t * p.motor_dirs[2],
                 -k_q/k_t * p.motor_dirs[3]]           # τz (yaw)
            ])

        # Inverse for control allocation
        self.alloc_matrix_inv = np.linalg.pinv(self.alloc_matrix)

    def rotor_speed_to_thrust(self, omega: float) -> float:
        """Convert rotor speed (rad/s) to thrust (N)."""
        return self.params.k_thrust * omega**2

    def thrust_to_rotor_speed(self, thrust: float) -> float:
        """Convert thrust (N) to rotor speed (rad/s)."""
        thrust = max(0, thrust)
        return np.sqrt(thrust / self.params.k_thrust)

    def throttle_to_rotor_speed(self, throttle: float) -> float:
        """Convert throttle (0-1) to rotor speed (rad/s)."""
        throttle = np.clip(throttle, 0, 1)
        rpm_min = self.params.min_rpm
        rpm_max = self.params.max_rpm
        rpm = rpm_min + throttle * (rpm_max - rpm_min)
        return rpm * 2 * np.pi / 60

    def compute_forces_torques(self, state: QuadrotorState,
                                rotor_thrusts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute body forces and torques from rotor thrusts.

        Args:
            state: Current state
            rotor_thrusts: [T1, T2, T3, T4] rotor thrusts (N)

        Returns:
            F_body: [Fx, Fy, Fz] body frame forces (N)
            tau_body: [τx, τy, τz] body frame torques (N·m)
        """
        # Map thrusts to forces/torques
        forces_torques = self.alloc_matrix @ rotor_thrusts

        # Body force: rotors create thrust along body -z axis (up in body frame)
        # In NED body convention, -z is up, so thrust is negative z
        F_body = np.array([0, 0, -forces_torques[0]])

        # Body torques
        tau_body = forces_torques[1:4]

        # Add gyroscopic torque from rotors
        p = self.params
        omega_body = state.omega
        total_rotor_angular_momentum = 0
        for i in range(4):
            # Each rotor contributes angular momentum along z
            omega_rotor = self.thrust_to_rotor_speed(rotor_thrusts[i])
            total_rotor_angular_momentum += p.motor_dirs[i] * p.rotor_inertia * omega_rotor

        # Gyroscopic torque: τ_gyro = ω × h_rotor
        tau_gyro = np.array([
            -omega_body[1] * total_rotor_angular_momentum,  # affects roll
            omega_body[0] * total_rotor_angular_momentum,   # affects pitch
            0
        ])
        tau_body += tau_gyro

        return F_body, tau_body

    def compute_drag(self, state: QuadrotorState) -> np.ndarray:
        """Compute aerodynamic drag force in NED frame."""
        p = self.params
        v = state.vel
        v_mag = np.linalg.norm(v)

        if v_mag < 0.1:
            return np.zeros(3)

        # Drag: F_d = -0.5 * ρ * Cd * A * |v| * v
        rho = 1.225  # Air density (kg/m³)
        F_drag = -0.5 * rho * p.Cd * p.A_ref * v_mag * v

        return F_drag

    def dynamics(self, state: QuadrotorState, u: np.ndarray) -> np.ndarray:
        """
        Compute state derivative.

        Args:
            state: Current state
            u: Control input [T1, T2, T3, T4] rotor thrusts (N)
                or [u1, u2, u3, u4] throttle commands (0-1)

        Returns:
            x_dot: State derivative
        """
        p = self.params

        # If inputs are throttle (0-1), convert to thrust
        if np.all(u <= 1.0) and np.all(u >= 0.0):
            rotor_speeds = np.array([self.throttle_to_rotor_speed(ui) for ui in u])
            rotor_thrusts = np.array([self.rotor_speed_to_thrust(w) for w in rotor_speeds])
        else:
            rotor_thrusts = u

        # Body forces and torques
        F_body, tau_body = self.compute_forces_torques(state, rotor_thrusts)

        # Rotation matrix (body to NED)
        R = state.rotation_matrix()

        # Position derivative (NED velocity)
        pos_dot = state.vel

        # Velocity derivative (Newton's law in NED)
        # F_ned = R @ F_body + [0, 0, mg] + F_drag
        F_gravity = np.array([0, 0, p.mass * self.g])  # Down is positive in NED
        F_drag = self.compute_drag(state)
        F_ned = R @ F_body + F_gravity + F_drag
        vel_dot = F_ned / p.mass

        # Quaternion derivative
        # q_dot = 0.5 * q ⊗ [0, ω]
        qw, qx, qy, qz = state.quat
        p_rate, q_rate, r_rate = state.omega
        quat_dot = 0.5 * np.array([
            -qx*p_rate - qy*q_rate - qz*r_rate,
            qw*p_rate + qy*r_rate - qz*q_rate,
            qw*q_rate + qz*p_rate - qx*r_rate,
            qw*r_rate + qx*q_rate - qy*p_rate
        ])

        # Angular velocity derivative (Euler's equation)
        # J @ ω_dot = τ - ω × (J @ ω)
        J = p.inertia_matrix
        omega = state.omega
        omega_dot = p.inertia_inv @ (tau_body - np.cross(omega, J @ omega))

        return np.concatenate([pos_dot, vel_dot, quat_dot, omega_dot])

    def step(self, state: QuadrotorState, u: np.ndarray, dt: float) -> QuadrotorState:
        """
        Integrate dynamics for one time step using RK4.

        Args:
            state: Current state
            u: Control input
            dt: Time step (s)

        Returns:
            new_state: State after dt
        """
        x = state.to_array()[:13]  # Exclude rotor speeds for basic dynamics

        def f(x_):
            s = QuadrotorState.from_array(np.concatenate([x_, np.zeros(4)]))
            return self.dynamics(s, u)

        # RK4 integration
        k1 = f(x)
        k2 = f(x + 0.5*dt*k1)
        k3 = f(x + 0.5*dt*k2)
        k4 = f(x + dt*k3)

        x_new = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        # Create new state
        new_state = QuadrotorState.from_array(np.concatenate([x_new, np.zeros(4)]))
        new_state.normalize_quaternion()

        return new_state


class RotorShutdownModel:
    """
    Model for rotor spindown during landing.

    Predicts rotor behavior during shutdown sequence
    for precise touchdown timing.
    """

    def __init__(self, params: QuadrotorParams):
        self.params = params

    def spindown_time(self, initial_rpm: float, final_rpm: float = 0) -> float:
        """
        Estimate time to spin down from initial to final RPM.

        Simple first-order model: ω(t) = ω0 * exp(-t/τ)
        """
        if initial_rpm <= final_rpm:
            return 0.0

        tau = self.params.tau_rotor * 5  # Spindown is slower than spinup
        # Time to reach final_rpm (or 1% of initial if final=0)
        target = max(final_rpm, 0.01 * initial_rpm)
        t = -tau * np.log(target / initial_rpm)
        return t

    def thrust_during_spindown(self, t: float, initial_thrust: float) -> float:
        """
        Thrust at time t after shutdown command.

        Args:
            t: Time since shutdown command (s)
            initial_thrust: Thrust at shutdown (N)

        Returns:
            Current thrust (N)
        """
        tau = self.params.tau_rotor * 5
        return initial_thrust * np.exp(-t / tau)

    def residual_thrust_at_touchdown(self, shutdown_advance: float,
                                      hover_thrust: float) -> float:
        """
        Calculate residual thrust at touchdown given shutdown timing.

        Args:
            shutdown_advance: Time before touchdown to command shutdown (s)
            hover_thrust: Thrust at moment of shutdown command (N)

        Returns:
            Residual thrust at touchdown (N)
        """
        return self.thrust_during_spindown(shutdown_advance, hover_thrust)


def simulate_hover(duration: float = 10.0, dt: float = 0.01) -> dict:
    """Simulate hover to verify dynamics."""
    params = QuadrotorParams()
    dynamics = QuadrotorDynamics(params)

    # Initial state: 10m altitude, level
    state = QuadrotorState(
        pos=np.array([0, 0, -10]),  # 10m up (NED: negative z is up)
        vel=np.zeros(3),
        quat=np.array([1, 0, 0, 0]),
        omega=np.zeros(3)
    )

    # Hover throttle
    hover = params.hover_throttle
    u = np.array([hover, hover, hover, hover])

    # Simulate
    t = 0
    history = {'t': [], 'z': [], 'vz': [], 'roll': [], 'pitch': []}

    while t < duration:
        history['t'].append(t)
        history['z'].append(-state.pos[2])  # Convert to altitude
        history['vz'].append(-state.vel[2])
        history['roll'].append(np.degrees(state.roll))
        history['pitch'].append(np.degrees(state.pitch))

        state = dynamics.step(state, u, dt)
        t += dt

    return history


def simulate_landing_approach(initial_alt: float = 20.0,
                               descent_rate: float = 2.0,
                               duration: float = 15.0,
                               dt: float = 0.01) -> dict:
    """Simulate a simple descent to test dynamics."""
    params = QuadrotorParams()
    dynamics = QuadrotorDynamics(params)
    shutdown = RotorShutdownModel(params)

    # Initial state
    state = QuadrotorState(
        pos=np.array([0, 0, -initial_alt]),
        vel=np.array([0, 0, descent_rate]),  # Descending (positive vz in NED)
        quat=np.array([1, 0, 0, 0]),
        omega=np.zeros(3)
    )

    # Simple proportional control for descent rate
    hover = params.hover_throttle
    g = 9.81
    m = params.mass

    t = 0
    history = {'t': [], 'z': [], 'vz': [], 'thrust': []}
    shutdown_commanded = False
    shutdown_time = None

    while t < duration:
        alt = -state.pos[2]
        vz = state.vel[2]

        history['t'].append(t)
        history['z'].append(alt)
        history['vz'].append(vz)

        if alt < 0.5 and not shutdown_commanded:
            # Command shutdown 0.5m above ground
            shutdown_commanded = True
            shutdown_time = t
            print(f"Shutdown commanded at t={t:.2f}s, alt={alt:.2f}m, vz={vz:.2f}m/s")

        if shutdown_commanded:
            # Spindown
            t_since_shutdown = t - shutdown_time
            thrust_factor = np.exp(-t_since_shutdown / (params.tau_rotor * 5))
            u = np.array([hover * thrust_factor] * 4)
        else:
            # Simple P control for descent rate
            vz_error = vz - descent_rate
            throttle_adjust = -0.1 * vz_error
            throttle = hover + throttle_adjust
            throttle = np.clip(throttle, 0.1, 0.9)
            u = np.array([throttle] * 4)

        thrust = 4 * dynamics.rotor_speed_to_thrust(
            dynamics.throttle_to_rotor_speed(u[0])
        )
        history['thrust'].append(thrust)

        state = dynamics.step(state, u, dt)
        t += dt

        # Check if landed
        if state.pos[2] >= 0:
            print(f"Touchdown at t={t:.2f}s, vz={state.vel[2]:.2f}m/s")
            break

    return history


def demo():
    """Demonstrate quadrotor dynamics."""
    print("Quadrotor Dynamics Demo")
    print("=" * 50)

    params = QuadrotorParams()
    print(f"Mass: {params.mass} kg")
    print(f"Arm length: {params.arm_length} m")
    print(f"Max thrust: {params.max_total_thrust:.1f} N")
    print(f"Hover throttle: {params.hover_throttle:.3f}")
    print(f"Thrust/weight ratio: {params.max_total_thrust / (params.mass * 9.81):.2f}")
    print()

    # Test hover
    print("Testing hover stability...")
    history = simulate_hover(duration=5.0)
    alt_mean = np.mean(history['z'])
    alt_std = np.std(history['z'])
    print(f"Hover altitude: {alt_mean:.2f} ± {alt_std:.4f} m")
    print()

    # Test descent
    print("Testing descent and shutdown...")
    history = simulate_landing_approach(initial_alt=15.0, descent_rate=1.5)
    print()

    # Rotor shutdown model
    print("Rotor shutdown characteristics:")
    shutdown = RotorShutdownModel(params)
    for advance in [0.1, 0.2, 0.5, 1.0]:
        hover_thrust = params.mass * 9.81 / 4  # Per rotor at hover
        residual = shutdown.residual_thrust_at_touchdown(advance, hover_thrust)
        print(f"  Shutdown {advance:.1f}s before: residual thrust = {residual:.2f}N ({100*residual/hover_thrust:.1f}%)")

    print("\nDemo complete.")


if __name__ == "__main__":
    demo()
