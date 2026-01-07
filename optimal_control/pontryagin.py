"""
Pontryagin Minimum Principle for Shipboard Landing

Optimal trajectory generation for quadrotor landing on moving ship deck.
Matches position, velocity, and attitude at touchdown while minimizing
control effort.

Problem formulation:
- Minimize: J = ∫(u'Ru)dt + terminal cost
- Subject to: ẋ = f(x, u), x(0) = x0, ψ(x(tf), xd(tf)) = 0

Terminal constraints:
- Position: r_quad(tf) = r_deck(tf)
- Velocity: v_quad(tf) = v_deck(tf)
- Attitude: level relative to deck
- Angular rates: matched to deck rates

Pontryagin conditions:
- Hamiltonian: H = L(x,u) + λ'f(x,u)
- Optimal control: ∂H/∂u = 0
- Costate dynamics: λ̇ = -∂H/∂x
- Transversality: λ(tf) = ∂Φ/∂x + ν'∂ψ/∂x
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Callable, Optional, List
from scipy.optimize import minimize, root
from scipy.integrate import solve_ivp
import sys
sys.path.append('..')

from quad_dynamics.quadrotor import QuadrotorParams, QuadrotorState, QuadrotorDynamics


@dataclass
class LandingProblem:
    """Definition of the landing optimal control problem."""
    # Initial quad state
    x0_pos: np.ndarray      # [x, y, z] NED position (m)
    x0_vel: np.ndarray      # [vx, vy, vz] NED velocity (m/s)
    x0_att: np.ndarray      # [roll, pitch, yaw] (rad)

    # Terminal deck state (predicted at tf)
    xf_pos: np.ndarray      # Deck position at tf
    xf_vel: np.ndarray      # Deck velocity at tf
    xf_att: np.ndarray      # Deck attitude at tf (roll, pitch, yaw)
    xf_omega: np.ndarray    # Deck angular rates at tf

    # Time
    tf_guess: float = 5.0   # Initial guess for final time (s)
    tf_fixed: bool = False  # If True, tf is fixed; else, free final time

    # Cost weights
    R: np.ndarray = None    # Control cost weight (4x4)
    Qf: np.ndarray = None   # Terminal state cost (for soft constraints)

    def __post_init__(self):
        if self.R is None:
            self.R = np.eye(4) * 0.01  # Low control cost
        if self.Qf is None:
            self.Qf = np.diag([100, 100, 100,    # position
                              50, 50, 50,        # velocity
                              10, 10, 10,        # attitude
                              5, 5, 5])          # angular rate


class SimplifiedDynamics:
    """
    Simplified quadrotor dynamics for optimal control.

    Uses 12-state model:
    - Position [x, y, z] in NED (3)
    - Velocity [vx, vy, vz] in NED (3)
    - Attitude [φ, θ, ψ] Euler angles (3)
    - Angular rate [p, q, r] body rates (3)

    Control: [T, τx, τy, τz] = total thrust and body torques
    """

    def __init__(self, params: QuadrotorParams = None):
        self.params = params if params is not None else QuadrotorParams()
        self.g = 9.81
        self.n_states = 12
        self.n_controls = 4

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute state derivative.

        Args:
            x: State [pos(3), vel(3), att(3), omega(3)]
            u: Control [T, τx, τy, τz]

        Returns:
            x_dot: State derivative
        """
        p = self.params

        # Unpack state
        pos = x[0:3]
        vel = x[3:6]
        att = x[6:9]  # [roll, pitch, yaw]
        omega = x[9:12]  # [p, q, r]

        # Unpack control
        T = u[0]      # Total thrust
        tau = u[1:4]  # Body torques

        # Rotation matrix (body to NED)
        phi, theta, psi = att
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cy, sy = np.cos(psi), np.sin(psi)

        R = np.array([
            [ct*cy, sp*st*cy - cp*sy, cp*st*cy + sp*sy],
            [ct*sy, sp*st*sy + cp*cy, cp*st*sy - sp*cy],
            [-st,   sp*ct,            cp*ct]
        ])

        # Position derivative
        pos_dot = vel

        # Velocity derivative
        # Thrust in body frame: [0, 0, -T] (up)
        # Gravity in NED: [0, 0, mg] (down)
        thrust_body = np.array([0, 0, -T])
        thrust_ned = R @ thrust_body
        gravity = np.array([0, 0, p.mass * self.g])
        vel_dot = (thrust_ned + gravity) / p.mass

        # Attitude derivative (Euler rate from body rates)
        # Using small angle approximation for simplicity
        # More accurate: use kinematic equations
        if np.abs(theta) < np.pi/2 - 0.1:
            att_dot = np.array([
                omega[0] + sp*np.tan(theta)*omega[1] + cp*np.tan(theta)*omega[2],
                cp*omega[1] - sp*omega[2],
                sp/ct*omega[1] + cp/ct*omega[2]
            ])
        else:
            # Near singularity, use simplified
            att_dot = omega

        # Angular rate derivative (Euler's equation)
        J = p.inertia_matrix
        J_inv = p.inertia_inv
        omega_dot = J_inv @ (tau - np.cross(omega, J @ omega))

        return np.concatenate([pos_dot, vel_dot, att_dot, omega_dot])

    def linearize(self, x0: np.ndarray, u0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize dynamics around operating point.

        Returns:
            A: State Jacobian ∂f/∂x
            B: Control Jacobian ∂f/∂u
        """
        eps = 1e-6
        n, m = self.n_states, self.n_controls

        f0 = self.dynamics(x0, u0)

        # Numerical Jacobians
        A = np.zeros((n, n))
        for i in range(n):
            x_p = x0.copy()
            x_p[i] += eps
            A[:, i] = (self.dynamics(x_p, u0) - f0) / eps

        B = np.zeros((n, m))
        for i in range(m):
            u_p = u0.copy()
            u_p[i] += eps
            B[:, i] = (self.dynamics(x0, u_p) - f0) / eps

        return A, B


class PontryaginSolver:
    """
    Solve optimal landing problem using Pontryagin's Minimum Principle.

    Uses indirect shooting method:
    1. Guess initial costate λ(0)
    2. Integrate state and costate forward
    3. Check terminal conditions
    4. Adjust λ(0) to satisfy boundary conditions
    """

    def __init__(self, dynamics: SimplifiedDynamics, problem: LandingProblem):
        self.dyn = dynamics
        self.prob = problem
        self.n = dynamics.n_states
        self.m = dynamics.n_controls

    def running_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """Running cost L(x, u) = u'Ru."""
        return 0.5 * u @ self.prob.R @ u

    def terminal_cost(self, x: np.ndarray, xd: np.ndarray) -> float:
        """Terminal cost Φ(x(tf), xd(tf))."""
        error = x - xd
        return 0.5 * error @ self.prob.Qf @ error

    def hamiltonian(self, x: np.ndarray, u: np.ndarray, lam: np.ndarray) -> float:
        """Hamiltonian H = L + λ'f."""
        L = self.running_cost(x, u)
        f = self.dyn.dynamics(x, u)
        return L + lam @ f

    def optimal_control(self, x: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """
        Compute optimal control from ∂H/∂u = 0.

        For quadratic cost L = 0.5*u'Ru and linear control in dynamics:
        u* = -R^{-1} B' λ

        But our dynamics are nonlinear, so we minimize H numerically.
        """
        p = self.dyn.params

        # Control bounds
        T_min = 0
        T_max = p.max_total_thrust
        tau_max = p.max_total_thrust * p.arm_length / 2

        def objective(u):
            return self.hamiltonian(x, u, lam)

        # Gradient of H w.r.t. u
        def gradient(u):
            eps = 1e-6
            grad = np.zeros(4)
            H0 = objective(u)
            for i in range(4):
                u_p = u.copy()
                u_p[i] += eps
                grad[i] = (objective(u_p) - H0) / eps
            return grad

        # Initial guess: hover thrust, zero torques
        u0 = np.array([p.mass * self.dyn.g, 0, 0, 0])

        # Bounds
        bounds = [
            (T_min, T_max),
            (-tau_max, tau_max),
            (-tau_max, tau_max),
            (-tau_max, tau_max)
        ]

        # Minimize Hamiltonian
        result = minimize(objective, u0, method='L-BFGS-B', bounds=bounds,
                         jac=gradient, options={'maxiter': 50})

        return result.x

    def costate_dynamics(self, x: np.ndarray, lam: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Costate dynamics: λ̇ = -∂H/∂x.
        """
        eps = 1e-6
        lam_dot = np.zeros(self.n)

        H0 = self.hamiltonian(x, u, lam)
        for i in range(self.n):
            x_p = x.copy()
            x_p[i] += eps
            # Need optimal u for perturbed x
            u_p = self.optimal_control(x_p, lam)
            H_p = self.hamiltonian(x_p, u_p, lam)
            lam_dot[i] = -(H_p - H0) / eps

        return lam_dot

    def augmented_dynamics(self, t: float, z: np.ndarray) -> np.ndarray:
        """
        Augmented state-costate dynamics for integration.

        z = [x; λ]
        """
        x = z[:self.n]
        lam = z[self.n:]

        # Optimal control
        u = self.optimal_control(x, lam)

        # State dynamics
        x_dot = self.dyn.dynamics(x, u)

        # Costate dynamics
        lam_dot = self.costate_dynamics(x, lam, u)

        return np.concatenate([x_dot, lam_dot])

    def terminal_constraints(self, x_tf: np.ndarray) -> np.ndarray:
        """
        Terminal constraint ψ(x(tf), xd(tf)) = 0.

        Matches position, velocity to deck.
        Attitude: level (roll=pitch=0 relative to deck).
        """
        prob = self.prob

        # Target state at tf
        xd = np.concatenate([
            prob.xf_pos,
            prob.xf_vel,
            prob.xf_att,  # Deck attitude
            prob.xf_omega  # Deck angular rates
        ])

        # Position error
        pos_error = x_tf[0:3] - xd[0:3]

        # Velocity error
        vel_error = x_tf[3:6] - xd[3:6]

        # Attitude error (relative to deck)
        att_error = x_tf[6:9] - xd[6:9]

        # Angular rate error
        omega_error = x_tf[9:12] - xd[9:12]

        return np.concatenate([pos_error, vel_error, att_error, omega_error])

    def solve_shooting(self, lam0_guess: np.ndarray = None,
                       tf: float = None,
                       max_iter: int = 50) -> dict:
        """
        Solve using indirect single shooting.

        Args:
            lam0_guess: Initial costate guess
            tf: Final time
            max_iter: Maximum iterations

        Returns:
            Dictionary with solution
        """
        prob = self.prob
        tf = tf if tf is not None else prob.tf_guess

        # Initial state
        x0 = np.concatenate([
            prob.x0_pos,
            prob.x0_vel,
            prob.x0_att,
            np.zeros(3)  # Initial angular rates = 0
        ])

        # Initial costate guess
        if lam0_guess is None:
            lam0_guess = np.zeros(self.n)
            # Rough guess based on terminal cost gradient
            xd = np.concatenate([prob.xf_pos, prob.xf_vel, prob.xf_att, prob.xf_omega])
            lam0_guess = prob.Qf @ (x0 - xd) * 0.1

        def shooting_residual(lam0):
            """Residual to drive to zero."""
            z0 = np.concatenate([x0, lam0])

            # Integrate forward
            sol = solve_ivp(
                self.augmented_dynamics,
                [0, tf],
                z0,
                method='RK45',
                max_step=0.1
            )

            if not sol.success:
                return np.ones(self.n) * 1e6

            z_tf = sol.y[:, -1]
            x_tf = z_tf[:self.n]
            lam_tf = z_tf[self.n:]

            # Terminal constraints
            psi = self.terminal_constraints(x_tf)

            # Transversality (simplified)
            xd = np.concatenate([prob.xf_pos, prob.xf_vel, prob.xf_att, prob.xf_omega])
            lam_tf_required = prob.Qf @ (x_tf - xd)
            transversality = lam_tf - lam_tf_required

            return np.concatenate([psi, transversality])

        # Solve for initial costate
        print("Solving shooting problem...")
        result = root(shooting_residual, lam0_guess, method='hybr',
                     options={'maxfev': max_iter * self.n * 2})

        if not result.success:
            print(f"Warning: Shooting did not converge. Message: {result.message}")

        lam0_opt = result.x

        # Final integration with optimal initial costate
        z0 = np.concatenate([x0, lam0_opt])
        sol = solve_ivp(
            self.augmented_dynamics,
            [0, tf],
            z0,
            method='RK45',
            max_step=0.05,
            dense_output=True
        )

        # Extract trajectory
        t_traj = np.linspace(0, tf, 100)
        z_traj = sol.sol(t_traj)
        x_traj = z_traj[:self.n, :]
        lam_traj = z_traj[self.n:, :]

        # Compute optimal controls along trajectory
        u_traj = np.zeros((self.m, len(t_traj)))
        for i, t in enumerate(t_traj):
            u_traj[:, i] = self.optimal_control(x_traj[:, i], lam_traj[:, i])

        return {
            'success': result.success,
            't': t_traj,
            'x': x_traj,
            'lam': lam_traj,
            'u': u_traj,
            'tf': tf,
            'terminal_error': self.terminal_constraints(x_traj[:, -1])
        }


def solve_landing_trajectory(quad_pos: np.ndarray,
                             quad_vel: np.ndarray,
                             deck_pos: np.ndarray,
                             deck_vel: np.ndarray,
                             deck_att: np.ndarray,
                             deck_omega: np.ndarray,
                             tf_guess: float = 5.0) -> dict:
    """
    High-level interface to solve landing trajectory.

    Args:
        quad_pos: Current quadrotor position [x, y, z] NED (m)
        quad_vel: Current quadrotor velocity [vx, vy, vz] NED (m/s)
        deck_pos: Predicted deck position at tf
        deck_vel: Predicted deck velocity at tf
        deck_att: Predicted deck attitude at tf [roll, pitch, yaw]
        deck_omega: Predicted deck angular rates at tf [p, q, r]
        tf_guess: Initial guess for time-to-land (s)

    Returns:
        Solution dictionary with trajectory
    """
    # Create problem
    problem = LandingProblem(
        x0_pos=quad_pos,
        x0_vel=quad_vel,
        x0_att=np.zeros(3),  # Assume initially level
        xf_pos=deck_pos,
        xf_vel=deck_vel,
        xf_att=deck_att,
        xf_omega=deck_omega,
        tf_guess=tf_guess
    )

    # Create dynamics and solver
    params = QuadrotorParams()
    dynamics = SimplifiedDynamics(params)
    solver = PontryaginSolver(dynamics, problem)

    # Solve
    solution = solver.solve_shooting()

    return solution


def demo():
    """Demonstrate optimal landing trajectory."""
    print("Pontryagin Optimal Landing Demo")
    print("=" * 50)

    # Scenario: Quad approaching ship deck
    # Quad starts 20m above and 30m behind deck
    quad_pos = np.array([-30, 0, -20])  # 30m behind, 20m up
    quad_vel = np.array([5, 0, 0])       # Moving toward ship at 5 m/s

    # Deck state at predicted tf (ship moving forward)
    deck_pos = np.array([0, 0, -8])      # Deck at z=-8m (8m above water)
    deck_vel = np.array([7.7, 0, 0])     # Ship at 15 kts ≈ 7.7 m/s
    deck_att = np.array([0.05, 0.02, 0]) # Slight roll/pitch
    deck_omega = np.array([0.1, 0.05, 0]) # Some angular motion

    print("Initial conditions:")
    print(f"  Quad position: {quad_pos} m")
    print(f"  Quad velocity: {quad_vel} m/s")
    print(f"  Deck position: {deck_pos} m")
    print(f"  Deck velocity: {deck_vel} m/s")
    print()

    # Solve
    print("Computing optimal trajectory...")
    solution = solve_landing_trajectory(
        quad_pos, quad_vel,
        deck_pos, deck_vel, deck_att, deck_omega,
        tf_guess=6.0
    )

    if solution['success']:
        print("\nSolution found!")
    else:
        print("\nWarning: Solution may not be optimal")

    print(f"Final time: {solution['tf']:.2f} s")

    # Terminal errors
    err = solution['terminal_error']
    print(f"\nTerminal errors:")
    print(f"  Position: [{err[0]:.3f}, {err[1]:.3f}, {err[2]:.3f}] m")
    print(f"  Velocity: [{err[3]:.3f}, {err[4]:.3f}, {err[5]:.3f}] m/s")
    print(f"  Attitude: [{np.degrees(err[6]):.2f}, {np.degrees(err[7]):.2f}, {np.degrees(err[8]):.2f}] deg")

    # Trajectory summary
    x = solution['x']
    u = solution['u']
    t = solution['t']

    print(f"\nTrajectory summary:")
    print(f"  Initial altitude: {-x[2, 0]:.1f} m")
    print(f"  Final altitude: {-x[2, -1]:.1f} m")
    print(f"  Max thrust: {np.max(u[0, :]):.1f} N")
    print(f"  Min thrust: {np.min(u[0, :]):.1f} N")
    print(f"  Final descent rate: {x[5, -1]:.2f} m/s")

    print("\nDemo complete.")

    return solution


if __name__ == "__main__":
    demo()
