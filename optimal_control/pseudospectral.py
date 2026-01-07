"""
Pseudospectral Optimal Control for Shipboard Landing

Fast direct collocation method using Legendre-Gauss-Lobatto (LGL) nodes.
Much faster than indirect shooting for trajectory optimization.

Terminal constraints:
- Position match: r_quad = r_deck
- Velocity match: v_quad = v_deck (translation)
- Attitude match: roll_quad = roll_deck, pitch_quad = pitch_deck
- Thrust zero: T = 0 at touchdown (rotors shutdown)

References:
- Fahroo & Ross, "Direct Trajectory Optimization by a Chebyshev Pseudospectral Method"
- Garg et al., "A Unified Framework for the Numerical Solution of Optimal Control Problems"
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import legendre
from dataclasses import dataclass
from typing import Tuple, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quad_dynamics.quadrotor import QuadrotorParams


@dataclass
class LandingConstraints:
    """Terminal constraints for landing."""
    # Match position
    match_position: bool = True

    # Match velocity (translation)
    match_velocity: bool = True

    # Match attitude (roll, pitch only - yaw free)
    match_roll: bool = True
    match_pitch: bool = True

    # Zero thrust at touchdown
    zero_thrust: bool = True


def lgr_nodes(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Legendre-Gauss-Radau (LGR) collocation points and weights.

    LGR includes the initial point but not the final point.
    Good for free final state problems.

    For simplicity, using LGL (Legendre-Gauss-Lobatto) which includes both endpoints.
    """
    # LGL nodes are roots of (1-x²)P'_N(x) where P_N is Legendre polynomial
    # For N+1 points, they include -1 and +1

    if N < 2:
        raise ValueError("Need at least 2 nodes")

    # Chebyshev nodes as initial guess (close to LGL)
    theta = np.linspace(np.pi, 0, N + 1)
    x = np.cos(theta)

    # Newton iteration to find LGL nodes
    P = legendre(N)
    Pp = P.deriv()

    for _ in range(10):
        # LGL nodes satisfy: (1-x²)P'_N(x) = 0
        # Interior nodes satisfy P'_N(x) = 0
        Ppx = np.polyval(Pp, x[1:-1])
        Pppx = np.polyval(Pp.deriv(), x[1:-1])
        x[1:-1] = x[1:-1] - Ppx / Pppx

    # Weights
    Px = np.polyval(P, x)
    w = 2 / (N * (N + 1) * Px**2)

    return x, w


def differentiation_matrix(tau: np.ndarray) -> np.ndarray:
    """
    Compute Lagrange polynomial differentiation matrix D.

    D[i,j] = derivative of j-th Lagrange polynomial at tau[i]
    """
    N = len(tau)
    D = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                # Product of (tau_i - tau_k) for k != j
                num = 1.0
                den = 1.0
                for k in range(N):
                    if k != j:
                        num *= (tau[i] - tau[k])
                        den *= (tau[j] - tau[k])
                D[i, j] = num / (den * (tau[i] - tau[j]))

        # Diagonal: D[i,i] = -sum of off-diagonal in row i
        D[i, i] = -np.sum(D[i, :]) + D[i, i]

    return D


class PseudospectralSolver:
    """
    Pseudospectral (direct collocation) solver for landing trajectory.

    Transcribes continuous OCP to nonlinear program (NLP):
    - State at collocation nodes: X = [x_0, x_1, ..., x_N]
    - Control at collocation nodes: U = [u_0, u_1, ..., u_N]
    - Decision variables: z = [X, U, tf]

    Constraints:
    - Dynamics at collocation points: D*X = (tf/2)*f(X, U)
    - Initial conditions: x_0 = x_init
    - Terminal conditions: ψ(x_N, x_deck) = 0
    - Control bounds
    """

    def __init__(self, N: int = 20, params: QuadrotorParams = None):
        """
        Args:
            N: Number of collocation nodes
            params: Quadrotor parameters
        """
        self.N = N
        self.params = params if params is not None else QuadrotorParams()

        # LGL nodes and differentiation matrix
        self.tau, self.w = lgr_nodes(N)
        self.D = differentiation_matrix(self.tau)

        # State and control dimensions
        # Simplified state: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        self.nx = 12
        self.nu = 4  # [T, tau_x, tau_y, tau_z]

        # Gravity
        self.g = 9.81

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Simplified quadrotor dynamics.

        State: [x, y, z, vx, vy, vz, φ, θ, ψ, p, q, r]
        Control: [T, τx, τy, τz]
        """
        p = self.params

        # Unpack
        pos = x[0:3]
        vel = x[3:6]
        phi, theta, psi = x[6:9]
        omega = x[9:12]

        T = u[0]
        tau = u[1:4]

        # Trig
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cy, sy = np.cos(psi), np.sin(psi)

        # Rotation matrix (body to NED)
        R = np.array([
            [ct*cy, sp*st*cy - cp*sy, cp*st*cy + sp*sy],
            [ct*sy, sp*st*sy + cp*cy, cp*st*sy - sp*cy],
            [-st,   sp*ct,            cp*ct]
        ])

        # Position derivative
        pos_dot = vel

        # Velocity derivative
        thrust_body = np.array([0, 0, -T])
        thrust_ned = R @ thrust_body
        gravity = np.array([0, 0, p.mass * self.g])
        vel_dot = (thrust_ned + gravity) / p.mass

        # Attitude derivative (simplified - small angle for rates)
        # More accurate would use full Euler kinematic equations
        ct_safe = max(abs(ct), 0.1) * np.sign(ct) if ct != 0 else 0.1
        att_dot = np.array([
            omega[0] + sp * st / ct_safe * omega[1] + cp * st / ct_safe * omega[2],
            cp * omega[1] - sp * omega[2],
            sp / ct_safe * omega[1] + cp / ct_safe * omega[2]
        ])

        # Angular rate derivative
        J = p.inertia_matrix
        J_inv = p.inertia_inv
        omega_dot = J_inv @ (tau - np.cross(omega, J @ omega))

        return np.concatenate([pos_dot, vel_dot, att_dot, omega_dot])

    def pack_decision_vars(self, X: np.ndarray, U: np.ndarray, tf: float) -> np.ndarray:
        """Pack state, control, final time into decision vector."""
        return np.concatenate([X.flatten(), U.flatten(), [tf]])

    def unpack_decision_vars(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Unpack decision vector into state, control, final time."""
        N, nx, nu = self.N + 1, self.nx, self.nu

        X = z[:N * nx].reshape(N, nx)
        U = z[N * nx:N * nx + N * nu].reshape(N, nu)
        tf = z[-1]

        return X, U, tf

    def objective(self, z: np.ndarray) -> float:
        """
        Cost function: minimize control effort + time.

        J = ∫(u'Ru)dt + ρ*tf
        """
        X, U, tf = self.unpack_decision_vars(z)

        # Control cost (quadrature using LGL weights)
        R = np.diag([0.001, 0.01, 0.01, 0.01])  # Thrust cheap, torques moderate
        control_cost = 0
        for i in range(self.N + 1):
            control_cost += self.w[i] * (U[i] @ R @ U[i])
        control_cost *= tf / 2  # Scale to physical time

        # Time penalty (encourage faster landing)
        time_penalty = 0.5 * tf

        return control_cost + time_penalty

    def constraints(self, z: np.ndarray, x_init: np.ndarray,
                    deck_state: np.ndarray,
                    constraints: LandingConstraints) -> np.ndarray:
        """
        Equality constraints.

        Returns vector that should equal zero.
        """
        X, U, tf = self.unpack_decision_vars(z)
        p = self.params

        residuals = []

        # 1. Dynamics constraints at collocation points
        # D*X = (tf/2)*f(X, U)
        for i in range(self.N + 1):
            # Derivative from differentiation matrix
            x_dot_approx = np.zeros(self.nx)
            for j in range(self.N + 1):
                x_dot_approx += self.D[i, j] * X[j]
            x_dot_approx *= 2 / tf  # Transform from [-1,1] to [0,tf]

            # Derivative from dynamics
            x_dot_exact = self.dynamics(X[i], U[i])

            residuals.append(x_dot_approx - x_dot_exact)

        # 2. Initial conditions
        residuals.append(X[0] - x_init)

        # 3. Terminal conditions
        x_final = X[-1]
        deck_pos = deck_state[0:3]
        deck_vel = deck_state[3:6]
        deck_att = deck_state[6:9]

        if constraints.match_position:
            residuals.append(x_final[0:3] - deck_pos)

        if constraints.match_velocity:
            residuals.append(x_final[3:6] - deck_vel)

        if constraints.match_roll:
            residuals.append(np.array([x_final[6] - deck_att[0]]))

        if constraints.match_pitch:
            residuals.append(np.array([x_final[7] - deck_att[1]]))

        if constraints.zero_thrust:
            # Thrust = 0 at touchdown
            residuals.append(np.array([U[-1, 0]]))

        return np.concatenate(residuals)

    def bounds(self, x_init: np.ndarray, tf_min: float = 1.0, tf_max: float = 20.0):
        """
        Variable bounds.

        Returns (lower, upper) bound arrays.
        """
        N, nx, nu = self.N + 1, self.nx, self.nu
        p = self.params

        n_vars = N * nx + N * nu + 1

        lb = np.zeros(n_vars)
        ub = np.zeros(n_vars)

        # State bounds (fairly loose)
        for i in range(N):
            idx = i * nx
            # Position: large bounds
            lb[idx:idx+3] = -500
            ub[idx:idx+3] = 500
            # Velocity
            lb[idx+3:idx+6] = -50
            ub[idx+3:idx+6] = 50
            # Attitude (rad)
            lb[idx+6:idx+9] = -np.pi/2
            ub[idx+6:idx+9] = np.pi/2
            # Angular rates
            lb[idx+9:idx+12] = -10
            ub[idx+9:idx+12] = 10

        # Control bounds
        ctrl_start = N * nx
        T_max = p.max_total_thrust
        tau_max = T_max * p.arm_length / 2

        for i in range(N):
            idx = ctrl_start + i * nu
            lb[idx] = 0              # Thrust >= 0
            ub[idx] = T_max          # Thrust <= max
            lb[idx+1:idx+4] = -tau_max
            ub[idx+1:idx+4] = tau_max

        # Final time bounds
        lb[-1] = tf_min
        ub[-1] = tf_max

        return lb, ub

    def solve(self, x_init: np.ndarray, deck_state: np.ndarray,
              constraints: LandingConstraints = None,
              tf_guess: float = 5.0,
              verbose: bool = True) -> dict:
        """
        Solve the optimal landing problem.

        Args:
            x_init: Initial quadrotor state [pos, vel, att, omega]
            deck_state: Target deck state at tf [pos, vel, att]
            constraints: Terminal constraint specification
            tf_guess: Initial guess for final time
            verbose: Print progress

        Returns:
            Solution dictionary
        """
        if constraints is None:
            constraints = LandingConstraints()

        N, nx, nu = self.N + 1, self.nx, self.nu

        # Initial guess: linear interpolation
        X_guess = np.zeros((N, nx))
        U_guess = np.zeros((N, nu))

        for i in range(N):
            alpha = (self.tau[i] + 1) / 2  # 0 to 1
            X_guess[i, 0:3] = (1 - alpha) * x_init[0:3] + alpha * deck_state[0:3]
            X_guess[i, 3:6] = (1 - alpha) * x_init[3:6] + alpha * deck_state[3:6]
            X_guess[i, 6:9] = (1 - alpha) * x_init[6:9] + alpha * deck_state[6:9]
            U_guess[i, 0] = self.params.mass * self.g * (1 - alpha)  # Decrease thrust

        z0 = self.pack_decision_vars(X_guess, U_guess, tf_guess)

        # Bounds
        lb, ub = self.bounds(x_init)

        # Constraint function
        def eq_constraint(z):
            return self.constraints(z, x_init, deck_state, constraints)

        # Number of equality constraints
        n_dynamics = (self.N + 1) * self.nx
        n_init = self.nx
        n_terminal = 0
        if constraints.match_position:
            n_terminal += 3
        if constraints.match_velocity:
            n_terminal += 3
        if constraints.match_roll:
            n_terminal += 1
        if constraints.match_pitch:
            n_terminal += 1
        if constraints.zero_thrust:
            n_terminal += 1

        n_eq = n_dynamics + n_init + n_terminal

        if verbose:
            print(f"Pseudospectral OCP: {N} nodes, {len(z0)} vars, {n_eq} eq constraints")

        # Solve NLP using SLSQP (fast, handles equality constraints well)
        eq_cons = {'type': 'eq', 'fun': eq_constraint}

        result = minimize(
            self.objective,
            z0,
            method='SLSQP',
            bounds=list(zip(lb, ub)),
            constraints=eq_cons,
            options={'maxiter': 500, 'ftol': 1e-6, 'disp': verbose}
        )

        # Check constraint satisfaction even if optimizer reports failure
        final_constraint_violation = np.linalg.norm(eq_constraint(result.x))
        converged = final_constraint_violation < 1e-3

        # Unpack solution
        X, U, tf = self.unpack_decision_vars(result.x)

        # Convert tau to physical time
        t = (self.tau + 1) / 2 * tf

        # Compute terminal errors
        x_final = X[-1]
        pos_error = x_final[0:3] - deck_state[0:3]
        vel_error = x_final[3:6] - deck_state[3:6]
        att_error = x_final[6:9] - deck_state[6:9]

        return {
            'success': converged,
            'message': result.message,
            'constraint_violation': final_constraint_violation,
            't': t,
            'X': X,
            'U': U,
            'tf': tf,
            'pos_error': pos_error,
            'vel_error': vel_error,
            'att_error': att_error,
            'final_thrust': U[-1, 0],
            'cost': result.fun
        }


def solve_landing(quad_pos: np.ndarray,
                  quad_vel: np.ndarray,
                  deck_pos: np.ndarray,
                  deck_vel: np.ndarray,
                  deck_att: np.ndarray = None,
                  tf_guess: float = 5.0,
                  N: int = 20) -> dict:
    """
    High-level interface for landing trajectory optimization.

    Args:
        quad_pos: Quadrotor position [x, y, z] NED (m)
        quad_vel: Quadrotor velocity [vx, vy, vz] NED (m/s)
        deck_pos: Target deck position [x, y, z] NED (m)
        deck_vel: Target deck velocity [vx, vy, vz] NED (m/s)
        deck_att: Target deck attitude [roll, pitch, yaw] (rad), default [0,0,0]
        tf_guess: Initial guess for landing time (s)
        N: Number of collocation nodes

    Returns:
        Solution dictionary with optimal trajectory
    """
    if deck_att is None:
        deck_att = np.zeros(3)

    # Construct full states
    x_init = np.concatenate([
        quad_pos,
        quad_vel,
        np.zeros(3),  # Initial attitude (level)
        np.zeros(3)   # Initial angular rates
    ])

    deck_state = np.concatenate([
        deck_pos,
        deck_vel,
        deck_att
    ])

    # Constraints: match position, velocity, roll/pitch, zero thrust
    constraints = LandingConstraints(
        match_position=True,
        match_velocity=True,
        match_roll=True,
        match_pitch=True,
        zero_thrust=True
    )

    # Solve
    solver = PseudospectralSolver(N=N)
    return solver.solve(x_init, deck_state, constraints, tf_guess)


def demo():
    """Demonstrate pseudospectral landing solver."""
    print("Pseudospectral Optimal Landing Demo")
    print("=" * 50)

    # Scenario: Quad approaching ship deck
    quad_pos = np.array([-30, 5, -25])   # 30m behind, 5m right, 25m up
    quad_vel = np.array([8, -1, 2])      # Approaching, slight drift

    # Deck state (ship moving)
    deck_pos = np.array([0, 0, -8])      # Deck 8m above waterline
    deck_vel = np.array([7.7, 0, 0.2])   # Ship at 15 kts + heave
    deck_att = np.array([0.05, 0.02, 0]) # 3° roll, 1° pitch

    print("Initial conditions:")
    print(f"  Quad position: {quad_pos} m")
    print(f"  Quad velocity: {quad_vel} m/s")
    print(f"  Deck position: {deck_pos} m")
    print(f"  Deck velocity: {deck_vel} m/s")
    print(f"  Deck attitude: roll={np.degrees(deck_att[0]):.1f}°, pitch={np.degrees(deck_att[1]):.1f}°")
    print()

    # Solve
    solution = solve_landing(
        quad_pos, quad_vel,
        deck_pos, deck_vel, deck_att,
        tf_guess=5.0,
        N=25
    )

    print(f"\nSolution {'found' if solution['success'] else 'FAILED'}!")
    print(f"Final time: {solution['tf']:.2f} s")
    print(f"Cost: {solution['cost']:.3f}")

    print(f"\nTerminal errors:")
    print(f"  Position: {np.linalg.norm(solution['pos_error']):.4f} m")
    print(f"  Velocity: {np.linalg.norm(solution['vel_error']):.4f} m/s")
    print(f"  Attitude: roll={np.degrees(solution['att_error'][0]):.2f}°, pitch={np.degrees(solution['att_error'][1]):.2f}°")
    print(f"  Final thrust: {solution['final_thrust']:.3f} N")

    # Trajectory summary
    X = solution['X']
    U = solution['U']
    t = solution['t']

    print(f"\nTrajectory summary:")
    print(f"  Duration: {t[-1]:.2f} s")
    print(f"  Initial altitude: {-X[0, 2]:.1f} m")
    print(f"  Final altitude: {-X[-1, 2]:.1f} m (target: {-deck_pos[2]:.1f} m)")
    print(f"  Max thrust: {np.max(U[:, 0]):.1f} N")
    print(f"  Final descent rate: {X[-1, 5]:.2f} m/s (target: {deck_vel[2]:.2f} m/s)")

    # Check thrust profile
    print(f"\nThrust profile:")
    for i in [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]:
        print(f"  t={t[i]:.2f}s: T={U[i, 0]:.1f}N, alt={-X[i, 2]:.1f}m")

    print("\nDemo complete.")
    return solution


if __name__ == "__main__":
    demo()
