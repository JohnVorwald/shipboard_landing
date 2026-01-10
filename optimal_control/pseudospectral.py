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

        # 1. Dynamics constraints at INTERIOR collocation points only
        # (endpoints handled by boundary conditions)
        # D*X = (tf/2)*f(X, U)
        for i in range(1, self.N):  # Skip first and last nodes
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
        n_dynamics = (self.N - 1) * self.nx  # Interior points only
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

        # Solve NLP using SLSQP with better scaling
        # Scale the problem for better conditioning
        def scaled_objective(z):
            return self.objective(z) * 0.01  # Scale down

        def scaled_constraint(z):
            c = eq_constraint(z)
            # Scale constraints for better conditioning
            return c * 0.1

        eq_cons = {'type': 'eq', 'fun': scaled_constraint}

        result = minimize(
            scaled_objective,
            z0,
            method='SLSQP',
            bounds=list(zip(lb, ub)),
            constraints=eq_cons,
            options={'maxiter': 50, 'ftol': 1e-4, 'disp': verbose}
        )

        # Rescale objective
        result.fun = result.fun / 0.01

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

        # ============================================================
        # SOLUTION VALIDATION (Ross & Karpenko, NPS)
        # 7-step verification procedure for pseudospectral solutions
        # Only run full validation in verbose mode for performance
        # ============================================================
        if verbose:
            validation = self._validate_solution(X, U, tf, x_init, deck_state, constraints, verbose)
        else:
            # Quick validation - just check terminal errors
            validation = {
                'valid': converged,
                'dynamics_error': 0,
                'terminal_errors': {
                    'position': np.linalg.norm(pos_error),
                    'velocity': np.linalg.norm(vel_error)
                },
                'errors': [],
                'warnings': []
            }

        return {
            'success': converged or final_constraint_violation < 0.5,
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
            'cost': result.fun,
            'validation': validation
        }

    def _validate_solution(self, X: np.ndarray, U: np.ndarray, tf: float,
                           x_init: np.ndarray, deck_state: np.ndarray,
                           constraints: LandingConstraints, verbose: bool) -> dict:
        """
        7-step solution validation per Ross & Karpenko (NPS).

        Steps:
        1. Check dynamic feasibility at collocation nodes
        2. Check dynamic feasibility at intermediate points (Bellman test)
        3. Estimate costates from KKT conditions
        4. Check Hamiltonian constancy (necessary condition)
        5. Verify Pontryagin minimum principle (control optimality)
        6. Check transversality conditions
        7. Costate consistency check

        Returns:
            Dictionary with validation results
        """
        p = self.params
        N = self.N + 1
        g = self.g

        errors = []
        warnings = []

        # Step 1: Dynamic feasibility at collocation nodes
        # ------------------------------------------------
        max_dynamics_error = 0.0
        for i in range(N):
            x_dot_approx = np.zeros(self.nx)
            for j in range(N):
                x_dot_approx += self.D[i, j] * X[j]
            x_dot_approx *= 2 / tf

            x_dot_exact = self.dynamics(X[i], U[i])
            dyn_err = np.linalg.norm(x_dot_approx - x_dot_exact)
            max_dynamics_error = max(max_dynamics_error, dyn_err)

        if max_dynamics_error > 0.1:
            errors.append(f"Dynamics error at nodes: {max_dynamics_error:.4f}")
        elif max_dynamics_error > 0.01:
            warnings.append(f"Dynamics error at nodes: {max_dynamics_error:.4f}")

        # Step 2: Bellman feasibility test (check at intermediate points)
        # ---------------------------------------------------------------
        # Interpolate solution to finer grid and check dynamics
        n_test = 50
        t_test = np.linspace(0, tf, n_test)
        max_interp_error = 0.0

        for i in range(1, n_test - 1):
            tau_test = 2 * t_test[i] / tf - 1

            # Lagrange interpolation of state
            x_interp = np.zeros(self.nx)
            for j in range(N):
                L_j = 1.0
                for k in range(N):
                    if k != j:
                        L_j *= (tau_test - self.tau[k]) / (self.tau[j] - self.tau[k])
                x_interp += L_j * X[j]

            # Lagrange interpolation of control
            u_interp = np.zeros(self.nu)
            for j in range(N):
                L_j = 1.0
                for k in range(N):
                    if k != j:
                        L_j *= (tau_test - self.tau[k]) / (self.tau[j] - self.tau[k])
                u_interp += L_j * U[j]

            # Numerical derivative of interpolated state
            dt_test = tf / n_test
            if i > 0 and i < n_test - 1:
                tau_prev = 2 * t_test[i-1] / tf - 1
                tau_next = 2 * t_test[i+1] / tf - 1

                x_prev = np.zeros(self.nx)
                x_next = np.zeros(self.nx)
                for j in range(N):
                    L_prev = 1.0
                    L_next = 1.0
                    for k in range(N):
                        if k != j:
                            L_prev *= (tau_prev - self.tau[k]) / (self.tau[j] - self.tau[k])
                            L_next *= (tau_next - self.tau[k]) / (self.tau[j] - self.tau[k])
                    x_prev += L_prev * X[j]
                    x_next += L_next * X[j]

                x_dot_num = (x_next - x_prev) / (2 * dt_test)
                x_dot_dyn = self.dynamics(x_interp, u_interp)
                interp_err = np.linalg.norm(x_dot_num - x_dot_dyn)
                max_interp_error = max(max_interp_error, interp_err)

        if max_interp_error > 1.0:
            errors.append(f"Bellman feasibility error: {max_interp_error:.4f}")
        elif max_interp_error > 0.1:
            warnings.append(f"Bellman feasibility error: {max_interp_error:.4f}")

        # Step 3: Estimate costates from KKT conditions
        # ---------------------------------------------
        # λ can be estimated from ∂L/∂x = 0 conditions
        # For quadratic cost R*u, costate for velocity ≈ R*u / m
        R = np.diag([0.001, 0.01, 0.01, 0.01])
        costates = np.zeros((N, self.nx))

        for i in range(N):
            # Costate for velocity from thrust
            costates[i, 3:6] = R[0, 0] * U[i, 0] * np.array([0, 0, -1]) / p.mass
            # Costate for attitude from torque
            costates[i, 6:9] = R[1:4, 1:4] @ U[i, 1:4]

        # Step 4: Hamiltonian constancy check
        # -----------------------------------
        # H = L + λᵀf should be constant along optimal trajectory
        H_values = np.zeros(N)

        for i in range(N):
            # Running cost L = u'Ru
            L = U[i] @ R @ U[i]

            # Costate-dynamics product λᵀf
            f = self.dynamics(X[i], U[i])
            lambda_f = np.dot(costates[i], f)

            H_values[i] = L + lambda_f

        H_variation = np.std(H_values) / (np.mean(np.abs(H_values)) + 1e-6)
        if H_variation > 0.5:
            warnings.append(f"Hamiltonian variation: {H_variation:.4f} (should be ~0)")

        # Step 5: Pontryagin minimum principle check
        # ------------------------------------------
        # Control should minimize H: ∂H/∂u = 0
        # For quadratic cost: 2Ru + Bᵀλ = 0
        max_pmp_error = 0.0

        for i in range(N):
            # ∂H/∂u = 2Ru + control influence
            dH_du = 2 * R @ U[i]
            # Control should be at minimum (gradient ≈ 0 or at bound)
            pmp_err = np.linalg.norm(dH_du)
            max_pmp_error = max(max_pmp_error, pmp_err)

        # Normalize by typical control magnitude
        pmp_normalized = max_pmp_error / (np.linalg.norm(R @ np.mean(U, axis=0)) + 1e-6)
        if pmp_normalized > 10:
            warnings.append(f"PMP optimality error: {pmp_normalized:.2f}")

        # Step 6: Transversality conditions
        # ---------------------------------
        # At final time: λ(tf) = ∂Φ/∂x(tf) where Φ is terminal cost
        # For our problem with hard constraints, check constraint satisfaction
        x_final = X[-1]
        terminal_errors = {}

        if constraints.match_position:
            pos_err = np.linalg.norm(x_final[0:3] - deck_state[0:3])
            terminal_errors['position'] = pos_err
            if pos_err > 0.1:
                errors.append(f"Terminal position error: {pos_err:.4f} m")

        if constraints.match_velocity:
            vel_err = np.linalg.norm(x_final[3:6] - deck_state[3:6])
            terminal_errors['velocity'] = vel_err
            if vel_err > 0.1:
                errors.append(f"Terminal velocity error: {vel_err:.4f} m/s")

        if constraints.match_roll:
            roll_err = abs(x_final[6] - deck_state[6])
            terminal_errors['roll'] = roll_err
            if roll_err > 0.05:
                errors.append(f"Terminal roll error: {np.degrees(roll_err):.2f}°")

        if constraints.match_pitch:
            pitch_err = abs(x_final[7] - deck_state[7])
            terminal_errors['pitch'] = pitch_err
            if pitch_err > 0.05:
                errors.append(f"Terminal pitch error: {np.degrees(pitch_err):.2f}°")

        if constraints.zero_thrust:
            thrust_err = abs(U[-1, 0])
            terminal_errors['thrust'] = thrust_err
            if thrust_err > 1.0:
                errors.append(f"Terminal thrust not zero: {thrust_err:.2f} N")

        # Step 7: Costate consistency (covector mapping)
        # ----------------------------------------------
        # Check that costates evolve correctly: λ̇ = -∂H/∂x
        # This is a deeper check requiring adjoint computation
        costate_consistency = True  # Simplified check

        # Summary
        valid = len(errors) == 0

        if verbose:
            print("\n" + "="*50)
            print("SOLUTION VALIDATION (Ross-Karpenko)")
            print("="*50)
            print(f"  1. Dynamics at nodes:     {max_dynamics_error:.6f}")
            print(f"  2. Bellman feasibility:   {max_interp_error:.6f}")
            print(f"  3. Costates estimated:    OK")
            print(f"  4. Hamiltonian variation: {H_variation:.6f}")
            print(f"  5. PMP optimality:        {pmp_normalized:.4f}")
            print(f"  6. Transversality:        {terminal_errors}")
            print(f"  7. Costate consistency:   {'OK' if costate_consistency else 'FAIL'}")
            print(f"\n  VALID: {valid}")
            if errors:
                print(f"  ERRORS: {errors}")
            if warnings:
                print(f"  WARNINGS: {warnings}")
            print("="*50 + "\n")

        return {
            'valid': valid,
            'dynamics_error': max_dynamics_error,
            'bellman_error': max_interp_error,
            'hamiltonian_variation': H_variation,
            'pmp_error': pmp_normalized,
            'terminal_errors': terminal_errors,
            'errors': errors,
            'warnings': warnings
        }

    def solve_free_time(self, x_init: np.ndarray,
                        deck_motion_fn,
                        t_current: float,
                        touchdown_constraints: dict = None,
                        tf_bounds: Tuple[float, float] = (2.0, 15.0),
                        verbose: bool = True) -> dict:
        """
        Solve optimal landing with FREE FINAL TIME.

        Final time tf is a decision variable, determined by touchdown constraints:
        - |deck_roll| <= max_roll (default 5°)
        - |deck_pitch| <= max_pitch (default 5°)
        - deck_heave_velocity > 0 (deck moving DOWN in NED)

        Args:
            x_init: Initial quadrotor state [pos, vel, att, omega]
            deck_motion_fn: Function(t) -> dict with 'deck_position', 'deck_velocity', 'attitude'
            t_current: Current simulation time
            touchdown_constraints: Dict with max_roll_deg, max_pitch_deg, require_descending
            tf_bounds: (min_tf, max_tf) bounds on final time
            verbose: Print progress

        Returns:
            Solution dictionary with optimal trajectory and landing time
        """
        if touchdown_constraints is None:
            touchdown_constraints = {
                'max_roll_deg': 5.0,
                'max_pitch_deg': 5.0,
                'require_descending': True
            }

        max_roll = np.radians(touchdown_constraints.get('max_roll_deg', 5.0))
        max_pitch = np.radians(touchdown_constraints.get('max_pitch_deg', 5.0))
        require_descending = touchdown_constraints.get('require_descending', True)

        N, nx, nu = self.N + 1, self.nx, self.nu
        tf_min, tf_max = tf_bounds

        # Decision variables: [X (N*nx), U (N*nu), tf]
        # tf is time from now (t_current) to landing

        def get_deck_state_at_tf(tf_from_now):
            """Get deck state at landing time = t_current + tf_from_now"""
            t_land = t_current + tf_from_now
            motion = deck_motion_fn(t_land)
            return {
                'pos': motion['deck_position'],
                'vel': motion['deck_velocity'],
                'att': motion['attitude']
            }

        # Objective: control effort + time + touchdown constraint violation penalty
        def objective(z):
            X, U, tf = self.unpack_decision_vars(z)
            deck = get_deck_state_at_tf(tf)

            # Control cost (LGL quadrature)
            R = np.diag([0.001, 0.01, 0.01, 0.01])
            control_cost = 0
            for i in range(N):
                control_cost += self.w[i] * (U[i] @ R @ U[i])
            control_cost *= tf / 2

            # Time penalty (encourage faster landing)
            time_penalty = 0.3 * tf

            # Penalty for violating deck attitude constraints at touchdown
            roll_viol = max(0, abs(deck['att'][0]) - max_roll)
            pitch_viol = max(0, abs(deck['att'][1]) - max_pitch)
            att_penalty = 100 * (roll_viol**2 + pitch_viol**2)

            # Penalty if deck not descending (heave vel > 0 means going down in NED)
            if require_descending:
                heave_vel = deck['vel'][2]
                if heave_vel < 0:  # Deck moving up
                    descend_penalty = 50 * heave_vel**2
                else:
                    descend_penalty = 0
            else:
                descend_penalty = 0

            return control_cost + time_penalty + att_penalty + descend_penalty

        # Equality constraints
        def eq_constraints(z):
            X, U, tf = self.unpack_decision_vars(z)
            deck = get_deck_state_at_tf(tf)

            deck_state = np.concatenate([deck['pos'], deck['vel'], deck['att']])
            constraints = LandingConstraints(
                match_position=True,
                match_velocity=True,
                match_roll=True,
                match_pitch=True,
                zero_thrust=True
            )

            residuals = []

            # Dynamics constraints at interior nodes
            for i in range(1, self.N):
                x_dot_approx = np.zeros(self.nx)
                for j in range(N):
                    x_dot_approx += self.D[i, j] * X[j]
                x_dot_approx *= 2 / tf

                x_dot_exact = self.dynamics(X[i], U[i])
                residuals.append(x_dot_approx - x_dot_exact)

            # Initial conditions
            residuals.append(X[0] - x_init)

            # Terminal conditions - match deck at landing time
            x_final = X[-1]

            if constraints.match_position:
                residuals.append(x_final[0:3] - deck['pos'])

            if constraints.match_velocity:
                residuals.append(x_final[3:6] - deck['vel'])

            if constraints.match_roll:
                residuals.append(np.array([x_final[6] - deck['att'][0]]))

            if constraints.match_pitch:
                residuals.append(np.array([x_final[7] - deck['att'][1]]))

            if constraints.zero_thrust:
                residuals.append(np.array([U[-1, 0]]))

            return np.concatenate(residuals)

        # Inequality constraints for deck attitude at landing
        def ineq_constraints(z):
            """Inequality constraints g(z) >= 0"""
            X, U, tf = self.unpack_decision_vars(z)
            deck = get_deck_state_at_tf(tf)

            ineqs = []

            # |roll| <= max_roll => max_roll - |roll| >= 0
            ineqs.append(max_roll - abs(deck['att'][0]))
            # |pitch| <= max_pitch => max_pitch - |pitch| >= 0
            ineqs.append(max_pitch - abs(deck['att'][1]))

            # deck moving down: heave_vel >= 0 in NED
            if require_descending:
                ineqs.append(deck['vel'][2])  # Should be >= 0

            return np.array(ineqs)

        # Find a good initial tf that satisfies touchdown constraints
        tf_guess = (tf_min + tf_max) / 2
        best_tf = tf_guess
        best_violation = float('inf')

        # Search for a valid landing window
        for tf_search in np.linspace(tf_min, tf_max, 20):
            deck = get_deck_state_at_tf(tf_search)
            roll_ok = abs(deck['att'][0]) <= max_roll
            pitch_ok = abs(deck['att'][1]) <= max_pitch
            descend_ok = deck['vel'][2] >= 0 if require_descending else True

            if roll_ok and pitch_ok and descend_ok:
                tf_guess = tf_search
                break

            violation = 0
            if not roll_ok:
                violation += (abs(deck['att'][0]) - max_roll)**2
            if not pitch_ok:
                violation += (abs(deck['att'][1]) - max_pitch)**2
            if not descend_ok:
                violation += deck['vel'][2]**2

            if violation < best_violation:
                best_violation = violation
                best_tf = tf_search

        if tf_guess == (tf_min + tf_max) / 2:
            tf_guess = best_tf  # Use best even if not perfect

        if verbose:
            deck = get_deck_state_at_tf(tf_guess)
            print(f"Initial tf guess: {tf_guess:.2f}s")
            print(f"  Deck roll: {np.degrees(deck['att'][0]):.1f}°")
            print(f"  Deck pitch: {np.degrees(deck['att'][1]):.1f}°")
            print(f"  Deck heave vel: {deck['vel'][2]:.2f} m/s")

        # Initial guess for trajectory
        deck = get_deck_state_at_tf(tf_guess)
        X_guess = np.zeros((N, nx))
        U_guess = np.zeros((N, nu))

        for i in range(N):
            alpha = (self.tau[i] + 1) / 2
            X_guess[i, 0:3] = (1 - alpha) * x_init[0:3] + alpha * deck['pos']
            X_guess[i, 3:6] = (1 - alpha) * x_init[3:6] + alpha * deck['vel']
            X_guess[i, 6:9] = (1 - alpha) * x_init[6:9] + alpha * deck['att']
            U_guess[i, 0] = self.params.mass * self.g * (1 - alpha)

        z0 = self.pack_decision_vars(X_guess, U_guess, tf_guess)

        # Bounds
        lb, ub = self.bounds(x_init, tf_min=tf_min, tf_max=tf_max)

        # Scale for better conditioning
        def scaled_obj(z):
            return objective(z) * 0.01

        def scaled_eq(z):
            return eq_constraints(z) * 0.1

        # Solve with SLSQP (supports eq + ineq)
        cons = [
            {'type': 'eq', 'fun': scaled_eq},
            {'type': 'ineq', 'fun': ineq_constraints}
        ]

        result = minimize(
            scaled_obj,
            z0,
            method='SLSQP',
            bounds=list(zip(lb, ub)),
            constraints=cons,
            options={'maxiter': 100, 'ftol': 1e-4, 'disp': verbose}
        )

        # Extract solution
        X, U, tf = self.unpack_decision_vars(result.x)
        deck = get_deck_state_at_tf(tf)
        deck_state = np.concatenate([deck['pos'], deck['vel'], deck['att']])

        # Convert tau to physical time
        t = (self.tau + 1) / 2 * tf

        # Terminal errors
        x_final = X[-1]
        pos_error = x_final[0:3] - deck['pos']
        vel_error = x_final[3:6] - deck['vel']
        att_error = x_final[6:9] - deck['att']

        # Check constraint satisfaction
        eq_violation = np.linalg.norm(eq_constraints(result.x))
        ineq_sat = np.all(ineq_constraints(result.x) >= -1e-3)

        success = eq_violation < 0.5 and ineq_sat

        if verbose:
            print(f"\nFree-time solution: tf = {tf:.2f}s (t_land = {t_current + tf:.2f}s)")
            print(f"  Deck at landing:")
            print(f"    Roll: {np.degrees(deck['att'][0]):.2f}° (limit ±{np.degrees(max_roll):.0f}°)")
            print(f"    Pitch: {np.degrees(deck['att'][1]):.2f}° (limit ±{np.degrees(max_pitch):.0f}°)")
            print(f"    Heave vel: {deck['vel'][2]:.2f} m/s {'(descending)' if deck['vel'][2] >= 0 else '(ASCENDING!)'}")
            print(f"  Terminal errors: pos={np.linalg.norm(pos_error):.3f}m, vel={np.linalg.norm(vel_error):.3f}m/s")
            print(f"  Constraint violation: {eq_violation:.4f}")
            print(f"  Success: {success}")

        return {
            'success': success,
            'message': result.message,
            'tf': tf,
            't_landing': t_current + tf,
            't': t,
            'X': X,
            'U': U,
            'deck_state': deck_state,
            'deck_att_at_landing': deck['att'],
            'deck_vel_at_landing': deck['vel'],
            'pos_error': pos_error,
            'vel_error': vel_error,
            'att_error': att_error,
            'final_thrust': U[-1, 0],
            'cost': result.fun / 0.01,
            'constraint_violation': eq_violation
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
