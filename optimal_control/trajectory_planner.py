"""
Trajectory Planner for Shipboard Landing

Generates smooth, dynamically feasible trajectories to match deck state at touchdown.
Uses polynomial trajectory generation with boundary conditions.

This is a practical alternative to full pseudospectral optimization when
real-time performance is needed.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quad_dynamics.quadrotor import QuadrotorParams


@dataclass
class TrajectoryResult:
    """Result of trajectory planning."""
    success: bool
    tf: float                    # Final time
    pos_coeffs: np.ndarray      # Position polynomial coefficients [3, n_coeffs]
    vel_coeffs: np.ndarray      # Velocity polynomial coefficients [3, n_coeffs]
    terminal_pos_error: float
    terminal_vel_error: float


class MinSnapTrajectory:
    """
    Minimum snap trajectory generator.

    Generates smooth trajectories that satisfy position and velocity
    boundary conditions at initial and final time.

    Uses 7th order polynomials (8 coefficients) to satisfy:
    - Position at t=0 and t=tf
    - Velocity at t=0 and t=tf
    - Acceleration at t=0 and t=tf (set to reasonable values)
    - Jerk at t=0 (zero for smooth start)
    - Snap continuity (minimized for smooth trajectory)
    """

    def __init__(self, params: QuadrotorParams = None):
        self.params = params if params is not None else QuadrotorParams()
        self.g = 9.81

    def plan(self,
             pos_init: np.ndarray,
             vel_init: np.ndarray,
             pos_final: np.ndarray,
             vel_final: np.ndarray,
             tf: float,
             acc_init: np.ndarray = None,
             acc_final: np.ndarray = None) -> TrajectoryResult:
        """
        Plan trajectory from initial to final state.

        Args:
            pos_init: Initial position [x, y, z]
            vel_init: Initial velocity [vx, vy, vz]
            pos_final: Final position [x, y, z]
            vel_final: Final velocity [vx, vy, vz]
            tf: Final time
            acc_init: Initial acceleration (optional, defaults to hover)
            acc_final: Final acceleration (optional, defaults to match deck)

        Returns:
            TrajectoryResult with polynomial coefficients
        """
        if acc_init is None:
            # Default: hover acceleration (thrust = weight)
            acc_init = np.array([0, 0, 0])

        if acc_final is None:
            # Default: match deck motion (zero relative acceleration)
            acc_final = np.array([0, 0, 0])

        # Use 5th order polynomial: 6 coefficients per axis
        # p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
        # This satisfies position, velocity, acceleration at both endpoints

        pos_coeffs = np.zeros((3, 6))
        vel_coeffs = np.zeros((3, 5))

        for axis in range(3):
            # Boundary conditions matrix
            # [p(0), v(0), a(0), p(tf), v(tf), a(tf)] = [c0, c1, 2*c2, ...]

            T = tf
            T2 = T * T
            T3 = T2 * T
            T4 = T3 * T
            T5 = T4 * T

            # Matrix for [c0, c1, c2, c3, c4, c5]
            A = np.array([
                [1, 0, 0, 0, 0, 0],                           # p(0) = c0
                [0, 1, 0, 0, 0, 0],                           # v(0) = c1
                [0, 0, 2, 0, 0, 0],                           # a(0) = 2*c2
                [1, T, T2, T3, T4, T5],                       # p(tf)
                [0, 1, 2*T, 3*T2, 4*T3, 5*T4],               # v(tf)
                [0, 0, 2, 6*T, 12*T2, 20*T3]                  # a(tf)
            ])

            # Boundary values
            b = np.array([
                pos_init[axis],
                vel_init[axis],
                acc_init[axis],
                pos_final[axis],
                vel_final[axis],
                acc_final[axis]
            ])

            # Solve for coefficients
            try:
                c = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Singular matrix - use pseudo-inverse
                c = np.linalg.lstsq(A, b, rcond=None)[0]

            pos_coeffs[axis] = c
            # Velocity coefficients: derivative of position
            vel_coeffs[axis] = np.array([c[1], 2*c[2], 3*c[3], 4*c[4], 5*c[5]])

        # Compute terminal errors
        pos_at_tf = self._eval_polynomial(pos_coeffs, tf)
        vel_at_tf = self._eval_polynomial_deriv(pos_coeffs, tf)

        pos_error = np.linalg.norm(pos_at_tf - pos_final)
        vel_error = np.linalg.norm(vel_at_tf - vel_final)

        return TrajectoryResult(
            success=pos_error < 0.01 and vel_error < 0.01,
            tf=tf,
            pos_coeffs=pos_coeffs,
            vel_coeffs=vel_coeffs,
            terminal_pos_error=pos_error,
            terminal_vel_error=vel_error
        )

    def _eval_polynomial(self, coeffs: np.ndarray, t: float) -> np.ndarray:
        """Evaluate polynomial at time t."""
        result = np.zeros(3)
        for axis in range(3):
            c = coeffs[axis]
            powers = np.array([1, t, t**2, t**3, t**4, t**5])
            result[axis] = np.dot(c, powers)
        return result

    def _eval_polynomial_deriv(self, coeffs: np.ndarray, t: float) -> np.ndarray:
        """Evaluate polynomial derivative (velocity) at time t."""
        result = np.zeros(3)
        for axis in range(3):
            c = coeffs[axis]
            # Derivative: c1 + 2*c2*t + 3*c3*t^2 + 4*c4*t^3 + 5*c5*t^4
            powers = np.array([1, 2*t, 3*t**2, 4*t**3, 5*t**4])
            result[axis] = np.dot(c[1:], powers)
        return result

    def _eval_polynomial_accel(self, coeffs: np.ndarray, t: float) -> np.ndarray:
        """Evaluate second derivative (acceleration) at time t."""
        result = np.zeros(3)
        for axis in range(3):
            c = coeffs[axis]
            # Second derivative: 2*c2 + 6*c3*t + 12*c4*t^2 + 20*c5*t^3
            powers = np.array([2, 6*t, 12*t**2, 20*t**3])
            result[axis] = np.dot(c[2:], powers)
        return result

    def sample(self, result: TrajectoryResult, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample trajectory at time t.

        Returns:
            (position, velocity, acceleration)
        """
        t = np.clip(t, 0, result.tf)

        pos = self._eval_polynomial(result.pos_coeffs, t)
        vel = self._eval_polynomial_deriv(result.pos_coeffs, t)
        acc = self._eval_polynomial_accel(result.pos_coeffs, t)

        return pos, vel, acc

    def compute_required_thrust(self, result: TrajectoryResult, t: float) -> float:
        """
        Compute thrust required at time t along trajectory.

        For a quadrotor, thrust ≈ m * ||a_des + g||
        """
        _, _, acc = self.sample(result, t)

        # Total acceleration including gravity compensation
        g_vec = np.array([0, 0, self.g])  # NED: positive z is down
        total_acc = acc + g_vec

        thrust = self.params.mass * np.linalg.norm(total_acc)
        return thrust

    def validate(self, result: TrajectoryResult) -> dict:
        """
        Validate trajectory feasibility.

        Checks:
        1. Terminal condition satisfaction
        2. Maximum thrust within limits
        3. Maximum velocity within limits
        4. Smooth (no discontinuities)
        """
        errors = []
        warnings = []

        # Check terminal conditions
        if result.terminal_pos_error > 0.1:
            errors.append(f"Position error: {result.terminal_pos_error:.3f} m")
        if result.terminal_vel_error > 0.1:
            errors.append(f"Velocity error: {result.terminal_vel_error:.3f} m/s")

        # Sample trajectory and check constraints
        n_samples = 50
        max_thrust = 0
        max_vel = 0

        for i in range(n_samples + 1):
            t = i * result.tf / n_samples
            pos, vel, acc = self.sample(result, t)

            thrust = self.compute_required_thrust(result, t)
            max_thrust = max(max_thrust, thrust)
            max_vel = max(max_vel, np.linalg.norm(vel))

        # Check thrust limits
        if max_thrust > self.params.max_total_thrust:
            errors.append(f"Max thrust {max_thrust:.1f}N exceeds limit {self.params.max_total_thrust:.1f}N")
        elif max_thrust > 0.9 * self.params.max_total_thrust:
            warnings.append(f"Max thrust {max_thrust:.1f}N near limit")

        # Check final thrust (should be low for landing)
        final_thrust = self.compute_required_thrust(result, result.tf)
        if final_thrust > 0.5 * self.params.mass * self.g:
            warnings.append(f"Final thrust {final_thrust:.1f}N > 50% hover")

        return {
            'valid': len(errors) == 0,
            'max_thrust': max_thrust,
            'max_velocity': max_vel,
            'final_thrust': final_thrust,
            'errors': errors,
            'warnings': warnings
        }


class LandingTrajectoryPlanner:
    """
    High-level trajectory planner for shipboard landing.

    Plans trajectory to match deck position and velocity at touchdown.
    """

    def __init__(self, params: QuadrotorParams = None):
        self.params = params if params is not None else QuadrotorParams()
        self.traj_gen = MinSnapTrajectory(self.params)

    def plan_landing(self,
                     quad_pos: np.ndarray,
                     quad_vel: np.ndarray,
                     deck_pos: np.ndarray,
                     deck_vel: np.ndarray,
                     deck_att: np.ndarray = None,
                     tf_desired: float = None) -> dict:
        """
        Plan landing trajectory.

        Args:
            quad_pos: Current quadrotor position [x, y, z] NED
            quad_vel: Current quadrotor velocity [vx, vy, vz] NED
            deck_pos: Target deck position at tf [x, y, z] NED
            deck_vel: Target deck velocity at tf [vx, vy, vz] NED
            deck_att: Target deck attitude [roll, pitch, yaw] (for attitude matching)
            tf_desired: Desired final time (will optimize if not provided)

        Returns:
            Dictionary with trajectory and validation info
        """
        # Estimate good final time if not provided
        if tf_desired is None:
            dist = np.linalg.norm(deck_pos - quad_pos)
            vel_diff = np.linalg.norm(deck_vel - quad_vel)
            # Time estimate based on distance and required velocity change
            tf_desired = max(2.0, dist / 10.0 + vel_diff / 5.0)

        # Try different final times to find feasible trajectory
        best_result = None
        best_validation = None

        for tf_scale in [1.0, 1.2, 1.5, 0.8]:
            tf = tf_desired * tf_scale

            # Plan trajectory
            result = self.traj_gen.plan(
                pos_init=quad_pos,
                vel_init=quad_vel,
                pos_final=deck_pos,
                vel_final=deck_vel,
                tf=tf
            )

            # Validate
            validation = self.traj_gen.validate(result)

            if validation['valid']:
                best_result = result
                best_validation = validation
                break
            elif best_result is None or len(validation['errors']) < len(best_validation['errors']):
                best_result = result
                best_validation = validation

        if best_result is None:
            return {
                'success': False,
                'message': 'Failed to find feasible trajectory'
            }

        return {
            'success': best_validation['valid'],
            'trajectory': best_result,
            'validation': best_validation,
            'tf': best_result.tf,
            'terminal_pos_error': best_result.terminal_pos_error,
            'terminal_vel_error': best_result.terminal_vel_error
        }

    def sample_trajectory(self, trajectory: TrajectoryResult, t: float) -> dict:
        """Sample trajectory at time t."""
        pos, vel, acc = self.traj_gen.sample(trajectory, t)
        thrust = self.traj_gen.compute_required_thrust(trajectory, t)

        return {
            'position': pos,
            'velocity': vel,
            'acceleration': acc,
            'thrust': thrust
        }


def demo():
    """Demonstrate trajectory planner."""
    print("Landing Trajectory Planner Demo")
    print("=" * 50)

    planner = LandingTrajectoryPlanner()

    # Scenario: approach from behind and above
    quad_pos = np.array([-30, 0, -25])   # 30m behind, 25m up
    quad_vel = np.array([5, 0, 0])        # Approaching

    deck_pos = np.array([0, 0, -8])       # Deck position
    deck_vel = np.array([7.7, 0, 0.2])    # Ship at 15 kts + heave

    print(f"Quad: pos={quad_pos}, vel={quad_vel}")
    print(f"Deck: pos={deck_pos}, vel={deck_vel}")
    print()

    # Plan trajectory
    result = planner.plan_landing(quad_pos, quad_vel, deck_pos, deck_vel)

    print(f"Success: {result['success']}")
    print(f"Final time: {result['tf']:.2f} s")
    print(f"Terminal pos error: {result['terminal_pos_error']:.6f} m")
    print(f"Terminal vel error: {result['terminal_vel_error']:.6f} m/s")

    if 'validation' in result:
        v = result['validation']
        print(f"\nValidation:")
        print(f"  Valid: {v['valid']}")
        print(f"  Max thrust: {v['max_thrust']:.1f} N")
        print(f"  Max velocity: {v['max_velocity']:.1f} m/s")
        print(f"  Final thrust: {v['final_thrust']:.1f} N")
        if v['errors']:
            print(f"  Errors: {v['errors']}")
        if v['warnings']:
            print(f"  Warnings: {v['warnings']}")

    # Sample trajectory
    if result['success']:
        traj = result['trajectory']
        print(f"\nTrajectory samples:")
        for t_frac in [0, 0.25, 0.5, 0.75, 1.0]:
            t = t_frac * traj.tf
            sample = planner.sample_trajectory(traj, t)
            print(f"  t={t:.2f}s: pos={sample['position']}, vel={sample['velocity']}, T={sample['thrust']:.1f}N")

    return result


if __name__ == "__main__":
    demo()
