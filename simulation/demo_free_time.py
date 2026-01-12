#!/usr/bin/env python3
"""
Demo: Free Final Time PMP Landing with Touchdown Constraints

tf is unknown and determined by constraints:
- |deck_roll| <= 5 deg
- |deck_pitch| <= 5 deg
- deck moving down (heave velocity > 0 in NED)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ship_motion.ddg_motion import DDGParams, SeaState, DDGMotionSimulator
from quad_dynamics.quadrotor import QuadrotorParams, QuadrotorState
from optimal_control.pseudospectral import PseudospectralSolver


def demo_free_time_landing(sea_state_num: int = 4):
    """
    Demonstrate free-time optimal landing.

    The solver finds the optimal tf that satisfies deck constraints:
    - |roll| <= 5 deg
    - |pitch| <= 5 deg
    - deck moving DOWN (heave vel > 0 in NED)
    """
    print("=" * 70)
    print("  FREE FINAL TIME PMP LANDING")
    print("  tf determined by touchdown constraints")
    print("=" * 70)

    # Ship motion setup
    sea_state = SeaState.from_state_number(sea_state_num, direction=30.0)
    ship_params = DDGParams()
    ship_sim = DDGMotionSimulator(ship_params, sea_state)

    print(f"\nSea State: {sea_state_num}")
    print(f"  Sig wave height: {sea_state.Hs:.1f} m")
    print(f"  Wave direction: {sea_state.direction:.0f} deg")

    # DDGMotionSimulator uses analytical wave superposition
    # We can query any time directly with get_motion(t)
    t_current = 5.0  # Start at t=5s

    # Get current ship state
    motion = ship_sim.get_motion(t_current)
    deck_pos = motion['deck_position']

    print(f"\nShip state at t={t_current:.1f}s:")
    print(f"  Deck position: [{deck_pos[0]:.1f}, {deck_pos[1]:.1f}, {deck_pos[2]:.1f}] m")
    print(f"  Roll: {np.degrees(motion['attitude'][0]):.2f} deg")
    print(f"  Pitch: {np.degrees(motion['attitude'][1]):.2f} deg")

    # Quad initial state - approach from behind and above
    quad_pos = deck_pos + np.array([-40, 5, -25])  # 40m behind, 5m right, 25m above
    quad_vel = np.array([6, 0, 0])  # Approaching at 6 m/s

    print(f"\nQuad initial state:")
    print(f"  Position: [{quad_pos[0]:.1f}, {quad_pos[1]:.1f}, {quad_pos[2]:.1f}] m")
    print(f"  Velocity: [{quad_vel[0]:.1f}, {quad_vel[1]:.1f}, {quad_vel[2]:.1f}] m/s")

    # Full initial state
    x_init = np.concatenate([
        quad_pos,
        quad_vel,
        np.zeros(3),  # Level attitude
        np.zeros(3)   # Zero angular rates
    ])

    # Create deck motion function - can query any time directly
    def deck_motion_fn(t):
        """Get deck state at time t."""
        return ship_sim.get_motion(t)

    t_init = t_current

    # Touchdown constraints
    touchdown_constraints = {
        'max_roll_deg': 5.0,
        'max_pitch_deg': 5.0,
        'require_descending': True
    }

    print("\nTouchdown constraints:")
    print(f"  |roll| <= {touchdown_constraints['max_roll_deg']:.0f} deg")
    print(f"  |pitch| <= {touchdown_constraints['max_pitch_deg']:.0f} deg")
    print(f"  deck moving down: {touchdown_constraints['require_descending']}")

    # Scan for valid landing windows
    print("\n" + "-" * 70)
    print("Scanning for valid landing windows...")
    print("-" * 70)

    tf_min, tf_max = 3.0, 15.0
    valid_windows = []

    for tf_scan in np.arange(tf_min, tf_max, 0.5):
        deck = deck_motion_fn(t_init + tf_scan)
        roll = np.degrees(deck['attitude'][0])
        pitch = np.degrees(deck['attitude'][1])
        heave_vel = deck['deck_velocity'][2]

        roll_ok = abs(roll) <= 5.0
        pitch_ok = abs(pitch) <= 5.0
        descend_ok = heave_vel >= 0

        status = "VALID" if (roll_ok and pitch_ok and descend_ok) else "invalid"
        if status == "VALID":
            valid_windows.append(tf_scan)

        print(f"  tf={tf_scan:5.1f}s: roll={roll:+5.1f}° pitch={pitch:+5.1f}° heave_vel={heave_vel:+5.2f} m/s  [{status}]")

    if valid_windows:
        print(f"\nFound {len(valid_windows)} valid landing windows")
    else:
        print("\nNo valid windows found - solver will find best compromise")

    # Solve with free final time
    print("\n" + "=" * 70)
    print("Solving PMP with free final time...")
    print("=" * 70)

    solver = PseudospectralSolver(N=25)
    solution = solver.solve_free_time(
        x_init=x_init,
        deck_motion_fn=deck_motion_fn,
        t_current=t_init,
        touchdown_constraints=touchdown_constraints,
        tf_bounds=(tf_min, tf_max),
        verbose=True
    )

    # Results
    print("\n" + "=" * 70)
    print("SOLUTION SUMMARY")
    print("=" * 70)

    if solution['success']:
        print("\n*** OPTIMAL LANDING FOUND ***")
    else:
        print("\n--- Solver converged with constraints ---")

    print(f"\nOptimal landing time: tf = {solution['tf']:.2f}s")
    print(f"Landing at t = {solution['t_landing']:.2f}s")

    print(f"\nDeck state at landing:")
    print(f"  Roll:  {np.degrees(solution['deck_att_at_landing'][0]):+5.2f}° (limit ±5°)")
    print(f"  Pitch: {np.degrees(solution['deck_att_at_landing'][1]):+5.2f}° (limit ±5°)")
    print(f"  Heave velocity: {solution['deck_vel_at_landing'][2]:+5.2f} m/s {'(deck descending)' if solution['deck_vel_at_landing'][2] >= 0 else '(deck ASCENDING)'}")

    print(f"\nTerminal errors:")
    print(f"  Position: {np.linalg.norm(solution['pos_error']):.4f} m")
    print(f"  Velocity: {np.linalg.norm(solution['vel_error']):.4f} m/s")
    print(f"  Final thrust: {solution['final_thrust']:.3f} N")

    print(f"\nCost: {solution['cost']:.3f}")

    # Trajectory summary
    X = solution['X']
    t = solution['t']

    print("\nTrajectory profile:")
    print(f"  {'Time':>6}  {'Alt':>8}  {'Dist':>8}  {'Vz':>8}")
    for i in [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]:
        alt = -X[i, 2]
        dist = np.sqrt(X[i, 0]**2 + X[i, 1]**2)
        vz = X[i, 5]
        print(f"  {t[i]:6.2f}s  {alt:8.1f}m  {dist:8.1f}m  {vz:+8.2f}m/s")

    print("\n" + "=" * 70)
    return solution


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sea-state', type=int, default=4, choices=[3, 4, 5, 6])
    args = parser.parse_args()

    demo_free_time_landing(sea_state_num=args.sea_state)
