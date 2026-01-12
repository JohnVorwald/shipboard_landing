#!/usr/bin/env python3
"""Debug script to analyze landing errors."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.landing_sim import LandingSimulator, LandingConfig

def debug_landing():
    """Run simulation with detailed logging."""

    config = LandingConfig(
        sea_state=3,
        ship_speed_kts=12.0,
        wave_direction=30.0,
        approach_altitude=25.0,
        approach_distance=40.0
    )

    sim = LandingSimulator(config, use_pmp=True)
    sim.reset()

    dt = 0.02

    print("=" * 70)
    print("DEBUG: PMP Landing Simulation")
    print("=" * 70)

    # Log key events
    log = {
        't': [],
        'quad_pos': [],
        'quad_vel': [],
        'deck_pos': [],
        'deck_vel': [],
        'traj_pos': [],
        'traj_vel': [],
        'control': [],
        'tracking_err': [],
        'deck_err': []
    }

    print("\nInitial state:")
    print(f"  Quad pos: {sim.quad_state.pos}")
    print(f"  Quad vel: {sim.quad_state.vel}")

    deck = sim.ship_sim.get_motion(0)
    print(f"  Deck pos: {deck['deck_position']}")
    print(f"  Deck vel: {deck['deck_velocity']}")

    print("\n" + "-" * 70)
    print("Simulation log (every 1s):")
    print("-" * 70)

    last_print = -1

    while sim.t < 15 and not sim.landed:
        sim.step(dt)
        t = sim.t

        # Get states
        deck = sim.ship_sim.get_motion(t)
        quad_pos = sim.quad_state.pos.copy()
        quad_vel = sim.quad_state.vel.copy()
        deck_pos = deck['deck_position']
        deck_vel = deck['deck_velocity']

        # Get trajectory target if available
        if sim.current_trajectory is not None and sim.trajectory_start_time is not None:
            t_traj = t - sim.trajectory_start_time
            traj = sim.current_trajectory
            if hasattr(traj, 'pos_coeffs'):
                # TrajectoryResult from min-snap planner
                traj_pos, traj_vel, _ = sim.planner.trajectory.sample(traj, t_traj)
            else:
                traj_pos = quad_pos  # fallback
                traj_vel = quad_vel
        else:
            traj_pos = quad_pos
            traj_vel = quad_vel

        # Log
        log['t'].append(t)
        log['quad_pos'].append(quad_pos)
        log['quad_vel'].append(quad_vel)
        log['deck_pos'].append(deck_pos)
        log['deck_vel'].append(deck_vel)
        log['traj_pos'].append(traj_pos)
        log['traj_vel'].append(traj_vel)
        log['tracking_err'].append(np.linalg.norm(quad_pos - traj_pos))
        log['deck_err'].append(np.linalg.norm(quad_pos - deck_pos))

        # Print every second
        if t - last_print >= 1.0:
            pos_err = quad_pos - deck_pos
            vel_err = quad_vel - deck_vel
            track_err = quad_pos - traj_pos

            print(f"\nt={t:.1f}s:")
            print(f"  Quad:  pos=[{quad_pos[0]:7.2f}, {quad_pos[1]:7.2f}, {quad_pos[2]:7.2f}]  vel=[{quad_vel[0]:6.2f}, {quad_vel[1]:6.2f}, {quad_vel[2]:6.2f}]")
            print(f"  Deck:  pos=[{deck_pos[0]:7.2f}, {deck_pos[1]:7.2f}, {deck_pos[2]:7.2f}]  vel=[{deck_vel[0]:6.2f}, {deck_vel[1]:6.2f}, {deck_vel[2]:6.2f}]")
            print(f"  Traj:  pos=[{traj_pos[0]:7.2f}, {traj_pos[1]:7.2f}, {traj_pos[2]:7.2f}]  vel=[{traj_vel[0]:6.2f}, {traj_vel[1]:6.2f}, {traj_vel[2]:6.2f}]")
            print(f"  Deck error:     {np.linalg.norm(pos_err):7.2f} m,  {np.linalg.norm(vel_err):6.2f} m/s")
            print(f"  Tracking error: {np.linalg.norm(track_err):7.2f} m")

            # Check trajectory target
            if sim.target_deck_state is not None:
                target = sim.target_deck_state
                if hasattr(target, 'position'):
                    print(f"  Target deck (at t={target.t:.1f}s): pos=[{target.position[0]:7.2f}, {target.position[1]:7.2f}, {target.position[2]:7.2f}]")
                    # Compare with where deck will actually be
                    actual_future_deck = sim.ship_sim.get_motion(target.t)['deck_position']
                    pred_err = np.linalg.norm(target.position - actual_future_deck)
                    print(f"  Actual deck at t={target.t:.1f}s:   pos=[{actual_future_deck[0]:7.2f}, {actual_future_deck[1]:7.2f}, {actual_future_deck[2]:7.2f}]")
                    print(f"  PREDICTION ERROR: {pred_err:.2f} m")

            last_print = t

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Analyze errors
    log['t'] = np.array(log['t'])
    log['tracking_err'] = np.array(log['tracking_err'])
    log['deck_err'] = np.array(log['deck_err'])

    print(f"\nTracking error (quad vs trajectory):")
    print(f"  Mean: {np.mean(log['tracking_err']):.2f} m")
    print(f"  Max:  {np.max(log['tracking_err']):.2f} m")
    print(f"  Final: {log['tracking_err'][-1]:.2f} m")

    print(f"\nDeck error (quad vs deck):")
    print(f"  Mean: {np.mean(log['deck_err']):.2f} m")
    print(f"  Max:  {np.max(log['deck_err']):.2f} m")
    print(f"  Final: {log['deck_err'][-1]:.2f} m")

    # Check if trajectory is being followed
    if np.mean(log['tracking_err']) > 5:
        print("\n*** ISSUE: Large tracking error - quad not following trajectory ***")
        print("    Possible causes:")
        print("    - Controller gains too low")
        print("    - Dynamics too aggressive for controller")
        print("    - Control saturation")

    # Check if trajectory targets deck
    quad_final = log['quad_pos'][-1]
    deck_final = log['deck_pos'][-1]
    traj_final = log['traj_pos'][-1]

    traj_to_deck = np.linalg.norm(traj_final - deck_final)
    print(f"\nTrajectory endpoint vs deck: {traj_to_deck:.2f} m")

    if traj_to_deck > 5:
        print("\n*** ISSUE: Trajectory not targeting current deck position ***")
        print("    Possible causes:")
        print("    - Trajectory planned to old deck prediction")
        print("    - Replanning not happening")
        print("    - Target deck state stale")

    # Check velocity matching
    quad_vel_final = log['quad_vel'][-1]
    deck_vel_final = log['deck_vel'][-1]
    vel_err = np.linalg.norm(quad_vel_final - deck_vel_final)
    print(f"\nFinal velocity error: {vel_err:.2f} m/s")

    if vel_err > 2:
        print("\n*** ISSUE: Not matching deck velocity ***")
        print("    Possible causes:")
        print("    - Trajectory not including deck velocity in boundary conditions")
        print("    - Approaching too fast")


if __name__ == "__main__":
    debug_landing()
