#!/usr/bin/env python3
"""
Simple terminal demo runner for PMP shipboard landing.
Uses the full landing_sim infrastructure with terminal output.

Usage:
    python3 run_demo.py [--sea-state N]
"""

import numpy as np
import sys
import time
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.landing_sim import LandingSimulator, LandingConfig


def print_status(t, state, deck, phase, landing_time=None):
    """Print current status to terminal."""
    # Clear screen and move cursor to top
    print("\033[2J\033[H", end="")

    print("=" * 70)
    print(f"  PMP SHIPBOARD LANDING - t={t:.1f}s  Phase: {phase}")
    print("=" * 70)

    if landing_time:
        print(f"  Landing in: {max(0, landing_time - t):.1f}s")

    # Position
    print("\n  POSITION (m)")
    print("  " + "-" * 50)
    print(f"  {'':12} {'X':>10} {'Y':>10} {'Z (alt)':>10}")
    print(f"  {'UAV:':12} {state.pos[0]:>10.2f} {state.pos[1]:>10.2f} {-state.pos[2]:>10.2f}")
    print(f"  {'Deck:':12} {deck['position'][0]:>10.2f} {deck['position'][1]:>10.2f} {-deck['position'][2]:>10.2f}")

    pos_err = state.pos - deck['position']
    print(f"  {'Error:':12} {pos_err[0]:>10.2f} {pos_err[1]:>10.2f} {pos_err[2]:>10.2f}")

    # Velocity
    print("\n  VELOCITY (m/s)")
    print("  " + "-" * 50)
    print(f"  {'UAV:':12} {state.vel[0]:>10.2f} {state.vel[1]:>10.2f} {-state.vel[2]:>10.2f}")
    print(f"  {'Deck:':12} {deck['velocity'][0]:>10.2f} {deck['velocity'][1]:>10.2f} {-deck['velocity'][2]:>10.2f}")

    vel_err = state.vel - deck['velocity']
    print(f"  {'Error:':12} {vel_err[0]:>10.2f} {vel_err[1]:>10.2f} {vel_err[2]:>10.2f}")

    # Deck attitude
    print("\n  DECK ATTITUDE")
    print("  " + "-" * 50)
    roll_deg = np.degrees(deck['attitude'][0])
    pitch_deg = np.degrees(deck['attitude'][1])
    print(f"  Roll: {roll_deg:>6.1f} deg   Pitch: {pitch_deg:>6.1f} deg")

    # Error metrics
    pos_err_mag = np.linalg.norm(pos_err)
    vel_err_mag = np.linalg.norm(vel_err)
    print(f"\n  Total Position Error: {pos_err_mag:.3f} m")
    print(f"  Total Velocity Error: {vel_err_mag:.3f} m/s")

    # ASCII side view
    print("\n  SIDE VIEW")
    draw_side_view(state.pos, deck['position'], deck['attitude'][1])

    print("=" * 70)
    sys.stdout.flush()


def draw_side_view(uav_pos, deck_pos, pitch):
    """Draw simple ASCII side view."""
    width = 50
    height = 10

    view = [[' ' for _ in range(width)] for _ in range(height)]

    # Ocean
    for x in range(width):
        view[height-1][x] = '~'

    # Scale: use relative positions
    rel_x = uav_pos[0] - deck_pos[0]  # Forward distance from deck
    rel_z = uav_pos[2] - deck_pos[2]  # Altitude above deck (negative is above)

    # Deck at center
    deck_vx = width // 2
    deck_vz = height - 3

    # Draw ship
    for dx in range(-6, 7):
        sx = deck_vx + dx
        if 0 <= sx < width:
            view[deck_vz][sx] = '='
    if 0 <= deck_vx < width:
        view[deck_vz][deck_vx] = 'H'

    # UAV position (scale: 5m per char horizontal, 3m per char vertical)
    uav_vx = deck_vx + int(rel_x / 5)
    uav_vz = deck_vz + int(rel_z / 3)  # rel_z is negative when above

    uav_vx = max(0, min(width-1, uav_vx))
    uav_vz = max(0, min(height-2, uav_vz))

    if 0 <= uav_vx < width and 0 <= uav_vz < height:
        view[uav_vz][uav_vx] = '*'
        if uav_vx > 0:
            view[uav_vz][uav_vx-1] = 'o'
        if uav_vx < width-1:
            view[uav_vz][uav_vx+1] = 'o'

    for row in view:
        print("  " + ''.join(row))


def run_demo(sea_state=4, duration=30.0):
    """Run landing demo with terminal output."""

    print("Initializing simulation...")

    # Configure simulation
    config = LandingConfig(
        sea_state=sea_state,
        ship_speed_kts=12.0,
        wave_direction=30.0,
        approach_altitude=25.0,
        approach_distance=40.0
    )

    # Create simulator
    sim = LandingSimulator(config, use_pmp=True)
    sim.reset()

    # Run simulation
    dt = 0.02
    t = 0
    last_display = -1

    log = {'t': [], 'pos_err': [], 'vel_err': []}

    print("Starting simulation...")
    print(f"  Sea state: {sea_state}")
    print(f"  Duration: {duration}s")
    time.sleep(1)

    try:
        while t < duration and not sim.landed:
            # Step simulation
            sim.step(dt)
            t = sim.t

            # Get current states
            deck = sim.ship_sim.get_motion(t)
            state = sim.quad_state

            # Log
            pos_err = np.linalg.norm(state.pos - deck['deck_position'])
            vel_err = np.linalg.norm(state.vel - deck['deck_velocity'])
            log['t'].append(t)
            log['pos_err'].append(pos_err)
            log['vel_err'].append(vel_err)

            # Update display at ~10 Hz
            if t - last_display >= 0.1:
                phase = "APPROACH"
                if sim.current_trajectory is not None:
                    phase = "LANDING"
                if sim.landed:
                    phase = "LANDED"

                print_status(t, state,
                           {'position': deck['deck_position'],
                            'velocity': deck['deck_velocity'],
                            'attitude': deck['attitude']},
                           phase,
                           None)
                last_display = t
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

    # Final results
    print("\n" + "=" * 70)
    print("  LANDING RESULTS")
    print("=" * 70)

    if len(log['pos_err']) > 50:
        final_pos_err = np.mean(log['pos_err'][-50:])
        final_vel_err = np.mean(log['vel_err'][-50:])

        print(f"\n  Final Position Error: {final_pos_err:.3f} m")
        print(f"  Final Velocity Error: {final_vel_err:.3f} m/s")

        if final_pos_err < 1.0 and final_vel_err < 0.5:
            print("\n  *** SUCCESSFUL LANDING ***")
        else:
            print("\n  --- LANDING NEEDS TUNING ---")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PMP Landing Demo')
    parser.add_argument('--sea-state', type=int, default=4,
                        choices=[3, 4, 5], help='Sea state (3-5)')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Max simulation duration')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  PMP SHIPBOARD LANDING DEMONSTRATION")
    print("  Matching deck displacement & velocity at touchdown")
    print("=" * 70)
    print(f"\n  Sea State: {args.sea_state}")
    print("  Press Ctrl+C to stop\n")
    time.sleep(2)

    run_demo(sea_state=args.sea_state, duration=args.duration)
