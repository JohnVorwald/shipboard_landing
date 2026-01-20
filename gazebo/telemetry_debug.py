#!/usr/bin/env python3
"""
Telemetry Debugging Script for Ship Landing Simulation

Subscribes to both quadcopter and ship poses, calculates relative
error, and monitors heave prediction accuracy.

Usage:
    python3 telemetry_debug.py [--log FILE]

This script uses gz topic commands since gz-transport Python bindings
are not readily available. For production, consider using the C++ API.
"""

import subprocess
import time
import sys
import os
import re
import math
import argparse
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ship_wave_motion import ShipWaveMotion, SEA_STATE_4

# Gazebo environment
GZ_ENV = os.environ.copy()
GZ_ENV['GZ_IP'] = '127.0.0.1'
GZ_ENV['GZ_PARTITION'] = 'gazebo_default'

WORLD = "ship_landing_debug"
HELIPAD_OFFSET = np.array([15.0, 0.0, 4.0])


@dataclass
class TelemetryFrame:
    """Single telemetry frame for logging."""
    timestamp: float
    sim_time: float
    quad_pos: List[float]
    quad_vel: List[float]
    ship_pos: List[float]
    helipad_pos: List[float]
    predicted_helipad_pos: List[float]
    relative_error: List[float]
    height_above_deck: float
    horizontal_distance: float
    heave_prediction_error: float
    phase: str


class TelemetryMonitor:
    """Real-time telemetry monitoring and logging."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.start_time = time.time()
        self.prev_quad_pos = None
        self.prev_quad_time = None
        self.quad_vel = np.zeros(3)

        # Wave model for heave prediction
        self.wave_model = ShipWaveMotion(sea_state=SEA_STATE_4, forward_speed=2.5)
        self.ship_initial = np.array([0.0, 0.0, 0.0])

        # Statistics
        self.heave_errors = []
        self.position_errors = []
        self.frames = []

        if self.log_file:
            self.log_handle = open(self.log_file, 'w')
            self.log_handle.write("# Telemetry Log - Ship Landing Debug\n")
            self.log_handle.write(f"# Started: {datetime.now().isoformat()}\n")
            self.log_handle.write("# Format: JSON Lines\n\n")
        else:
            self.log_handle = None

    def close(self):
        """Close log file and print statistics."""
        if self.log_handle:
            self.log_handle.close()

        if self.heave_errors:
            print("\n" + "=" * 60)
            print("TELEMETRY STATISTICS")
            print("=" * 60)
            print(f"Frames captured: {len(self.frames)}")
            print(f"Heave prediction error (RMS): {np.sqrt(np.mean(np.square(self.heave_errors))):.3f} m")
            print(f"Heave prediction error (max): {np.max(np.abs(self.heave_errors)):.3f} m")
            if self.position_errors:
                print(f"Position error (final): {self.position_errors[-1]:.3f} m")
                print(f"Position error (min): {np.min(self.position_errors):.3f} m")

    def get_pose_from_topic(self, model_name: str) -> Optional[np.ndarray]:
        """Get model pose from Gazebo topic."""
        try:
            result = subprocess.run(
                ['gz', 'topic', '-e', '-t', f'/world/{WORLD}/dynamic_pose/info', '-n', '1'],
                capture_output=True, text=True, timeout=1.5, env=GZ_ENV
            )
            if result.returncode == 0:
                return self._parse_model_pose(result.stdout, model_name)
        except Exception as e:
            pass
        return None

    def _parse_model_pose(self, output: str, model_name: str) -> Optional[np.ndarray]:
        """Parse pose from Gazebo message."""
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if f'name: "{model_name}"' in line:
                pos = [0.0, 0.0, 0.0]
                in_position = False
                for j in range(i, min(i + 20, len(lines))):
                    pline = lines[j].strip()
                    if pline == 'position {':
                        in_position = True
                    elif pline == '}' and in_position:
                        break
                    elif in_position:
                        for idx, coord in enumerate(['x:', 'y:', 'z:']):
                            if pline.startswith(coord):
                                match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', pline)
                                if match:
                                    pos[idx] = float(match.group())
                return np.array(pos)
        return None

    def estimate_quad_velocity(self, pos: np.ndarray) -> np.ndarray:
        """Estimate quadcopter velocity from position."""
        now = time.time()
        if self.prev_quad_pos is not None and self.prev_quad_time is not None:
            dt = now - self.prev_quad_time
            if 0.01 < dt < 0.5:
                new_vel = (pos - self.prev_quad_pos) / dt
                new_vel = np.clip(new_vel, -50, 50)
                self.quad_vel = 0.3 * new_vel + 0.7 * self.quad_vel
        self.prev_quad_pos = pos.copy()
        self.prev_quad_time = now
        return self.quad_vel

    def get_predicted_helipad(self, sim_time: float) -> tuple:
        """Get predicted helipad position from wave model."""
        return self.wave_model.get_helipad_state(sim_time, self.ship_initial, HELIPAD_OFFSET)

    def determine_phase(self, height: float, horiz_dist: float, vel_z: float) -> str:
        """Determine landing phase."""
        if height > 8.0:
            return "APPROACH"
        elif height > 3.0:
            return "DESCENT"
        elif height > 0.5:
            return "TERMINAL"
        elif horiz_dist < 3.0 and abs(vel_z) < 1.0:
            return "LANDED"
        else:
            return "CONTACT"

    def process_frame(self):
        """Process one telemetry frame."""
        sim_time = time.time() - self.start_time

        # Get actual poses
        quad_pos = self.get_pose_from_topic('quadcopter')
        ship_pos = self.get_pose_from_topic('ship')

        if quad_pos is None:
            return None

        if ship_pos is None:
            ship_pos = np.array([0.0, 0.0, 0.0])

        # Estimate velocity
        quad_vel = self.estimate_quad_velocity(quad_pos)

        # Actual helipad position
        actual_helipad = ship_pos + HELIPAD_OFFSET

        # Predicted helipad position from wave model
        predicted_helipad, _ = self.get_predicted_helipad(sim_time)

        # Errors
        relative_error = quad_pos - actual_helipad
        height = relative_error[2]
        horiz_dist = np.linalg.norm(relative_error[:2])

        # Heave prediction error (most critical for moving deck)
        heave_error = predicted_helipad[2] - actual_helipad[2]
        self.heave_errors.append(heave_error)
        self.position_errors.append(np.linalg.norm(relative_error))

        # Determine phase
        phase = self.determine_phase(height, horiz_dist, quad_vel[2])

        # Create frame
        frame = TelemetryFrame(
            timestamp=time.time(),
            sim_time=sim_time,
            quad_pos=quad_pos.tolist(),
            quad_vel=quad_vel.tolist(),
            ship_pos=ship_pos.tolist(),
            helipad_pos=actual_helipad.tolist(),
            predicted_helipad_pos=predicted_helipad.tolist(),
            relative_error=relative_error.tolist(),
            height_above_deck=height,
            horizontal_distance=horiz_dist,
            heave_prediction_error=heave_error,
            phase=phase
        )

        self.frames.append(frame)

        # Log to file
        if self.log_handle:
            self.log_handle.write(json.dumps(asdict(frame)) + '\n')
            self.log_handle.flush()

        return frame

    def print_frame(self, frame: TelemetryFrame):
        """Print telemetry frame to console."""
        print(f"\r[{frame.sim_time:6.1f}s] {frame.phase:8s} | "
              f"Q:({frame.quad_pos[0]:7.1f},{frame.quad_pos[1]:6.1f},{frame.quad_pos[2]:6.1f}) | "
              f"V:({frame.quad_vel[0]:+5.1f},{frame.quad_vel[1]:+4.1f},{frame.quad_vel[2]:+5.1f}) | "
              f"H:{frame.height_above_deck:6.1f}m D:{frame.horizontal_distance:5.1f}m | "
              f"Heave Err:{frame.heave_prediction_error:+5.2f}m", end='', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Telemetry debugging for ship landing')
    parser.add_argument('--log', type=str, help='Log file path (JSON Lines format)')
    parser.add_argument('--duration', type=float, default=60.0, help='Monitoring duration in seconds')
    parser.add_argument('--rate', type=float, default=10.0, help='Update rate in Hz')
    args = parser.parse_args()

    print("=" * 80)
    print("TELEMETRY DEBUG MONITOR")
    print("=" * 80)
    print(f"World: {WORLD}")
    print(f"Logging to: {args.log or 'console only'}")
    print(f"Duration: {args.duration}s, Rate: {args.rate}Hz")
    print("-" * 80)

    # Check Gazebo
    try:
        result = subprocess.run(['gz', 'topic', '-l'], capture_output=True, text=True,
                               timeout=2, env=GZ_ENV)
        if f'/world/{WORLD}' not in result.stdout:
            print(f"ERROR: World '{WORLD}' not found in Gazebo")
            print("Start Gazebo with: gz sim -s -r worlds/ship_landing_debug.world")
            return 1
    except Exception as e:
        print(f"ERROR: Cannot connect to Gazebo: {e}")
        return 1

    monitor = TelemetryMonitor(log_file=args.log)

    print("\nMonitoring... (Ctrl+C to stop)")
    print("-" * 80)

    dt = 1.0 / args.rate
    start = time.time()

    try:
        while time.time() - start < args.duration:
            loop_start = time.time()

            frame = monitor.process_frame()
            if frame:
                monitor.print_frame(frame)

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    monitor.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
