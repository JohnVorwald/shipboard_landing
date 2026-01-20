"""
Realistic Ship Wave Motion Model

Implements 6-DOF ship motion response to ocean waves based on:
- Sea State parameters (significant wave height, modal period)
- Ship response characteristics (natural frequencies, damping)
- Realistic phase relationships between motion modes

Reference: Standard NATO sea state definitions and ship motion theory.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class SeaState:
    """Sea state parameters."""
    significant_wave_height: float  # Hs in meters
    modal_period: float  # Tp in seconds (peak period)

    # NATO Sea State reference values:
    # SS3: Hs=0.5-1.25m, Tp=5-7s (slight)
    # SS4: Hs=1.25-2.5m, Tp=7-10s (moderate)
    # SS5: Hs=2.5-4.0m, Tp=10-13s (rough)


@dataclass
class ShipCharacteristics:
    """Ship response characteristics."""
    length: float = 50.0  # meters
    beam: float = 12.0  # meters

    # Natural periods (typical for 50m vessel)
    roll_period: float = 8.0  # seconds
    pitch_period: float = 5.0  # seconds
    heave_period: float = 6.0  # seconds

    # Damping ratios
    roll_damping: float = 0.1
    pitch_damping: float = 0.15
    heave_damping: float = 0.2


# Standard sea states
SEA_STATE_3 = SeaState(0.875, 6.0)   # Slight
SEA_STATE_4 = SeaState(1.875, 8.5)   # Moderate
SEA_STATE_5 = SeaState(3.25, 11.5)   # Rough


class ShipWaveMotion:
    """
    Realistic 6-DOF ship motion model responding to ocean waves.

    Motion is computed as a superposition of wave-induced motions
    with appropriate phase relationships and frequency content.
    """

    def __init__(self,
                 sea_state: SeaState = SEA_STATE_4,
                 ship: ShipCharacteristics = None,
                 forward_speed: float = 2.5,  # m/s (~5 knots)
                 heading: float = 0.0):  # radians (0 = into waves)

        self.sea_state = sea_state
        self.ship = ship or ShipCharacteristics()
        self.forward_speed = forward_speed
        self.heading = heading

        # Wave encounter frequency increases with forward speed
        self.encounter_factor = 1.0 + forward_speed * 0.1

        # Compute motion amplitudes based on sea state
        self._compute_motion_amplitudes()

        # Random phase offsets for each motion (for realistic randomness)
        np.random.seed(42)  # Reproducible for testing
        self.phases = {
            'heave': np.random.uniform(0, 2*np.pi),
            'roll': np.random.uniform(0, 2*np.pi),
            'pitch': np.random.uniform(0, 2*np.pi),
            'surge': np.random.uniform(0, 2*np.pi),
            'sway': np.random.uniform(0, 2*np.pi),
            'yaw': np.random.uniform(0, 2*np.pi),
        }

        # Component frequencies (multiple wave components)
        self.n_components = 5
        self.wave_freqs = []
        self.wave_amps = []
        self._generate_wave_spectrum()

    def _compute_motion_amplitudes(self):
        """Compute motion amplitudes from sea state and ship characteristics."""
        Hs = self.sea_state.significant_wave_height
        Tp = self.sea_state.modal_period

        # Motion amplitudes scale with wave height
        # These are approximate RAOs (Response Amplitude Operators)

        # Heave: ships typically have heave amplitude ~0.5-0.8 of wave height
        self.heave_amp = Hs * 0.6

        # Roll: most significant motion, depends on beam and wave period
        # Roll amplitude in radians, typically 5-15 degrees in moderate seas
        roll_factor = 1.0 - abs(self.ship.roll_period - Tp) / max(self.ship.roll_period, Tp)
        self.roll_amp = math.radians(8.0) * (Hs / 2.0) * (0.5 + 0.5 * roll_factor)

        # Pitch: typically smaller than roll, depends on length
        pitch_factor = 1.0 - abs(self.ship.pitch_period - Tp) / max(self.ship.pitch_period, Tp)
        self.pitch_amp = math.radians(4.0) * (Hs / 2.0) * (0.5 + 0.5 * pitch_factor)

        # Surge: horizontal motion in direction of travel
        self.surge_amp = Hs * 0.2

        # Sway: horizontal motion perpendicular to travel
        self.sway_amp = Hs * 0.15

        # Yaw: rotational motion about vertical axis
        self.yaw_amp = math.radians(2.0) * (Hs / 2.0)

    def _generate_wave_spectrum(self):
        """Generate wave components based on JONSWAP/Pierson-Moskowitz spectrum."""
        Tp = self.sea_state.modal_period
        omega_p = 2 * np.pi / Tp  # Peak frequency

        # Generate frequencies around peak
        for i in range(self.n_components):
            # Spread frequencies around peak
            factor = 0.7 + 0.6 * i / (self.n_components - 1)
            omega = omega_p * factor
            self.wave_freqs.append(omega)

            # Amplitude based on simplified spectrum shape
            # Peak at modal frequency, decreasing away from it
            rel_freq = omega / omega_p
            if rel_freq <= 1:
                amp = rel_freq ** 5
            else:
                amp = math.exp(-1.25 * (rel_freq - 1) ** 2)
            self.wave_amps.append(amp)

        # Normalize amplitudes
        total = sum(self.wave_amps)
        self.wave_amps = [a / total for a in self.wave_amps]

    def _compute_motion(self, t: float, base_amp: float, natural_period: float,
                       damping: float, phase: float) -> Tuple[float, float]:
        """
        Compute motion and velocity for one degree of freedom.

        Returns displacement and velocity.
        """
        omega_n = 2 * np.pi / natural_period

        displacement = 0.0
        velocity = 0.0

        for i, (omega, amp_factor) in enumerate(zip(self.wave_freqs, self.wave_amps)):
            # Encounter frequency adjustment
            omega_e = omega * self.encounter_factor

            # Phase for this component
            phi = phase + i * 0.7  # Spread phases for realistic randomness

            # RAO-like response (simplified transfer function)
            freq_ratio = omega_e / omega_n
            response = 1.0 / math.sqrt((1 - freq_ratio**2)**2 + (2*damping*freq_ratio)**2)
            response = min(response, 3.0)  # Limit resonance

            # Motion
            amp = base_amp * amp_factor * response
            displacement += amp * math.sin(omega_e * t + phi)
            velocity += amp * omega_e * math.cos(omega_e * t + phi)

        return displacement, velocity

    def get_motion(self, t: float, initial_pos: np.ndarray) -> dict:
        """
        Get ship position, orientation, and velocities at time t.

        Args:
            t: Time in seconds
            initial_pos: Initial ship position [x, y, z]

        Returns:
            Dictionary with position, orientation, and velocities
        """
        # Forward motion
        x = initial_pos[0] + self.forward_speed * t
        y = initial_pos[1]
        z = initial_pos[2]

        # Surge (forward/back oscillation)
        surge, surge_vel = self._compute_motion(
            t, self.surge_amp, self.sea_state.modal_period, 0.2, self.phases['surge'])

        # Sway (side to side)
        sway, sway_vel = self._compute_motion(
            t, self.sway_amp, self.sea_state.modal_period * 1.1, 0.2, self.phases['sway'])

        # Heave (up/down)
        heave, heave_vel = self._compute_motion(
            t, self.heave_amp, self.ship.heave_period, self.ship.heave_damping, self.phases['heave'])

        # Roll (rotation about forward axis)
        roll, roll_vel = self._compute_motion(
            t, self.roll_amp, self.ship.roll_period, self.ship.roll_damping, self.phases['roll'])

        # Pitch (rotation about side axis)
        pitch, pitch_vel = self._compute_motion(
            t, self.pitch_amp, self.ship.pitch_period, self.ship.pitch_damping, self.phases['pitch'])

        # Yaw (rotation about vertical axis)
        yaw, yaw_vel = self._compute_motion(
            t, self.yaw_amp, self.sea_state.modal_period * 1.5, 0.3, self.phases['yaw'])

        # Combine motions
        position = np.array([
            x + surge,
            y + sway,
            z + heave
        ])

        # Linear velocities
        linear_velocity = np.array([
            self.forward_speed + surge_vel,
            sway_vel,
            heave_vel
        ])

        # Angular positions (Euler angles)
        orientation = np.array([roll, pitch, yaw])

        # Angular velocities
        angular_velocity = np.array([roll_vel, pitch_vel, yaw_vel])

        return {
            'position': position,
            'orientation': orientation,  # [roll, pitch, yaw] in radians
            'linear_velocity': linear_velocity,
            'angular_velocity': angular_velocity,
            'heave': heave,
            'roll_deg': math.degrees(roll),
            'pitch_deg': math.degrees(pitch),
        }

    def get_helipad_state(self, t: float, initial_ship_pos: np.ndarray,
                         helipad_offset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get helipad position and velocity accounting for ship motion.

        The helipad moves with the ship and is also affected by roll/pitch.

        Args:
            t: Time in seconds
            initial_ship_pos: Initial ship position [x, y, z]
            helipad_offset: Offset of helipad from ship origin [x, y, z]

        Returns:
            helipad_position, helipad_velocity (both numpy arrays)
        """
        motion = self.get_motion(t, initial_ship_pos)

        # Get ship state
        ship_pos = motion['position']
        roll = motion['orientation'][0]
        pitch = motion['orientation'][1]
        ship_vel = motion['linear_velocity']
        roll_vel = motion['angular_velocity'][0]
        pitch_vel = motion['angular_velocity'][1]

        # Transform helipad offset by ship rotation
        # Simplified rotation (small angles)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)

        # Rotation matrix (Rz ignored for simplicity since yaw is small)
        # R = Ry(pitch) * Rx(roll)
        offset_rotated = np.array([
            helipad_offset[0] * cp + helipad_offset[2] * sp,
            helipad_offset[1] * cr - helipad_offset[2] * sr * cp + helipad_offset[0] * sr * sp,
            -helipad_offset[0] * sp + helipad_offset[1] * sr + helipad_offset[2] * cr * cp
        ])

        helipad_pos = ship_pos + offset_rotated

        # Helipad velocity includes ship linear velocity plus angular contribution
        # v_helipad = v_ship + omega x r
        omega = motion['angular_velocity']
        r = offset_rotated
        angular_contribution = np.cross(omega, r)

        helipad_vel = ship_vel + angular_contribution

        return helipad_pos, helipad_vel


def demo():
    """Demonstrate ship motion model."""
    print("Ship Wave Motion Demo")
    print("=" * 60)

    # Create motion model with Sea State 4
    motion_model = ShipWaveMotion(
        sea_state=SEA_STATE_4,
        forward_speed=2.5  # ~5 knots
    )

    print(f"Sea State: Hs={motion_model.sea_state.significant_wave_height}m, "
          f"Tp={motion_model.sea_state.modal_period}s")
    print(f"Forward speed: {motion_model.forward_speed}m/s")
    print(f"Motion amplitudes:")
    print(f"  Heave: {motion_model.heave_amp:.2f}m")
    print(f"  Roll: {math.degrees(motion_model.roll_amp):.1f}°")
    print(f"  Pitch: {math.degrees(motion_model.pitch_amp):.1f}°")
    print()

    initial_pos = np.array([0.0, 0.0, 0.0])
    helipad_offset = np.array([15.0, 0.0, 4.0])

    print("Time     Ship X    Heave   Roll°  Pitch°  Helipad Z")
    print("-" * 60)

    for t in np.arange(0, 20, 1.0):
        state = motion_model.get_motion(t, initial_pos)
        helipad_pos, helipad_vel = motion_model.get_helipad_state(t, initial_pos, helipad_offset)

        print(f"{t:5.1f}s  {state['position'][0]:7.1f}m  "
              f"{state['heave']:+5.2f}m  {state['roll_deg']:+5.1f}  {state['pitch_deg']:+5.1f}  "
              f"{helipad_pos[2]:5.2f}m")


if __name__ == '__main__':
    demo()
