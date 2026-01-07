"""
DDG Ship Motion Model for Shipboard Landing

Simulates 6-DOF motion of DDG-type destroyer in sea states 3-5.
Uses spectral methods (Pierson-Moskowitz) for wave generation
and strip theory approximations for ship response.

Reference frames:
- NED: North-East-Down inertial frame
- Body: Ship body frame (x forward, y starboard, z down)
- Deck: Flight deck location in body frame
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import scipy.signal as signal


@dataclass
class DDGParams:
    """DDG-51 Arleigh Burke class approximate parameters."""
    # Hull dimensions
    length: float = 154.0  # m (505 ft)
    beam: float = 20.0     # m (66 ft)
    draft: float = 6.3     # m (20.5 ft)
    displacement: float = 8300.0  # metric tons

    # Mass properties (approximate)
    mass: float = 8.3e6    # kg
    Ixx: float = 1.5e9     # kg·m² (roll)
    Iyy: float = 8.0e9     # kg·m² (pitch)
    Izz: float = 8.0e9     # kg·m² (yaw)

    # Hydrostatic properties
    GM_t: float = 1.5      # m - transverse metacentric height
    GM_l: float = 150.0    # m - longitudinal metacentric height

    # Flight deck location (aft, centerline, above waterline)
    deck_x: float = -60.0  # m from midship (negative = aft)
    deck_y: float = 0.0    # m from centerline
    deck_z: float = -8.0   # m above waterline (negative = up in body NED)

    # Natural periods (approximate for DDG in moderate loading)
    T_roll: float = 12.0   # s - roll natural period
    T_pitch: float = 6.0   # s - pitch natural period
    T_heave: float = 7.0   # s - heave natural period

    # Damping ratios
    zeta_roll: float = 0.05
    zeta_pitch: float = 0.10
    zeta_heave: float = 0.15


@dataclass
class SeaState:
    """Sea state parameters."""
    number: int           # Sea state number (3, 4, or 5)
    Hs: float            # Significant wave height (m)
    Tp: float            # Peak period (s)
    direction: float     # Wave direction relative to ship heading (deg, 0=head seas)

    @classmethod
    def from_state_number(cls, state: int, direction: float = 0.0) -> 'SeaState':
        """Create sea state from state number (3-5)."""
        params = {
            3: (0.875, 7.5),    # Hs=0.5-1.25m, avg=0.875m
            4: (1.875, 9.0),    # Hs=1.25-2.5m, avg=1.875m
            5: (3.25, 11.0),    # Hs=2.5-4.0m, avg=3.25m
        }
        if state not in params:
            raise ValueError(f"Sea state must be 3, 4, or 5, got {state}")
        Hs, Tp = params[state]
        return cls(number=state, Hs=Hs, Tp=Tp, direction=direction)


class WaveSpectrum:
    """Pierson-Moskowitz wave spectrum for fully developed seas."""

    def __init__(self, Hs: float, Tp: float):
        """
        Args:
            Hs: Significant wave height (m)
            Tp: Peak period (s)
        """
        self.Hs = Hs
        self.Tp = Tp
        self.wp = 2 * np.pi / Tp  # Peak frequency (rad/s)

    def spectrum(self, omega: np.ndarray) -> np.ndarray:
        """
        Pierson-Moskowitz spectrum S(ω).

        S(ω) = (5/16) * Hs² * wp⁴ * ω⁻⁵ * exp(-5/4 * (wp/ω)⁴)

        Args:
            omega: Frequencies (rad/s)

        Returns:
            Spectral density (m²·s/rad)
        """
        omega = np.maximum(omega, 1e-6)  # Avoid division by zero
        wp = self.wp
        Hs = self.Hs

        S = (5/16) * Hs**2 * wp**4 * omega**(-5) * np.exp(-1.25 * (wp/omega)**4)
        return S

    def generate_wave_components(self, n_components: int = 50,
                                  omega_min: float = 0.2,
                                  omega_max: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate wave components for time-domain simulation.

        Returns:
            omega: Component frequencies (rad/s)
            amplitude: Component amplitudes (m)
            phase: Random phases (rad)
        """
        omega = np.linspace(omega_min, omega_max, n_components)
        d_omega = omega[1] - omega[0]

        # Amplitude from spectrum: a = sqrt(2 * S(ω) * dω)
        S = self.spectrum(omega)
        amplitude = np.sqrt(2 * S * d_omega)

        # Random phase
        phase = np.random.uniform(0, 2*np.pi, n_components)

        return omega, amplitude, phase


class DDGMotionSimulator:
    """
    6-DOF ship motion simulator using linear superposition.

    Simulates roll, pitch, heave and derived surge, sway, yaw
    based on wave excitation and ship response characteristics.
    """

    def __init__(self, ship: DDGParams, sea_state: SeaState, ship_speed_kts: float = 15.0):
        """
        Args:
            ship: Ship parameters
            sea_state: Sea state definition
            ship_speed_kts: Ship speed in knots
        """
        self.ship = ship
        self.sea_state = sea_state
        self.ship_speed = ship_speed_kts * 0.5144  # Convert to m/s

        # Wave spectrum
        self.spectrum = WaveSpectrum(sea_state.Hs, sea_state.Tp)

        # Generate wave components
        self.omega, self.wave_amp, self.wave_phase = self.spectrum.generate_wave_components()

        # Encounter frequency adjustment for ship speed
        # ωe = ω - ω²V/g * cos(μ)  for head seas (μ=0), ωe > ω
        g = 9.81
        mu = np.radians(sea_state.direction)
        self.omega_e = self.omega - (self.omega**2 * self.ship_speed / g) * np.cos(mu)
        self.omega_e = np.maximum(self.omega_e, 0.1)  # Prevent negative frequencies

        # Ship response RAOs (simplified linear)
        self._compute_raos()

        # State: [x, y, z, phi, theta, psi, u, v, w, p, q, r]
        # x,y,z = NED position, phi,theta,psi = roll,pitch,yaw
        # u,v,w = body velocities, p,q,r = body angular rates
        self.state = np.zeros(12)
        self.state[0:3] = [0, 0, 0]  # Start at origin

    def _compute_raos(self):
        """Compute Response Amplitude Operators for ship motions."""
        ship = self.ship
        omega = self.omega_e

        # Natural frequencies
        wn_roll = 2 * np.pi / ship.T_roll
        wn_pitch = 2 * np.pi / ship.T_pitch
        wn_heave = 2 * np.pi / ship.T_heave

        # RAOs modeled as second-order systems
        # H(ω) = 1 / sqrt((1 - (ω/ωn)²)² + (2ζω/ωn)²)

        def rao_2nd_order(omega, wn, zeta, gain=1.0):
            r = omega / wn
            # Clamp resonance to prevent extreme values
            denom = np.sqrt((1 - r**2)**2 + (2*zeta*r)**2)
            denom = np.maximum(denom, 0.1)  # Prevent blow-up at resonance
            H = gain / denom
            H = np.minimum(H, gain * 10)  # Cap at 10x gain
            phase = -np.arctan2(2*zeta*r, 1 - r**2)
            return H, phase

        # Roll RAO (most significant for beam/quartering seas)
        # Typical DDG roll RAO peaks at ~5-8 deg/m wave height
        mu = np.radians(self.sea_state.direction)
        roll_excitation = np.abs(np.sin(mu))  # Max at beam seas
        self.rao_roll, self.phase_roll = rao_2nd_order(
            omega, wn_roll, ship.zeta_roll, gain=roll_excitation * 0.08  # rad/m
        )

        # Pitch RAO (significant for head/following seas)
        # Typical DDG pitch RAO peaks at ~1-2 deg/m wave height
        pitch_excitation = np.abs(np.cos(mu))
        self.rao_pitch, self.phase_pitch = rao_2nd_order(
            omega, wn_pitch, ship.zeta_pitch, gain=pitch_excitation * 0.03  # rad/m
        )

        # Heave RAO (approaches 1 at low frequency, <1 at high frequency)
        self.rao_heave, self.phase_heave = rao_2nd_order(
            omega, wn_heave, ship.zeta_heave, gain=0.8  # m/m
        )

    def get_motion(self, t: float) -> dict:
        """
        Get ship motion at time t.

        Returns dict with:
            position: [x, y, z] NED position (m)
            velocity: [vx, vy, vz] NED velocity (m/s)
            attitude: [roll, pitch, yaw] (rad)
            angular_rate: [p, q, r] body rates (rad/s)
            deck_position: [x, y, z] flight deck NED position (m)
            deck_velocity: [vx, vy, vz] flight deck NED velocity (m/s)
        """
        # Compute motions as superposition of wave components
        omega_e = self.omega_e
        amp = self.wave_amp
        phase = self.wave_phase

        # Roll (rotation about x)
        roll = np.sum(amp * self.rao_roll * np.sin(omega_e * t + phase + self.phase_roll))
        roll_rate = np.sum(amp * self.rao_roll * omega_e * np.cos(omega_e * t + phase + self.phase_roll))

        # Pitch (rotation about y)
        pitch = np.sum(amp * self.rao_pitch * np.sin(omega_e * t + phase + self.phase_pitch))
        pitch_rate = np.sum(amp * self.rao_pitch * omega_e * np.cos(omega_e * t + phase + self.phase_pitch))

        # Heave (translation in z)
        heave = np.sum(amp * self.rao_heave * np.sin(omega_e * t + phase + self.phase_heave))
        heave_rate = np.sum(amp * self.rao_heave * omega_e * np.cos(omega_e * t + phase + self.phase_heave))

        # Surge (ship forward motion - approximately constant speed + wave-induced)
        surge = self.ship_speed * t
        surge_rate = self.ship_speed

        # Sway and yaw (simplified - induced by roll/waves)
        sway = 0.3 * roll * self.ship.beam  # Approximate coupling
        sway_rate = 0.3 * roll_rate * self.ship.beam
        yaw = 0.0  # Assume course keeping
        yaw_rate = 0.0

        # Ship center position in NED
        position = np.array([surge, sway, heave])
        velocity = np.array([surge_rate, sway_rate, heave_rate])
        attitude = np.array([roll, pitch, yaw])
        angular_rate = np.array([roll_rate, pitch_rate, yaw_rate])

        # Flight deck position (transform from body to NED)
        deck_body = np.array([self.ship.deck_x, self.ship.deck_y, self.ship.deck_z])

        # Rotation matrix (simplified small angle)
        R = self._rotation_matrix(roll, pitch, yaw)

        # Deck position = ship position + R @ deck_body
        deck_position = position + R @ deck_body

        # Deck velocity = ship velocity + omega × (R @ deck_body)
        omega_body = np.array([roll_rate, pitch_rate, yaw_rate])
        deck_velocity = velocity + np.cross(omega_body, R @ deck_body)

        return {
            'time': t,
            'position': position,
            'velocity': velocity,
            'attitude': attitude,
            'angular_rate': angular_rate,
            'deck_position': deck_position,
            'deck_velocity': deck_velocity,
            'roll_deg': np.degrees(roll),
            'pitch_deg': np.degrees(pitch),
        }

    def _rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Compute rotation matrix from body to NED frame."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
            [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
            [-sp,   sr*cp,            cr*cp]
        ])
        return R

    def get_motion_history(self, t_start: float, t_end: float, dt: float = 0.1) -> dict:
        """Get motion history over time range."""
        times = np.arange(t_start, t_end, dt)
        history = {
            'time': times,
            'roll': np.zeros_like(times),
            'pitch': np.zeros_like(times),
            'heave': np.zeros_like(times),
            'deck_x': np.zeros_like(times),
            'deck_y': np.zeros_like(times),
            'deck_z': np.zeros_like(times),
            'deck_vx': np.zeros_like(times),
            'deck_vy': np.zeros_like(times),
            'deck_vz': np.zeros_like(times),
        }

        for i, t in enumerate(times):
            motion = self.get_motion(t)
            history['roll'][i] = motion['attitude'][0]
            history['pitch'][i] = motion['attitude'][1]
            history['heave'][i] = motion['position'][2]
            history['deck_x'][i] = motion['deck_position'][0]
            history['deck_y'][i] = motion['deck_position'][1]
            history['deck_z'][i] = motion['deck_position'][2]
            history['deck_vx'][i] = motion['deck_velocity'][0]
            history['deck_vy'][i] = motion['deck_velocity'][1]
            history['deck_vz'][i] = motion['deck_velocity'][2]

        return history


class ARMAPredictor:
    """
    ARMA-based predictor for ship deck motion.

    Uses past observations to predict future deck state.
    Fast enough for real-time trajectory planning.
    """

    def __init__(self, ar_order: int = 4, ma_order: int = 2):
        """
        Args:
            ar_order: Auto-regressive order (p)
            ma_order: Moving-average order (q)
        """
        self.ar_order = ar_order
        self.ma_order = ma_order

        # Coefficients (will be fitted)
        self.ar_coeffs = None  # φ
        self.ma_coeffs = None  # θ

        # History buffers
        self.history = []
        self.residuals = []

        # Fitted flag
        self.fitted = False

    def fit(self, data: np.ndarray, dt: float):
        """
        Fit ARMA model to historical data.

        Args:
            data: Time series data (n_samples, n_features)
            dt: Sample interval (s)
        """
        self.dt = dt
        n_samples, n_features = data.shape
        self.n_features = n_features

        # Normalize data for numerical stability
        self.data_mean = np.mean(data, axis=0)
        self.data_std = np.std(data, axis=0)
        self.data_std[self.data_std < 1e-6] = 1.0  # Prevent division by zero
        data_norm = (data - self.data_mean) / self.data_std

        # Fit AR model using regularized least squares for each feature
        self.ar_coeffs = np.zeros((n_features, self.ar_order))
        self.ma_coeffs = np.zeros((n_features, self.ma_order))

        for f in range(n_features):
            # Simple AR fit using least squares with regularization
            y = data_norm[self.ar_order:, f]
            X = np.zeros((len(y), self.ar_order))
            for i in range(self.ar_order):
                X[:, i] = data_norm[self.ar_order - 1 - i:-1 - i, f]

            # Ridge regression for stability
            ridge_lambda = 0.01
            XtX = X.T @ X + ridge_lambda * np.eye(self.ar_order)
            Xty = X.T @ y
            self.ar_coeffs[f] = np.linalg.solve(XtX, Xty)

            # Clamp coefficients for stability (sum of abs < 1 for stationarity)
            coeff_sum = np.sum(np.abs(self.ar_coeffs[f]))
            if coeff_sum > 0.95:
                self.ar_coeffs[f] *= 0.95 / coeff_sum

            # Compute residuals for MA estimation
            y_pred = X @ self.ar_coeffs[f]
            residuals = y - y_pred

            # Simple MA fit with regularization
            if len(residuals) > self.ma_order + 10:
                R = np.zeros((len(residuals) - self.ma_order, self.ma_order))
                for i in range(self.ma_order):
                    R[:, i] = residuals[self.ma_order - 1 - i:-1 - i]
                y_ma = residuals[self.ma_order:]
                RtR = R.T @ R + ridge_lambda * np.eye(self.ma_order)
                Rty = R.T @ y_ma
                self.ma_coeffs[f] = np.linalg.solve(RtR, Rty)

                # Clamp MA coefficients
                ma_sum = np.sum(np.abs(self.ma_coeffs[f]))
                if ma_sum > 0.9:
                    self.ma_coeffs[f] *= 0.9 / ma_sum

        # Store recent history for prediction (normalized)
        self.history = list(data_norm[-self.ar_order:])
        self.residuals = [np.zeros(n_features)] * self.ma_order

        self.fitted = True

    def update(self, observation: np.ndarray):
        """
        Update predictor with new observation.

        Args:
            observation: New measurement (n_features,)
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Normalize observation
        obs_norm = (observation - self.data_mean) / self.data_std

        # Compute prediction error (residual) in normalized space
        pred_norm = self._predict_one_step_normalized()
        residual = obs_norm - pred_norm

        # Update history (normalized)
        self.history.append(obs_norm)
        if len(self.history) > self.ar_order:
            self.history.pop(0)

        # Update residuals
        self.residuals.append(residual)
        if len(self.residuals) > self.ma_order:
            self.residuals.pop(0)

    def _predict_one_step_normalized(self) -> np.ndarray:
        """Predict one step ahead in normalized space."""
        pred = np.zeros(self.n_features)

        for f in range(self.n_features):
            # AR component
            for i, coeff in enumerate(self.ar_coeffs[f]):
                if i < len(self.history):
                    pred[f] += coeff * self.history[-(i+1)][f]

            # MA component
            for i, coeff in enumerate(self.ma_coeffs[f]):
                if i < len(self.residuals):
                    pred[f] += coeff * self.residuals[-(i+1)][f]

        return pred

    def _predict_one_step(self) -> np.ndarray:
        """Predict one step ahead (denormalized)."""
        pred_norm = self._predict_one_step_normalized()
        return pred_norm * self.data_std + self.data_mean

    def predict(self, horizon: int) -> np.ndarray:
        """
        Predict multiple steps ahead.

        Args:
            horizon: Number of steps to predict

        Returns:
            predictions: (horizon, n_features) predicted values (denormalized)
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        predictions_norm = np.zeros((horizon, self.n_features))

        # Temporary copies for multi-step prediction (in normalized space)
        temp_history = list(self.history)
        temp_residuals = list(self.residuals)

        for k in range(horizon):
            pred = np.zeros(self.n_features)

            for f in range(self.n_features):
                # AR component
                for i, coeff in enumerate(self.ar_coeffs[f]):
                    if i < len(temp_history):
                        pred[f] += coeff * temp_history[-(i+1)][f]

                # MA component (decays for multi-step - assume zero residuals for future)
                for i, coeff in enumerate(self.ma_coeffs[f]):
                    if i < len(temp_residuals) and k <= i:
                        pred[f] += coeff * temp_residuals[-(i+1)][f]

            predictions_norm[k] = pred

            # Update temporary history with prediction
            temp_history.append(pred)
            if len(temp_history) > self.ar_order:
                temp_history.pop(0)
            temp_residuals.append(np.zeros(self.n_features))
            if len(temp_residuals) > self.ma_order:
                temp_residuals.pop(0)

        # Denormalize predictions
        predictions = predictions_norm * self.data_std + self.data_mean
        return predictions

    def predict_with_covariance(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation.

        Returns:
            mean: (horizon, n_features) predicted values
            covariance: (horizon, n_features, n_features) prediction covariance
        """
        mean = self.predict(horizon)

        # Uncertainty grows with prediction horizon
        # Simple model: variance grows linearly
        base_var = 0.01  # Base variance from residuals
        covariance = np.zeros((horizon, self.n_features, self.n_features))

        for k in range(horizon):
            # Variance grows with k
            var = base_var * (1 + 0.1 * k)**2
            covariance[k] = np.eye(self.n_features) * var

        return mean, covariance


def demo():
    """Demonstrate ship motion simulation and prediction."""
    print("DDG Ship Motion Simulation Demo")
    print("=" * 50)

    # Create ship and sea state
    ship = DDGParams()
    sea_state = SeaState.from_state_number(4, direction=45)  # SS4, quartering seas

    print(f"Ship: DDG-51 class, {ship.length}m x {ship.beam}m")
    print(f"Sea State: {sea_state.number}, Hs={sea_state.Hs}m, Tp={sea_state.Tp}s")
    print(f"Wave direction: {sea_state.direction}° (0=head seas)")
    print(f"Ship speed: 15 knots")
    print()

    # Create simulator
    sim = DDGMotionSimulator(ship, sea_state, ship_speed_kts=15.0)

    # Get motion at several times
    print("Ship Motion Samples:")
    print("-" * 50)
    for t in [0, 5, 10, 15, 20]:
        motion = sim.get_motion(t)
        print(f"t={t:3.0f}s: Roll={motion['roll_deg']:+5.1f}°, "
              f"Pitch={motion['pitch_deg']:+5.1f}°, "
              f"Deck Z={motion['deck_position'][2]:+5.2f}m")

    # Generate history for ARMA fitting
    print("\nFitting ARMA predictor...")
    history = sim.get_motion_history(0, 60, dt=0.1)

    # Prepare data for ARMA (deck z position and velocity)
    data = np.column_stack([
        history['deck_z'],
        history['deck_vz'],
        history['roll'],
        history['pitch']
    ])

    # Fit ARMA
    predictor = ARMAPredictor(ar_order=6, ma_order=3)
    predictor.fit(data, dt=0.1)

    # Predict ahead
    print("\nPrediction vs Actual (5 seconds ahead):")
    print("-" * 50)

    t_test = 60.0

    # Update predictor with recent data
    for t in np.arange(55, 60, 0.1):
        motion = sim.get_motion(t)
        obs = np.array([
            motion['deck_position'][2],
            motion['deck_velocity'][2],
            motion['attitude'][0],
            motion['attitude'][1]
        ])
        predictor.update(obs)

    # Predict 5 seconds ahead (50 steps at 0.1s)
    predictions = predictor.predict(horizon=50)

    # Compare with actual
    for k in [10, 30, 50]:  # 1s, 3s, 5s ahead
        t_pred = t_test + k * 0.1
        actual = sim.get_motion(t_pred)

        print(f"t+{k*0.1:.1f}s: Deck Z pred={predictions[k-1, 0]:+5.2f}m, "
              f"actual={actual['deck_position'][2]:+5.2f}m, "
              f"error={predictions[k-1, 0] - actual['deck_position'][2]:+5.3f}m")

    print("\nDemo complete.")


if __name__ == "__main__":
    demo()
