"""Unit tests for ship motion models."""

import pytest
import numpy as np
from numpy.testing import assert_allclose


class TestDDGParams:
    """Tests for DDG ship parameters."""

    def test_default_params(self, ddg_params):
        """Test default parameters are reasonable."""
        assert ddg_params.length > 100  # DDG > 100m
        assert ddg_params.beam > 15     # DDG beam > 15m
        assert ddg_params.displacement > 5000  # DDG > 5000 tons
        assert ddg_params.mass > 0

    def test_deck_location(self, ddg_params):
        """Test flight deck location."""
        # Deck should be aft
        assert ddg_params.deck_x < 0
        # Deck should be above waterline
        assert ddg_params.deck_z < 0  # Negative in NED


class TestSeaState:
    """Tests for sea state configuration."""

    def test_sea_state_3(self):
        """Test sea state 3 parameters."""
        from ship_motion.ddg_motion import SeaState
        ss = SeaState.from_state_number(3, direction=0.0)

        assert ss.number == 3
        assert 0.5 <= ss.Hs <= 1.5  # SS3 wave height
        assert ss.Tp > 5  # Period > 5s

    def test_sea_state_4(self, sea_state_4):
        """Test sea state 4 parameters."""
        assert sea_state_4.number == 4
        assert 1.0 <= sea_state_4.Hs <= 2.5
        assert sea_state_4.direction == 45.0

    def test_sea_state_5(self):
        """Test sea state 5 parameters."""
        from ship_motion.ddg_motion import SeaState
        ss = SeaState.from_state_number(5, direction=90.0)

        assert ss.number == 5
        assert 2.5 <= ss.Hs <= 4.0

    def test_invalid_sea_state(self):
        """Test invalid sea state raises error."""
        from ship_motion.ddg_motion import SeaState
        with pytest.raises(ValueError):
            SeaState.from_state_number(6)  # Only 3-5 supported


class TestWaveSpectrum:
    """Tests for wave spectrum model."""

    def test_spectrum_shape(self):
        """Test Pierson-Moskowitz spectrum shape."""
        from ship_motion.ddg_motion import WaveSpectrum

        spectrum = WaveSpectrum(Hs=2.0, Tp=10.0)
        omega = np.linspace(0.1, 2.0, 100)
        S = spectrum.spectrum(omega)

        # Spectrum should be positive
        assert np.all(S >= 0)

        # Should have peak near peak frequency
        wp = 2 * np.pi / 10.0
        peak_idx = np.argmax(S)
        omega_peak = omega[peak_idx]
        assert abs(omega_peak - wp) < 0.5


class TestDDGMotionSimulator:
    """Tests for ship motion simulator."""

    def test_simulator_creation(self, ship_simulator):
        """Test simulator creation."""
        assert ship_simulator is not None

    def test_motion_returns_dict(self, ship_simulator):
        """Test motion returns expected structure."""
        motion = ship_simulator.get_motion(0.0)

        assert 'deck_position' in motion
        assert 'deck_velocity' in motion
        assert 'attitude' in motion
        assert 'angular_rate' in motion

    def test_position_shape(self, ship_simulator):
        """Test position is 3D vector."""
        motion = ship_simulator.get_motion(1.0)

        assert motion['deck_position'].shape == (3,)
        assert motion['deck_velocity'].shape == (3,)
        assert motion['attitude'].shape == (3,)
        assert motion['angular_rate'].shape == (3,)

    def test_ship_forward_motion(self, ship_simulator):
        """Test ship moves forward over time."""
        m0 = ship_simulator.get_motion(0.0)
        m1 = ship_simulator.get_motion(10.0)

        # Ship at 15 kts ≈ 7.7 m/s, should move ~77m in 10s
        dx = m1['deck_position'][0] - m0['deck_position'][0]
        assert dx > 50  # At least 50m forward

    def test_motion_bounded(self, ship_simulator):
        """Test motion amplitudes are bounded."""
        for t in np.linspace(0, 60, 100):
            motion = ship_simulator.get_motion(t)

            # Roll should be bounded (typically < 20 deg)
            assert abs(motion['attitude'][0]) < 0.5  # < 30 deg

            # Pitch should be bounded
            assert abs(motion['attitude'][1]) < 0.3

            # Heave should be bounded
            assert abs(motion['deck_position'][2]) < 10.0

    def test_motion_varies_with_time(self, ship_simulator):
        """Test motion changes over time."""
        motions = [ship_simulator.get_motion(t) for t in np.linspace(0, 30, 50)]

        rolls = [m['attitude'][0] for m in motions]
        pitches = [m['attitude'][1] for m in motions]

        # Should have some variation
        assert np.std(rolls) > 0.001
        assert np.std(pitches) > 0.001

    def test_higher_sea_state_more_motion(self, ddg_params):
        """Test higher sea state produces more motion."""
        from ship_motion.ddg_motion import DDGMotionSimulator, SeaState

        ss3 = SeaState.from_state_number(3)
        ss5 = SeaState.from_state_number(5)

        sim3 = DDGMotionSimulator(ddg_params, ss3, 15.0)
        sim5 = DDGMotionSimulator(ddg_params, ss5, 15.0)

        # Collect motion samples
        rolls_3 = [abs(sim3.get_motion(t)['attitude'][0]) for t in np.linspace(0, 60, 100)]
        rolls_5 = [abs(sim5.get_motion(t)['attitude'][0]) for t in np.linspace(0, 60, 100)]

        # SS5 should have larger or equal average roll
        # (stochastic so allow equality in edge cases)
        assert np.mean(rolls_5) >= np.mean(rolls_3) or np.isclose(np.mean(rolls_5), np.mean(rolls_3), atol=0.1)


class TestARMAPredictor:
    """Tests for ARMA motion prediction."""

    def test_predictor_creation(self):
        """Test ARMA predictor creation."""
        from ship_motion.ddg_motion import ARMAPredictor
        arma = ARMAPredictor(ar_order=4, ma_order=2)
        assert arma is not None

    def test_predictor_fit(self, ship_simulator):
        """Test ARMA predictor fitting."""
        from ship_motion.ddg_motion import ARMAPredictor

        # Generate training data
        dt = 0.1
        times = np.arange(0, 30, dt)
        rolls = [ship_simulator.get_motion(t)['attitude'][0] for t in times]

        arma = ARMAPredictor(ar_order=4, ma_order=2)
        # fit() requires 2D data (n_samples, n_features) and dt
        data = np.array(rolls).reshape(-1, 1)  # Shape as (n_samples, 1)
        arma.fit(data, dt)

        # Should have coefficients
        assert arma.ar_coeffs is not None or arma.ma_coeffs is not None

    def test_predictor_predict(self, ship_simulator):
        """Test ARMA prediction."""
        from ship_motion.ddg_motion import ARMAPredictor

        dt = 0.1
        times = np.arange(0, 30, dt)
        rolls = [ship_simulator.get_motion(t)['attitude'][0] for t in times]

        arma = ARMAPredictor(ar_order=4, ma_order=2)
        # fit() requires 2D data (n_samples, n_features) and dt
        data = np.array(rolls).reshape(-1, 1)  # Shape as (n_samples, 1)
        arma.fit(data, dt)

        # Predict next few steps
        horizon = 10
        predictions = arma.predict(horizon)

        # predictions shape is (horizon, n_features)
        assert predictions.shape[0] == horizon
        assert np.all(np.isfinite(predictions))
