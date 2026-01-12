"""Unit tests for guidance modules."""

import pytest
import numpy as np
from numpy.testing import assert_allclose


class TestTauGuidanceConfig:
    """Tests for tau guidance configuration."""

    def test_default_config(self):
        """Test default configuration."""
        from guidance.tau_guidance import TauGuidanceConfig
        config = TauGuidanceConfig()

        # Tau dot should be between 0 and 1
        assert 0 < config.tau_dot_approach < 1
        assert 0 < config.tau_dot_terminal < 1
        assert config.tau_dot_terminal <= config.tau_dot_approach

    def test_limits_positive(self):
        """Test limits are positive."""
        from guidance.tau_guidance import TauGuidanceConfig
        config = TauGuidanceConfig()

        assert config.max_accel > 0
        assert config.max_descent > 0
        assert config.max_roll > 0
        assert config.max_pitch > 0


class TestTauGuidanceController:
    """Tests for tau guidance controller."""

    def test_controller_creation(self, tau_guidance):
        """Test controller creation."""
        assert tau_guidance is not None
        assert hasattr(tau_guidance, 'config')

    def test_compute_tau_approaching(self, tau_guidance):
        """Test tau computation when approaching."""
        # Distance 100m, closing at 10 m/s -> tau = 10s
        r = 100.0
        r_dot = -10.0  # Closing (negative)

        tau = tau_guidance.compute_tau(r, r_dot)
        assert_allclose(tau, 10.0, rtol=0.1)

    def test_compute_tau_retreating(self, tau_guidance):
        """Test tau computation when moving away."""
        r = 100.0
        r_dot = 5.0  # Moving away (positive)

        tau = tau_guidance.compute_tau(r, r_dot)
        # Should be negative or large magnitude
        assert tau < 0 or abs(tau) > 50

    def test_compute_tau_stationary(self, tau_guidance):
        """Test tau computation when stationary."""
        r = 100.0
        r_dot = 0.0  # Not moving

        tau = tau_guidance.compute_tau(r, r_dot)
        # Should return large tau (far from contact)
        assert abs(tau) > 50

    def test_tau_dot_desired_high_altitude(self, tau_guidance):
        """Test tau_dot selection at high altitude."""
        height = 50.0
        horizontal = 100.0

        tau_h, tau_v = tau_guidance.compute_tau_dot_desired(height, horizontal)

        # At high altitude, use approach tau_dot
        assert tau_h == tau_guidance.config.tau_dot_horizontal or tau_h > 0
        assert tau_v > 0

    def test_tau_dot_desired_terminal(self, tau_guidance):
        """Test tau_dot selection near touchdown."""
        height = 2.0  # Below transition height
        horizontal = 5.0

        tau_h, tau_v = tau_guidance.compute_tau_dot_desired(height, horizontal)

        # Near ground, should use terminal (gentler) tau_dot
        assert tau_v <= tau_guidance.config.tau_dot_vertical

    def test_command_computation(self, tau_guidance):
        """Test guidance command computation."""
        # Quadrotor 100m away, 30m high, approaching
        quad_pos = np.array([100.0, 0.0, -30.0])
        quad_vel = np.array([-5.0, 0.0, 2.0])  # Approaching and descending
        deck_pos = np.array([0.0, 0.0, 0.0])
        deck_vel = np.array([0.0, 0.0, 0.0])
        deck_att = np.array([0.0, 0.0, 0.0])

        # Actual API is compute_control(), not compute_command()
        acc_cmd, info = tau_guidance.compute_control(
            quad_pos, quad_vel, deck_pos, deck_vel, deck_att
        )

        # Should return 3-element acceleration command
        assert len(acc_cmd) == 3
        assert np.all(np.isfinite(acc_cmd))
        # Should return info dict
        assert isinstance(info, dict)


class TestZEMGuidance:
    """Tests for Zero Effort Miss guidance (if implemented)."""

    def test_zem_computation(self):
        """Test ZEM guidance law."""
        try:
            from guidance.zem_guidance import ZEMGuidance
            zem = ZEMGuidance()

            # Position and velocity
            pos = np.array([100.0, 0.0, -30.0])
            vel = np.array([-5.0, 0.0, 2.0])
            target = np.array([0.0, 0.0, 0.0])

            accel = zem.compute_acceleration(pos, vel, target, t_go=10.0)

            assert len(accel) == 3
            assert np.all(np.isfinite(accel))
        except ImportError:
            pytest.skip("ZEMGuidance not implemented")


class TestHigherOrderGuidance:
    """Tests for higher-order tau guidance (if implemented)."""

    def test_second_order_tau(self):
        """Test 2nd order tau guidance."""
        try:
            from guidance.higher_order_tau import SecondOrderTauGuidance

            guidance = SecondOrderTauGuidance()
            assert guidance is not None
        except ImportError:
            pytest.skip("SecondOrderTauGuidance not implemented")
