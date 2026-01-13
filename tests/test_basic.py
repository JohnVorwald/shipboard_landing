"""Basic tests for shipboard_landing."""

import pytest
import numpy as np
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that core modules can be imported."""
    try:
        from quad_dynamics.quadrotor import QuadrotorParams
        from ship_motion.ddg_motion import DDGParams
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_numpy():
    """Test numpy is available."""
    assert np.array([1, 2, 3]).sum() == 6

def test_dummy():
    """Dummy test that always passes."""
    assert 1 + 1 == 2