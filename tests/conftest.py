"""Pytest configuration and fixtures for shipboard_landing tests."""

import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def ddg_params():
    """DDG ship parameters."""
    from ship_motion.ddg_motion import DDGParams
    return DDGParams()


@pytest.fixture
def sea_state_4():
    """Sea state 4 conditions."""
    from ship_motion.ddg_motion import SeaState
    return SeaState.from_state_number(4, direction=45.0)


@pytest.fixture
def ship_simulator(ddg_params, sea_state_4):
    """Ship motion simulator."""
    from ship_motion.ddg_motion import DDGMotionSimulator
    return DDGMotionSimulator(ddg_params, sea_state_4, ship_speed_kts=15.0)


@pytest.fixture
def quad_params():
    """Quadrotor parameters."""
    from quad_dynamics.quadrotor import QuadrotorParams
    return QuadrotorParams()


@pytest.fixture
def quad_state():
    """Sample quadrotor state."""
    from quad_dynamics.quadrotor import QuadrotorState
    return QuadrotorState(
        position=np.array([50.0, 0.0, -30.0]),
        velocity=np.array([5.0, 0.0, -2.0]),
        attitude=np.array([0.0, 0.0, 0.0]),
        angular_rate=np.array([0.0, 0.0, 0.0])
    )


@pytest.fixture
def tau_guidance():
    """Tau guidance controller."""
    from guidance.tau_guidance import TauGuidanceController, TauGuidanceConfig
    config = TauGuidanceConfig()
    return TauGuidanceController(config)


@pytest.fixture
def pmp_controller():
    """PMP controller."""
    from optimal_control.pmp_controller import PMPController
    return PMPController()


@pytest.fixture
def trajectory_planner():
    """Trajectory planner."""
    from optimal_control.trajectory_planner import LandingTrajectoryPlanner
    return LandingTrajectoryPlanner()
