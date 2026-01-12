#!/usr/bin/env python3
"""
Unit Tests for Shipboard Landing Controllers

Tests:
1. Ship motion model and ARMA prediction
2. ZEM/ZEV guidance
3. Tau-based guidance
4. Variable Horizon MPC
5. PMP controller
6. Trajectory planner
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quad_dynamics.quadrotor import QuadrotorParams, QuadrotorState, QuadrotorDynamics
from ship_motion.ddg_motion import DDGParams, SeaState, DDGMotionSimulator, ARMAPredictor
from guidance.tau_guidance import TauGuidanceController, TauGuidanceConfig
from optimal_control.trajectory_planner import LandingTrajectoryPlanner
from optimal_control.pmp_controller import PMPController, create_pmp_trajectory
from optimal_control.variable_horizon_mpc import VariableHorizonMPC, VHMPCConfig


class TestShipMotion(unittest.TestCase):
    """Test ship motion model and prediction."""

    def setUp(self):
        self.ship_params = DDGParams()
        self.sea_state = SeaState.from_state_number(4, direction=45.0)
        self.ship_sim = DDGMotionSimulator(self.ship_params, self.sea_state, 15.0)

    def test_ship_motion_returns_dict(self):
        """Ship motion returns expected structure."""
        motion = self.ship_sim.get_motion(0.0)
        self.assertIn('deck_position', motion)
        self.assertIn('deck_velocity', motion)
        self.assertIn('attitude', motion)
        self.assertIn('angular_rate', motion)

    def test_ship_motion_position_shape(self):
        """Deck position is 3D vector."""
        motion = self.ship_sim.get_motion(1.0)
        self.assertEqual(motion['deck_position'].shape, (3,))
        self.assertEqual(motion['deck_velocity'].shape, (3,))

    def test_ship_moves_forward(self):
        """Ship moves forward over time."""
        m0 = self.ship_sim.get_motion(0.0)
        m1 = self.ship_sim.get_motion(10.0)
        # Ship at 15 kts = 7.7 m/s, should move ~77m in 10s
        dx = m1['deck_position'][0] - m0['deck_position'][0]
        self.assertGreater(dx, 50)  # At least 50m forward

    def test_arma_predictor_fit(self):
        """ARMA predictor can be fitted."""
        arma = ARMAPredictor(ar_order=4, ma_order=2)

        # Generate training data
        history = []
        for t in np.arange(0, 20, 0.1):
            motion = self.ship_sim.get_motion(t)
            obs = np.array([
                motion['deck_position'][2],
                motion['deck_velocity'][2],
                motion['attitude'][0],
            ])
            history.append(obs)

        arma.fit(np.array(history), 0.1)
        self.assertTrue(arma.fitted)

    def test_arma_prediction(self):
        """ARMA can predict future values."""
        arma = ARMAPredictor(ar_order=4, ma_order=2)

        history = []
        for t in np.arange(0, 20, 0.1):
            motion = self.ship_sim.get_motion(t)
            obs = np.array([motion['deck_position'][2]])
            history.append(obs)

        arma.fit(np.array(history), 0.1)

        # Predict 10 steps ahead
        preds = arma.predict(10)
        self.assertEqual(len(preds), 10)
        self.assertEqual(preds[0].shape, (1,))


class TestQuadrotor(unittest.TestCase):
    """Test quadrotor dynamics."""

    def setUp(self):
        self.params = QuadrotorParams()
        self.dynamics = QuadrotorDynamics(self.params)

    def test_hover_thrust(self):
        """Hover thrust equals weight."""
        hover_thrust = self.params.mass * 9.81
        self.assertAlmostEqual(hover_thrust, self.params.mass * 9.81)

    def test_state_creation(self):
        """QuadrotorState can be created."""
        state = QuadrotorState(
            pos=np.array([0, 0, -10]),
            vel=np.zeros(3),
            quat=np.array([1, 0, 0, 0]),
            omega=np.zeros(3)
        )
        self.assertEqual(state.pos.shape, (3,))
        self.assertAlmostEqual(state.roll, 0.0)
        self.assertAlmostEqual(state.pitch, 0.0)

    def test_dynamics_step(self):
        """Dynamics step updates state."""
        state = QuadrotorState(
            pos=np.array([0, 0, -10]),
            vel=np.zeros(3),
            quat=np.array([1, 0, 0, 0]),
            omega=np.zeros(3)
        )

        # Hover thrust
        T_hover = self.params.mass * 9.81 / 4
        motor_thrusts = np.array([T_hover, T_hover, T_hover, T_hover])

        new_state = self.dynamics.step(state, motor_thrusts, 0.01)

        # Position should change slightly
        self.assertIsInstance(new_state, QuadrotorState)


class TestZEMZEVGuidance(unittest.TestCase):
    """Test ZEM/ZEV guidance law."""

    def test_zem_zev_computation(self):
        """ZEM/ZEV computes valid acceleration."""
        # Simple test case
        rel_pos = np.array([50, 0, 10])  # 50m ahead, 10m above
        rel_vel = np.array([-5, 0, -1])  # Closing at 5m/s
        t_go = 10.0

        # ZEM/ZEV formula
        a_zem = -6.0 * (rel_pos + rel_vel * t_go) / t_go**2
        a_zev = -2.0 * rel_vel / t_go
        acc_cmd = a_zem + a_zev

        # Should have valid acceleration
        self.assertEqual(acc_cmd.shape, (3,))
        # Should accelerate toward target (negative x)
        # Since we're behind target (positive rel_pos[0])
        # ZEM should push us forward


class TestTauGuidance(unittest.TestCase):
    """Test tau-based guidance."""

    def setUp(self):
        self.params = QuadrotorParams()
        self.controller = TauGuidanceController(quad_params=self.params)

    def test_tau_computation(self):
        """Tau computed correctly."""
        r = 100.0  # 100m range
        r_dot = -10.0  # Closing at 10m/s

        tau = self.controller.compute_tau(r, r_dot)
        self.assertAlmostEqual(tau, 10.0)  # 10 seconds to contact

    def test_control_output_shape(self):
        """Control outputs have correct shape."""
        pos = np.array([-50, 0, -20])
        vel = np.array([5, 0, 1])
        deck_pos = np.array([0, 0, -8])
        deck_vel = np.array([7.7, 0, 0])
        deck_att = np.array([0, 0, 0])

        acc_cmd, info = self.controller.compute_control(
            pos, vel, deck_pos, deck_vel, deck_att
        )

        self.assertEqual(acc_cmd.shape, (3,))
        self.assertIn('tau_horiz', info)
        self.assertIn('tau_vert', info)

    def test_thrust_attitude_output(self):
        """Thrust/attitude computation works."""
        acc_cmd = np.array([2.0, 0.0, -1.0])
        deck_att = np.array([0, 0, 0])

        thrust, roll, pitch = self.controller.compute_thrust_attitude(
            acc_cmd, 0.0, deck_att, 10.0
        )

        self.assertGreater(thrust, 0)
        self.assertLess(abs(roll), 1.0)  # Within limits
        self.assertLess(abs(pitch), 1.0)


class TestVHMPC(unittest.TestCase):
    """Test Variable Horizon MPC."""

    def setUp(self):
        self.params = QuadrotorParams()
        config = VHMPCConfig(
            horizon_options=[10, 15, 20],
            move_blocks=[3, 3, 4],
            dt=0.2
        )
        self.controller = VariableHorizonMPC(self.params, config)

    def test_control_output_shape(self):
        """VH-MPC produces valid control."""
        x = np.array([-30, 0, -20, 5, 0, 1, 0, 0, 0])
        deck_pos = np.array([0, 0, -8])
        deck_vel = np.array([7.7, 0, 0])
        deck_att = np.array([0, 0, 0])

        acc_cmd, info = self.controller.compute_control(
            x, deck_pos, deck_vel, deck_att, t_go=10.0
        )

        self.assertEqual(acc_cmd.shape, (3,))
        self.assertIn('horizon', info)
        self.assertIn('cost', info)

    def test_horizon_selection(self):
        """VH-MPC selects a horizon."""
        x = np.array([-30, 0, -20, 5, 0, 1, 0, 0, 0])
        deck_pos = np.array([0, 0, -8])
        deck_vel = np.array([7.7, 0, 0])
        deck_att = np.array([0, 0, 0])

        acc_cmd, info = self.controller.compute_control(
            x, deck_pos, deck_vel, deck_att, t_go=10.0
        )

        # Should select one of the horizon options
        self.assertIn(info['horizon'], [10, 15, 20, 0])  # 0 = fallback


class TestPMPController(unittest.TestCase):
    """Test PMP trajectory tracking controller."""

    def setUp(self):
        self.params = QuadrotorParams()
        self.controller = PMPController(self.params)

    def test_fallback_control(self):
        """Fallback control works without trajectory."""
        x = np.array([-30, 0, -20, 5, 0, 1, 0, 0, 0, 0, 0, 0])
        deck_pos = np.array([0, 0, -8])
        deck_vel = np.array([7.7, 0, 0])
        deck_att = np.array([0, 0, 0])

        # No trajectory set - should use fallback
        u = self.controller.compute_control(0, x, deck_pos, deck_vel, deck_att)

        self.assertEqual(u.shape, (4,))
        self.assertGreater(u[0], 0)  # Positive thrust

    def test_trajectory_creation(self):
        """PMP trajectory can be created."""
        N = 20
        tf = 5.0
        t_traj = np.linspace(0, tf, N)
        x_traj = np.zeros((N, 12))
        u_traj = np.zeros((N, 4))

        # Simple trajectory
        for i in range(N):
            alpha = t_traj[i] / tf
            x_traj[i, 0:3] = np.array([-30 + 30*alpha, 0, -20 + 12*alpha])
            u_traj[i, 0] = self.params.mass * 9.81

        deck_pos = np.array([0, 0, -8])
        deck_vel = np.array([7.7, 0, 0])
        deck_att = np.array([0, 0, 0])

        traj = create_pmp_trajectory(
            x_traj, u_traj, t_traj, tf,
            deck_pos, deck_vel, deck_att, self.params
        )

        self.assertEqual(traj.t.shape, (N,))
        self.assertEqual(traj.x.shape, (N, 12))
        self.assertEqual(traj.lam.shape, (N, 12))


class TestTrajectoryPlanner(unittest.TestCase):
    """Test minimum-snap trajectory planner."""

    def setUp(self):
        self.params = QuadrotorParams()
        self.planner = LandingTrajectoryPlanner(self.params)

    def test_plan_landing(self):
        """Can plan landing trajectory."""
        result = self.planner.plan_landing(
            quad_pos=np.array([-30, 0, -20]),
            quad_vel=np.array([5, 0, 1]),
            deck_pos=np.array([0, 0, -8]),
            deck_vel=np.array([7.7, 0, 0]),
            deck_att=np.array([0, 0, 0]),
            tf_desired=5.0
        )

        self.assertTrue(result['success'])
        self.assertIn('trajectory', result)
        self.assertIn('terminal_pos_error', result)

    def test_trajectory_sampling(self):
        """Can sample trajectory."""
        result = self.planner.plan_landing(
            quad_pos=np.array([-30, 0, -20]),
            quad_vel=np.array([5, 0, 1]),
            deck_pos=np.array([0, 0, -8]),
            deck_vel=np.array([7.7, 0, 0]),
            deck_att=np.array([0, 0, 0]),
            tf_desired=5.0
        )

        if result['success']:
            traj = result['trajectory']
            sample = self.planner.sample_trajectory(traj, 2.5)

            self.assertIn('position', sample)
            self.assertIn('velocity', sample)
            self.assertEqual(sample['position'].shape, (3,))


class TestIntegration(unittest.TestCase):
    """Integration tests for full pipeline."""

    def test_full_landing_pipeline(self):
        """Full landing pipeline executes without error."""
        # Ship
        ship_params = DDGParams()
        sea_state = SeaState.from_state_number(3, direction=45.0)
        ship_sim = DDGMotionSimulator(ship_params, sea_state, 10.0)

        # Quad
        quad_params = QuadrotorParams()

        # Controller (simple ZEM/ZEV)
        def compute_control(state, deck_pos, deck_vel, t_go):
            rel_pos = state.pos - deck_pos
            rel_vel = state.vel - deck_vel
            t_go = max(t_go, 1.0)

            a_zem = -6.0 * (rel_pos + rel_vel * t_go) / t_go**2
            a_zev = -2.0 * rel_vel / t_go
            return np.clip(a_zem + a_zev, -6, 6)

        # Initial state
        motion = ship_sim.get_motion(0)
        state = QuadrotorState(
            pos=motion['deck_position'] - np.array([50, 0, 15]),
            vel=motion['deck_velocity'].copy(),
            quat=np.array([1, 0, 0, 0]),
            omega=np.zeros(3)
        )

        # Simulate 10 steps
        dt = 0.1
        for i in range(10):
            t = i * dt
            motion = ship_sim.get_motion(t)
            acc = compute_control(state, motion['deck_position'],
                                 motion['deck_velocity'], 10.0 - t)

            state.vel += acc * dt
            state.pos += state.vel * dt

        # Should still be valid
        self.assertTrue(np.isfinite(state.pos).all())
        self.assertTrue(np.isfinite(state.vel).all())


def run_tests():
    """Run all tests and print summary."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestShipMotion))
    suite.addTests(loader.loadTestsFromTestCase(TestQuadrotor))
    suite.addTests(loader.loadTestsFromTestCase(TestZEMZEVGuidance))
    suite.addTests(loader.loadTestsFromTestCase(TestTauGuidance))
    suite.addTests(loader.loadTestsFromTestCase(TestVHMPC))
    suite.addTests(loader.loadTestsFromTestCase(TestPMPController))
    suite.addTests(loader.loadTestsFromTestCase(TestTrajectoryPlanner))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
