#!/usr/bin/env python3
"""
Compare forecasting methods:
1. ARMA - Direct time series on ship motion
2. ARMA Higher-Order - Includes velocity and acceleration terms
3. Wave Estimator - Physics-based wave model inversion

Evaluate RMS error vs forecast horizon for deck position prediction.
Updated forecasts every 0.5s with higher-order terms.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ship_motion.ddg_motion import (DDGParams, SeaState, DDGMotionSimulator,
                                     ARMAPredictor, WaveEstimatorPredictor)


class HigherOrderARMAPredictor:
    """
    Enhanced ARMA predictor with velocity and acceleration terms.

    State includes: [z, dz/dt, d²z/dt², roll, droll/dt, pitch, dpitch/dt]
    Uses Taylor expansion for prediction: x(t+dt) = x + dx*dt + 0.5*d2x*dt²
    """

    def __init__(self, ar_order: int = 8, ma_order: int = 4):
        self.ar_order = ar_order
        self.ma_order = ma_order

        # Separate predictors for each channel
        self.predictors = {}
        for key in ['z', 'z_vel', 'z_acc', 'roll', 'roll_rate', 'pitch', 'pitch_rate']:
            self.predictors[key] = {
                'ar_coeffs': None,
                'ma_coeffs': None,
                'history': [],
                'residuals': []
            }

        self.fitted = False
        self.dt = 0.1
        self.data_mean = {}
        self.data_std = {}

    def fit(self, data: np.ndarray, dt: float):
        """
        Fit higher-order ARMA model.

        Args:
            data: (n_samples, 7) - z, z_vel, z_acc, roll, roll_rate, pitch, pitch_rate
            dt: Sample interval
        """
        self.dt = dt
        n_samples, n_features = data.shape

        keys = ['z', 'z_vel', 'z_acc', 'roll', 'roll_rate', 'pitch', 'pitch_rate']

        for i, key in enumerate(keys):
            channel_data = data[:, i]

            # Normalize
            self.data_mean[key] = np.mean(channel_data)
            self.data_std[key] = np.std(channel_data)
            if self.data_std[key] < 1e-6:
                self.data_std[key] = 1.0

            data_norm = (channel_data - self.data_mean[key]) / self.data_std[key]

            # Fit AR using ridge regression
            y = data_norm[self.ar_order:]
            X = np.zeros((len(y), self.ar_order))
            for j in range(self.ar_order):
                X[:, j] = data_norm[self.ar_order - 1 - j:-1 - j]

            ridge_lambda = 0.01
            XtX = X.T @ X + ridge_lambda * np.eye(self.ar_order)
            Xty = X.T @ y
            ar_coeffs = np.linalg.solve(XtX, Xty)

            # Clamp for stability
            coeff_sum = np.sum(np.abs(ar_coeffs))
            if coeff_sum > 0.95:
                ar_coeffs *= 0.95 / coeff_sum

            # Compute residuals
            y_pred = X @ ar_coeffs
            residuals = y - y_pred

            # Fit MA
            ma_coeffs = np.zeros(self.ma_order)
            if len(residuals) > self.ma_order + 10:
                R = np.zeros((len(residuals) - self.ma_order, self.ma_order))
                for j in range(self.ma_order):
                    R[:, j] = residuals[self.ma_order - 1 - j:-1 - j]
                y_ma = residuals[self.ma_order:]
                RtR = R.T @ R + ridge_lambda * np.eye(self.ma_order)
                Rty = R.T @ y_ma
                ma_coeffs = np.linalg.solve(RtR, Rty)

                ma_sum = np.sum(np.abs(ma_coeffs))
                if ma_sum > 0.9:
                    ma_coeffs *= 0.9 / ma_sum

            self.predictors[key]['ar_coeffs'] = ar_coeffs
            self.predictors[key]['ma_coeffs'] = ma_coeffs
            self.predictors[key]['history'] = list(data_norm[-self.ar_order:])
            self.predictors[key]['residuals'] = [0.0] * self.ma_order

        self.fitted = True

    def update(self, observation: np.ndarray):
        """Update with new observation (7-element: z, z_vel, z_acc, roll, roll_rate, pitch, pitch_rate)."""
        if not self.fitted:
            return

        keys = ['z', 'z_vel', 'z_acc', 'roll', 'roll_rate', 'pitch', 'pitch_rate']

        for i, key in enumerate(keys):
            obs_norm = (observation[i] - self.data_mean[key]) / self.data_std[key]

            # Compute residual
            pred_norm = self._predict_channel_one_step(key)
            residual = obs_norm - pred_norm

            # Update history
            self.predictors[key]['history'].append(obs_norm)
            if len(self.predictors[key]['history']) > self.ar_order:
                self.predictors[key]['history'].pop(0)

            self.predictors[key]['residuals'].append(residual)
            if len(self.predictors[key]['residuals']) > self.ma_order:
                self.predictors[key]['residuals'].pop(0)

    def _predict_channel_one_step(self, key: str) -> float:
        """Predict one step for single channel."""
        pred = 0.0
        p = self.predictors[key]

        for i, coeff in enumerate(p['ar_coeffs']):
            if i < len(p['history']):
                pred += coeff * p['history'][-(i+1)]

        for i, coeff in enumerate(p['ma_coeffs']):
            if i < len(p['residuals']):
                pred += coeff * p['residuals'][-(i+1)]

        return pred

    def predict(self, horizon: int) -> np.ndarray:
        """
        Predict multiple steps using higher-order terms.

        Uses Taylor expansion: x(t+dt) = x + v*dt + 0.5*a*dt²
        Combined with ARMA predictions for each term.

        Returns:
            (horizon, 7) predictions
        """
        if not self.fitted:
            return np.zeros((horizon, 7))

        keys = ['z', 'z_vel', 'z_acc', 'roll', 'roll_rate', 'pitch', 'pitch_rate']
        predictions = np.zeros((horizon, 7))

        # Copy histories for multi-step
        temp_histories = {key: list(self.predictors[key]['history']) for key in keys}
        temp_residuals = {key: list(self.predictors[key]['residuals']) for key in keys}

        for k in range(horizon):
            # ARMA prediction for each channel
            arma_preds = {}
            for key in keys:
                pred = 0.0
                p = self.predictors[key]
                for i, coeff in enumerate(p['ar_coeffs']):
                    if i < len(temp_histories[key]):
                        pred += coeff * temp_histories[key][-(i+1)]
                for i, coeff in enumerate(p['ma_coeffs']):
                    if i < len(temp_residuals[key]) and k <= i:
                        pred += coeff * temp_residuals[key][-(i+1)]
                arma_preds[key] = pred

            # Denormalize ARMA predictions
            z_arma = arma_preds['z'] * self.data_std['z'] + self.data_mean['z']
            z_vel_arma = arma_preds['z_vel'] * self.data_std['z_vel'] + self.data_mean['z_vel']
            z_acc_arma = arma_preds['z_acc'] * self.data_std['z_acc'] + self.data_mean['z_acc']

            roll_arma = arma_preds['roll'] * self.data_std['roll'] + self.data_mean['roll']
            roll_rate_arma = arma_preds['roll_rate'] * self.data_std['roll_rate'] + self.data_mean['roll_rate']

            pitch_arma = arma_preds['pitch'] * self.data_std['pitch'] + self.data_mean['pitch']
            pitch_rate_arma = arma_preds['pitch_rate'] * self.data_std['pitch_rate'] + self.data_mean['pitch_rate']

            # Use Taylor expansion for position using velocity and acceleration from ARMA
            # x(t+dt) = x + v*dt + 0.5*a*dt²
            # This blends ARMA with kinematic consistency
            if k == 0:
                # First step: get current values from last history
                z_curr = temp_histories['z'][-1] * self.data_std['z'] + self.data_mean['z']
                roll_curr = temp_histories['roll'][-1] * self.data_std['roll'] + self.data_mean['roll']
                pitch_curr = temp_histories['pitch'][-1] * self.data_std['pitch'] + self.data_mean['pitch']
            else:
                z_curr = predictions[k-1, 0]
                roll_curr = predictions[k-1, 3]
                pitch_curr = predictions[k-1, 5]

            # Blend ARMA with Taylor expansion (50% each for robustness)
            z_taylor = z_curr + z_vel_arma * self.dt + 0.5 * z_acc_arma * self.dt**2
            z_pred = 0.5 * z_arma + 0.5 * z_taylor

            roll_taylor = roll_curr + roll_rate_arma * self.dt
            roll_pred = 0.5 * roll_arma + 0.5 * roll_taylor

            pitch_taylor = pitch_curr + pitch_rate_arma * self.dt
            pitch_pred = 0.5 * pitch_arma + 0.5 * pitch_taylor

            predictions[k] = [z_pred, z_vel_arma, z_acc_arma,
                             roll_pred, roll_rate_arma, pitch_pred, pitch_rate_arma]

            # Update temp histories with normalized predictions
            for i, key in enumerate(keys):
                val_norm = (predictions[k, i] - self.data_mean[key]) / self.data_std[key]
                temp_histories[key].append(val_norm)
                if len(temp_histories[key]) > self.ar_order:
                    temp_histories[key].pop(0)
                temp_residuals[key].append(0.0)
                if len(temp_residuals[key]) > self.ma_order:
                    temp_residuals[key].pop(0)

        return predictions


def run_comparison(sea_state_num: int = 4, n_trials: int = 10, update_interval: float = 0.5):
    """Compare ARMA vs Higher-Order ARMA vs Wave Estimator forecasting accuracy."""

    print("="*80)
    print("FORECASTING METHOD COMPARISON (Higher-Order with 0.5s Updates)")
    print("="*80)
    print(f"Sea State: {sea_state_num}")
    print(f"Trials: {n_trials}")
    print(f"Forecast Update Interval: {update_interval}s")
    print()

    horizons = [1, 2, 3, 5, 7, 10]  # seconds
    dt = 0.1  # Simulation timestep

    # Accumulators for errors
    arma_errors = {h: {'y': [], 'z': [], 'roll': [], 'pitch': []} for h in horizons}
    ho_arma_errors = {h: {'y': [], 'z': [], 'roll': [], 'pitch': []} for h in horizons}
    wave_errors = {h: {'y': [], 'z': [], 'roll': [], 'pitch': []} for h in horizons}

    for trial in range(n_trials):
        np.random.seed(trial * 100)

        # Setup
        ship_params = DDGParams()
        sea_state = SeaState.from_state_number(sea_state_num, direction=45.0)
        ship_sim = DDGMotionSimulator(ship_params, sea_state, ship_speed_kts=15.0)

        # Standard ARMA predictor (6 features)
        arma = ARMAPredictor(ar_order=8, ma_order=4)

        # Higher-order ARMA predictor (7 features with acceleration)
        ho_arma = HigherOrderARMAPredictor(ar_order=8, ma_order=4)

        # Wave estimator
        wave_est = WaveEstimatorPredictor(ship_params, sea_state,
                                          n_wave_components=15, ship_speed_kts=15.0)

        # Warm up period - collect 30s of history
        history_data = []
        ho_history_data = []
        prev_vel_z = 0

        for t in np.arange(0, 30, dt):
            motion = ship_sim.get_motion(t)

            # Standard ARMA observation
            obs = np.array([
                motion['deck_position'][2],
                motion['deck_velocity'][2],
                motion['deck_velocity'][0],
                motion['attitude'][0],
                motion['attitude'][1],
                motion['angular_rate'][0],
            ])
            history_data.append(obs)

            # Higher-order observation (with acceleration)
            acc_z = (motion['deck_velocity'][2] - prev_vel_z) / dt if t > 0 else 0
            prev_vel_z = motion['deck_velocity'][2]

            ho_obs = np.array([
                motion['deck_position'][2],
                motion['deck_velocity'][2],
                acc_z,
                motion['attitude'][0],
                motion['angular_rate'][0],
                motion['attitude'][1],
                motion['angular_rate'][1],
            ])
            ho_history_data.append(ho_obs)

            wave_est.add_observation(t, motion['attitude'][0],
                                    motion['attitude'][1],
                                    motion['deck_position'][2])

        # Fit predictors
        arma.fit(np.array(history_data), dt)
        ho_arma.fit(np.array(ho_history_data), dt)
        wave_est.fit()

        # Test period: 30s to 60s with 0.5s update intervals
        test_times = np.arange(30.0, 60.0, update_interval)

        for t_test in test_times:
            # Update predictors with observations since last update
            t_start = t_test - update_interval
            prev_vel = history_data[-1][1] if len(history_data) > 0 else 0

            for t in np.arange(t_start, t_test, dt):
                motion = ship_sim.get_motion(t)

                # Standard ARMA update
                obs = np.array([
                    motion['deck_position'][2],
                    motion['deck_velocity'][2],
                    motion['deck_velocity'][0],
                    motion['attitude'][0],
                    motion['attitude'][1],
                    motion['angular_rate'][0],
                ])
                arma.update(obs)

                # Higher-order update
                acc_z = (motion['deck_velocity'][2] - prev_vel) / dt
                prev_vel = motion['deck_velocity'][2]

                ho_obs = np.array([
                    motion['deck_position'][2],
                    motion['deck_velocity'][2],
                    acc_z,
                    motion['attitude'][0],
                    motion['angular_rate'][0],
                    motion['attitude'][1],
                    motion['angular_rate'][1],
                ])
                ho_arma.update(ho_obs)

                wave_est.add_observation(t, motion['attitude'][0],
                                        motion['attitude'][1],
                                        motion['deck_position'][2])

            # Refit wave estimator periodically
            wave_est.fit(refit_interval=1.0)

            current_motion = ship_sim.get_motion(t_test)

            for horizon in horizons:
                t_pred = t_test + horizon
                if t_pred > 60.0:
                    continue

                # Ground truth
                true_motion = ship_sim.get_motion(t_pred)
                true_deck_y = true_motion['deck_position'][1]
                true_deck_z = true_motion['deck_position'][2]
                true_roll = true_motion['attitude'][0]
                true_pitch = true_motion['attitude'][1]

                # Standard ARMA prediction
                arma_preds = arma.predict(int(horizon / dt))
                if len(arma_preds) > 0:
                    pred = arma_preds[-1]
                    arma_z = pred[0]
                    arma_roll = pred[3]
                    arma_pitch = pred[4]
                    arma_y = current_motion['deck_position'][1]  # Assume constant
                else:
                    arma_z = true_deck_z
                    arma_roll = true_roll
                    arma_pitch = true_pitch
                    arma_y = true_deck_y

                arma_errors[horizon]['y'].append(abs(arma_y - true_deck_y))
                arma_errors[horizon]['z'].append(abs(arma_z - true_deck_z))
                arma_errors[horizon]['roll'].append(abs(arma_roll - true_roll))
                arma_errors[horizon]['pitch'].append(abs(arma_pitch - true_pitch))

                # Higher-order ARMA prediction
                ho_preds = ho_arma.predict(int(horizon / dt))
                if len(ho_preds) > 0:
                    pred = ho_preds[-1]
                    ho_z = pred[0]
                    ho_roll = pred[3]
                    ho_pitch = pred[5]
                    ho_y = current_motion['deck_position'][1]
                else:
                    ho_z = true_deck_z
                    ho_roll = true_roll
                    ho_pitch = true_pitch
                    ho_y = true_deck_y

                ho_arma_errors[horizon]['y'].append(abs(ho_y - true_deck_y))
                ho_arma_errors[horizon]['z'].append(abs(ho_z - true_deck_z))
                ho_arma_errors[horizon]['roll'].append(abs(ho_roll - true_roll))
                ho_arma_errors[horizon]['pitch'].append(abs(ho_pitch - true_pitch))

                # Wave estimator prediction
                wave_pred = wave_est.predict_deck_state(t_pred, ship_sim)
                wave_y = wave_pred['deck_position'][1]
                wave_z = wave_pred['deck_position'][2]
                wave_roll = wave_pred['attitude'][0]
                wave_pitch = wave_pred['attitude'][1]

                wave_errors[horizon]['y'].append(abs(wave_y - true_deck_y))
                wave_errors[horizon]['z'].append(abs(wave_z - true_deck_z))
                wave_errors[horizon]['roll'].append(abs(wave_roll - true_roll))
                wave_errors[horizon]['pitch'].append(abs(wave_pitch - true_pitch))

    # Print results
    print("RMS Errors by Horizon (meters for Y/Z, degrees for Roll/Pitch):")
    print()

    # Header
    print(f"{'Horizon':>8} | {'Standard ARMA':^42} | {'Higher-Order ARMA':^42} | {'Wave Estimator':^42}")
    print(f"{'':>8} | {'Y':>10}{'Z':>10}{'Roll':>11}{'Pitch':>11} | {'Y':>10}{'Z':>10}{'Roll':>11}{'Pitch':>11} | {'Y':>10}{'Z':>10}{'Roll':>11}{'Pitch':>11}")
    print("-"*140)

    for h in horizons:
        if len(arma_errors[h]['z']) == 0:
            continue

        arma_y_rms = np.sqrt(np.mean(np.array(arma_errors[h]['y'])**2))
        arma_z_rms = np.sqrt(np.mean(np.array(arma_errors[h]['z'])**2))
        arma_roll_rms = np.degrees(np.sqrt(np.mean(np.array(arma_errors[h]['roll'])**2)))
        arma_pitch_rms = np.degrees(np.sqrt(np.mean(np.array(arma_errors[h]['pitch'])**2)))

        ho_y_rms = np.sqrt(np.mean(np.array(ho_arma_errors[h]['y'])**2))
        ho_z_rms = np.sqrt(np.mean(np.array(ho_arma_errors[h]['z'])**2))
        ho_roll_rms = np.degrees(np.sqrt(np.mean(np.array(ho_arma_errors[h]['roll'])**2)))
        ho_pitch_rms = np.degrees(np.sqrt(np.mean(np.array(ho_arma_errors[h]['pitch'])**2)))

        wave_y_rms = np.sqrt(np.mean(np.array(wave_errors[h]['y'])**2))
        wave_z_rms = np.sqrt(np.mean(np.array(wave_errors[h]['z'])**2))
        wave_roll_rms = np.degrees(np.sqrt(np.mean(np.array(wave_errors[h]['roll'])**2)))
        wave_pitch_rms = np.degrees(np.sqrt(np.mean(np.array(wave_errors[h]['pitch'])**2)))

        print(f"{h:>6}s  | {arma_y_rms:>10.3f}{arma_z_rms:>10.3f}{arma_roll_rms:>10.2f}°{arma_pitch_rms:>10.2f}° |"
              f" {ho_y_rms:>10.3f}{ho_z_rms:>10.3f}{ho_roll_rms:>10.2f}°{ho_pitch_rms:>10.2f}° |"
              f" {wave_y_rms:>10.3f}{wave_z_rms:>10.3f}{wave_roll_rms:>10.2f}°{wave_pitch_rms:>10.2f}°")

    print()
    print("Improvement Summary (Higher-Order vs Standard ARMA):")
    print("-"*60)

    for h in horizons:
        if len(arma_errors[h]['z']) == 0:
            continue

        arma_z_rms = np.sqrt(np.mean(np.array(arma_errors[h]['z'])**2))
        ho_z_rms = np.sqrt(np.mean(np.array(ho_arma_errors[h]['z'])**2))

        arma_roll_rms = np.sqrt(np.mean(np.array(arma_errors[h]['roll'])**2))
        ho_roll_rms = np.sqrt(np.mean(np.array(ho_arma_errors[h]['roll'])**2))

        z_improvement = (arma_z_rms - ho_z_rms) / arma_z_rms * 100 if arma_z_rms > 0 else 0
        roll_improvement = (arma_roll_rms - ho_roll_rms) / arma_roll_rms * 100 if arma_roll_rms > 0 else 0

        z_sign = '+' if z_improvement > 0 else ''
        roll_sign = '+' if roll_improvement > 0 else ''

        print(f"  {h}s horizon: Z {z_sign}{z_improvement:.1f}%, Roll {roll_sign}{roll_improvement:.1f}%")

    print()
    print("Key Insights:")
    print("  - Higher-order ARMA uses velocity and acceleration for kinematic consistency")
    print("  - Taylor expansion blended with ARMA for smooth predictions")
    print("  - 0.5s update interval balances computational cost and accuracy")
    print("  - Wave estimator still struggles with phase estimation")

    return arma_errors, ho_arma_errors, wave_errors


if __name__ == "__main__":
    run_comparison(sea_state_num=4, n_trials=10, update_interval=0.5)
