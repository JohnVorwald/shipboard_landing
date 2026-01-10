"""
Landing Visualization Video Generator

Creates an animated video showing:
1. 3D view of ship deck and UAV trajectory
2. Strip plots of x, y, z positions (past -5s, now, future +10s)
3. Strip plots of pitch, roll with forecast vs actual
4. Landing time highlighted

Usage:
    python landing_video.py [--output video.mp4] [--sea-state 3]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataclasses import dataclass
from typing import List, Dict, Optional
import sys
import os

sys.path.insert(0, '/mnt/sdcard/git_projects_backup/shipboard_landing')

from ship_motion.ddg_motion import DDGParams, SeaState, DDGMotionSimulator
from quad_dynamics.quadrotor import QuadrotorParams, QuadrotorState
from simulation.landing_sim import LandingSimulator, LandingConfig


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    fps: int = 20
    duration: float = 15.0
    dpi: int = 120

    # Strip plot window: now at center, -5s past, +10s future
    past_window: float = 5.0
    future_window: float = 10.0

    # Colors
    color_uav: str = '#2ecc71'
    color_deck: str = '#3498db'
    color_forecast: str = '#e74c3c'
    color_actual_future: str = '#9b59b6'
    color_history: str = '#34495e'
    color_landing: str = '#f1c40f'


class LandingVisualizer:
    """Generate landing visualization video."""

    def __init__(self, config: VisualizationConfig = None):
        self.vc = config if config is not None else VisualizationConfig()
        self.clear_data()

    def clear_data(self):
        """Clear stored data."""
        self.time_history = []
        self.uav_pos_history = []
        self.uav_att_history = []
        self.deck_pos_history = []
        self.deck_att_history = []
        self.forecasts = []
        self.landing_time = None

    def run_simulation(self, sim_config: LandingConfig = None) -> Dict:
        """Run simulation and collect data."""
        self.clear_data()

        if sim_config is None:
            sim_config = LandingConfig(
                sea_state=3,
                ship_speed_kts=12,
                approach_altitude=25,
                approach_distance=40,
                approach_speed=10.0
            )

        sim = LandingSimulator(sim_config, use_pmp=True)
        sim.reset()

        dt = 1.0 / self.vc.fps
        last_forecast = -1.0

        print("Running simulation...")
        while not sim.landed and sim.t < self.vc.duration:
            # Store state
            self.time_history.append(sim.t)
            self.uav_pos_history.append(sim.quad_state.pos.copy())
            self.uav_att_history.append(np.array([
                sim.quad_state.roll, sim.quad_state.pitch, sim.quad_state.yaw
            ]))

            deck = sim.ship_sim.get_motion(sim.t)
            self.deck_pos_history.append(deck['deck_position'].copy())
            self.deck_att_history.append(deck['attitude'].copy())

            # Store forecast every 0.5s
            if sim.t - last_forecast >= 0.5:
                preds = sim.predictor.predict(sim.t, self.vc.future_window, 0.1)
                if preds:
                    self.forecasts.append((sim.t, preds))
                last_forecast = sim.t

            sim.step(dt)

        if sim.landed:
            self.landing_time = sim.t
            print(f"Landing at t={self.landing_time:.2f}s")

        # Convert to numpy
        self.time_history = np.array(self.time_history)
        self.uav_pos_history = np.array(self.uav_pos_history)
        self.uav_att_history = np.array(self.uav_att_history)
        self.deck_pos_history = np.array(self.deck_pos_history)
        self.deck_att_history = np.array(self.deck_att_history)

        return {'landed': sim.landed, 'landing_time': self.landing_time}

    def get_forecast_at_time(self, t):
        """Get most recent forecast before time t."""
        for ft, preds in reversed(self.forecasts):
            if ft <= t:
                return ft, preds
        return None, None

    def create_figure(self):
        """Create figure layout."""
        fig = plt.figure(figsize=(14, 9))

        # 3D view (left, spans 2 rows)
        ax_3d = fig.add_subplot(2, 3, (1, 4), projection='3d')

        # Strip plots (right column)
        ax_x = fig.add_subplot(2, 3, 2)
        ax_y = fig.add_subplot(2, 3, 3)
        ax_z = fig.add_subplot(2, 3, 5)
        ax_att = fig.add_subplot(2, 3, 6)

        return fig, {'3d': ax_3d, 'x': ax_x, 'y': ax_y, 'z': ax_z, 'att': ax_att}

    def draw_deck(self, ax, pos, att):
        """Draw ship deck."""
        L, W = 20, 12
        corners = np.array([[-L/2,-W/2,0], [L/2,-W/2,0], [L/2,W/2,0], [-L/2,W/2,0]])

        r, p, y = att
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        R = np.array([[cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy],
                      [cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy],
                      [-sp, sr*cp, cr*cp]])

        corners_w = np.array([R @ c + pos for c in corners])
        deck = Poly3DCollection([corners_w], alpha=0.6)
        deck.set_facecolor(self.vc.color_deck)
        deck.set_edgecolor('black')
        ax.add_collection3d(deck)
        ax.scatter(*pos, c=self.vc.color_landing, s=100, marker='x', linewidths=2)

    def draw_uav(self, ax, pos, att):
        """Draw UAV."""
        size = 1.5
        r, p, y = att
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        R = np.array([[cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy],
                      [cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy],
                      [-sp, sr*cp, cr*cp]])

        arms = size * np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]])
        arms_w = np.array([R @ a + pos for a in arms])

        for i, arm in enumerate(arms_w):
            color = 'red' if i == 2 else 'black'
            ax.plot([pos[0], arm[0]], [pos[1], arm[1]], [pos[2], arm[2]],
                   color=color, linewidth=2)
        for arm in arms_w:
            ax.scatter(*arm, c=self.vc.color_uav, s=40)

    def update_frame(self, idx, fig, axes):
        """Update single frame."""
        if idx >= len(self.time_history):
            return

        t = self.time_history[idx]

        for ax in axes.values():
            ax.clear()

        # Current state
        uav_pos = self.uav_pos_history[idx]
        uav_att = self.uav_att_history[idx]
        deck_pos = self.deck_pos_history[idx]
        deck_att = self.deck_att_history[idx]

        # === 3D VIEW ===
        ax3 = axes['3d']
        self.draw_deck(ax3, deck_pos, deck_att)
        self.draw_uav(ax3, uav_pos, uav_att)

        # UAV trajectory
        mask = (self.time_history <= t) & (self.time_history >= t - self.vc.past_window)
        if np.any(mask):
            traj = self.uav_pos_history[mask]
            ax3.plot(traj[:,0], traj[:,1], traj[:,2], color=self.vc.color_history,
                    linewidth=1, alpha=0.5)

        cx = (uav_pos[0] + deck_pos[0]) / 2
        cy = (uav_pos[1] + deck_pos[1]) / 2
        ax3.set_xlim(cx-50, cx+50)
        ax3.set_ylim(cy-30, cy+30)
        ax3.set_zlim(-45, 5)
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        ax3.set_title(f'Shipboard Landing  t = {t:.2f}s', fontsize=11, fontweight='bold')
        ax3.invert_zaxis()

        # === STRIP PLOTS (dynamic window: now at center) ===
        t_min = t - self.vc.past_window
        t_max = t + self.vc.future_window

        # Get forecast
        _, preds = self.get_forecast_at_time(t)
        fc_t, fc_pos, fc_att = [], [], []
        if preds:
            for p in preds:
                if p.t > t:
                    fc_t.append(p.t)
                    fc_pos.append(p.position)
                    fc_att.append(p.attitude)
        fc_t = np.array(fc_t) if fc_t else np.array([])
        fc_pos = np.array(fc_pos) if fc_pos else np.zeros((0,3))
        fc_att = np.array(fc_att) if fc_att else np.zeros((0,3))

        # Actual future (what really happens)
        fut_mask = (self.time_history > t) & (self.time_history <= t_max)
        fut_t = self.time_history[fut_mask]
        fut_deck = self.deck_pos_history[fut_mask]
        fut_att = self.deck_att_history[fut_mask]

        # History
        hist_mask = (self.time_history <= t) & (self.time_history >= t_min)
        hist_t = self.time_history[hist_mask]
        hist_deck = self.deck_pos_history[hist_mask]
        hist_uav = self.uav_pos_history[hist_mask]
        hist_att = self.deck_att_history[hist_mask]

        def plot_strip(ax, ylabel, title, hist_y_deck, hist_y_uav, fc_y, fut_y, ylim=None, error_scale=0.5):
            # History
            ax.plot(hist_t, hist_y_deck, color=self.vc.color_deck, lw=2, label='Deck')
            if hist_y_uav is not None:
                ax.plot(hist_t, hist_y_uav, color=self.vc.color_uav, lw=2, label='UAV')
            # Forecast with error band
            if len(fc_t) > 0 and fc_y is not None:
                ax.plot(fc_t, fc_y, '--', color=self.vc.color_forecast, lw=2, label='Forecast')
                # Error band grows with time from forecast origin
                dt_from_now = fc_t - t
                error = error_scale * dt_from_now  # Error grows linearly with time
                ax.fill_between(fc_t, fc_y - error, fc_y + error,
                               color=self.vc.color_forecast, alpha=0.2)
            # Actual future
            if len(fut_t) > 0 and fut_y is not None:
                ax.plot(fut_t, fut_y, '-', color=self.vc.color_actual_future, lw=1.5,
                       alpha=0.8, label='Actual')
            # Now line
            ax.axvline(t, color='black', lw=1.5, alpha=0.7)
            ax.axvspan(t_min, t, alpha=0.08, color='gray')
            # Landing
            if self.landing_time and t_min <= self.landing_time <= t_max:
                ax.axvline(self.landing_time, color=self.vc.color_landing, lw=2,
                          ls='--', label='Landing')
            ax.set_xlim(t_min, t_max)
            if ylim:
                ax.set_ylim(ylim)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=7)

        # X position
        plot_strip(axes['x'], 'X (m)', 'Position X (Forward)',
                  hist_deck[:,0], hist_uav[:,0],
                  fc_pos[:,0] if len(fc_pos) else None,
                  fut_deck[:,0] if len(fut_deck) else None)

        # Y position
        plot_strip(axes['y'], 'Y (m)', 'Position Y (Starboard)',
                  hist_deck[:,1], hist_uav[:,1],
                  fc_pos[:,1] if len(fc_pos) else None,
                  fut_deck[:,1] if len(fut_deck) else None)

        # Z (altitude, flip sign)
        plot_strip(axes['z'], 'Alt (m)', 'Altitude',
                  -hist_deck[:,2], -hist_uav[:,2],
                  -fc_pos[:,2] if len(fc_pos) else None,
                  -fut_deck[:,2] if len(fut_deck) else None)
        axes['z'].set_xlabel('Time (s)')

        # Roll/Pitch (degrees) with error bands
        ax_att = axes['att']
        ax_att.plot(hist_t, np.degrees(hist_att[:,0]), color=self.vc.color_deck, lw=2, label='Roll')
        ax_att.plot(hist_t, np.degrees(hist_att[:,1]), '--', color=self.vc.color_deck, lw=2, label='Pitch')
        if len(fc_t) > 0:
            fc_roll = np.degrees(fc_att[:,0])
            fc_pitch = np.degrees(fc_att[:,1])
            ax_att.plot(fc_t, fc_roll, '--', color=self.vc.color_forecast, lw=2, label='Forecast')
            ax_att.plot(fc_t, fc_pitch, ':', color=self.vc.color_forecast, lw=2)
            # Error band (0.3 deg/s growth)
            dt_from_now = fc_t - t
            att_error = 0.3 * dt_from_now
            ax_att.fill_between(fc_t, fc_roll - att_error, fc_roll + att_error,
                               color=self.vc.color_forecast, alpha=0.15)
            ax_att.fill_between(fc_t, fc_pitch - att_error, fc_pitch + att_error,
                               color=self.vc.color_forecast, alpha=0.15)
        if len(fut_t) > 0:
            ax_att.plot(fut_t, np.degrees(fut_att[:,0]), '-', color=self.vc.color_actual_future, lw=1.5, alpha=0.8, label='Actual')
            ax_att.plot(fut_t, np.degrees(fut_att[:,1]), '-', color=self.vc.color_actual_future, lw=1.5, alpha=0.8)
        ax_att.axvline(t, color='black', lw=1.5, alpha=0.7)
        ax_att.axvspan(t_min, t, alpha=0.08, color='gray')
        if self.landing_time and t_min <= self.landing_time <= t_max:
            ax_att.axvline(self.landing_time, color=self.vc.color_landing, lw=2, ls='--')
        ax_att.set_xlim(t_min, t_max)
        ax_att.set_ylim(-12, 12)
        ax_att.set_xlabel('Sim Time (s)')
        ax_att.set_ylabel('Angle (deg)')
        ax_att.set_title('Deck Roll & Pitch', fontsize=10)
        ax_att.legend(loc='upper right', fontsize=7)
        ax_att.grid(True, alpha=0.3)

        plt.tight_layout()

    def generate_video(self, output_path: str = '/tmp/landing_video.mp4'):
        """Generate video."""
        if len(self.time_history) == 0:
            print("No data. Run simulation first.")
            return

        n_frames = len(self.time_history)
        print(f"Generating video: {n_frames} frames...")

        fig, axes = self.create_figure()

        def update(frame):
            self.update_frame(frame, fig, axes)
            if frame % 20 == 0:
                print(f"  Frame {frame}/{n_frames}")

        anim = FuncAnimation(fig, update, frames=n_frames,
                            interval=1000/self.vc.fps, blit=False)

        print(f"Saving to {output_path}...")
        try:
            writer = FFMpegWriter(fps=self.vc.fps, bitrate=3000)
            anim.save(output_path, writer=writer, dpi=self.vc.dpi)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"FFmpeg error: {e}")
            gif_path = output_path.replace('.mp4', '.gif')
            print(f"Trying GIF: {gif_path}")
            try:
                anim.save(gif_path, writer='pillow', fps=10)
                print(f"Saved: {gif_path}")
            except Exception as e2:
                print(f"GIF error: {e2}")
                self.save_frames(fig, axes, '/tmp/landing_frames')

        plt.close(fig)

    def save_frames(self, fig, axes, prefix: str, n: int = 8):
        """Save key frames as images."""
        os.makedirs(prefix, exist_ok=True)
        indices = np.linspace(0, len(self.time_history)-1, n, dtype=int)
        for i, idx in enumerate(indices):
            self.update_frame(idx, fig, axes)
            path = f"{prefix}/frame_{i:02d}_t{self.time_history[idx]:.1f}s.png"
            fig.savefig(path, dpi=self.vc.dpi, bbox_inches='tight')
            print(f"Saved {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='/tmp/landing_video.mp4')
    parser.add_argument('--sea-state', type=int, default=3)
    parser.add_argument('--fps', type=int, default=20)
    args = parser.parse_args()

    vc = VisualizationConfig(fps=args.fps)
    sc = LandingConfig(sea_state=args.sea_state, ship_speed_kts=12,
                       approach_altitude=25, approach_distance=40,
                       approach_speed=10.0)

    viz = LandingVisualizer(vc)
    result = viz.run_simulation(sc)
    print(f"Result: {'LANDED' if result['landed'] else 'TIMEOUT'}")
    viz.generate_video(args.output)


if __name__ == '__main__':
    main()
