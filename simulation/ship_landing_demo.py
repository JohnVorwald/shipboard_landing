#!/usr/bin/env python3
"""
Ship Landing Demo with Video Recording
- Launches Gazebo and ArduPilot SITL in visible terminal windows
- Arms quadcopter with retries
- Takes off and flies a pattern
- Records video with telemetry overlay (range, azimuth, elevation)
- Lands on helipad
- Cleans up all processes on exit

Usage:
    python3 ship_landing_demo.py              # Full GUI mode
    python3 ship_landing_demo.py --headless   # No GUI (no screen recording)
"""

import sys
# Force unbuffered output for better debugging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import time
import subprocess
import threading
import os
import sys
import signal
import atexit
import math
from datetime import datetime

# Try to import pymavlink
try:
    from pymavlink import mavutil
except ImportError:
    print("Installing pymavlink...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pymavlink", "-q"])
    from pymavlink import mavutil

# Try to import OpenCV for video recording
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    print("OpenCV not found - installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python", "-q"])
    try:
        import cv2
        import numpy as np
        HAS_OPENCV = True
    except ImportError:
        print("OpenCV installation failed - video recording disabled")
        HAS_OPENCV = False


def detect_display():
    """Auto-detect the correct X display"""
    # Check if DISPLAY is already set and valid
    current_display = os.environ.get('DISPLAY', '')

    # Try to detect active displays
    possible_displays = []

    # Check for remote desktop (typically :10)
    try:
        result = subprocess.run(['w', '-h'], capture_output=True, text=True, timeout=2)
        for line in result.stdout.split('\n'):
            if ':10' in line or ':11' in line:
                possible_displays.append(':10.0')
                break
    except:
        pass

    # Check for local displays
    for display in [':0', ':1', ':0.0', ':1.0']:
        x_lock = f'/tmp/.X{display.split(":")[1].split(".")[0]}-lock'
        if os.path.exists(x_lock):
            possible_displays.append(display if '.' in display else f'{display}.0')

    # Try to validate displays by checking if we can connect
    for display in possible_displays:
        try:
            env = os.environ.copy()
            env['DISPLAY'] = display
            result = subprocess.run(
                ['xdpyinfo'], env=env, capture_output=True, timeout=2
            )
            if result.returncode == 0:
                print(f"Detected active display: {display}")
                return display
        except:
            continue

    # Fall back to current or default
    if current_display:
        print(f"Using current DISPLAY: {current_display}")
        return current_display

    # Default based on common setups
    default = ':10.0' if os.path.exists('/tmp/.X10-lock') else ':0.0'
    print(f"Using default display: {default}")
    return default


def create_gazebo_gui_config():
    """Create Gazebo GUI config with camera displays"""
    config_dir = os.path.expanduser('~/.gz/sim/gui')
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, 'ship_landing_gui.config')

    config_content = '''<?xml version="1.0" ?>
<window>
  <width>1600</width>
  <height>900</height>
  <style
    material_theme="Light"
    material_primary="Blue"
    material_accent="DeepOrange"
    toolbar_color_light="#f3f3f3"
    toolbar_text_color_light="#111111"
    toolbar_color_dark="#414141"
    toolbar_text_color_dark="#f3f3f3"
    plugin_toolbar_color_light="#bbdefb"
    plugin_toolbar_text_color_light="#111111"
    plugin_toolbar_color_dark="#607d8b"
    plugin_toolbar_text_color_dark="#eeeeee"
  />
  <menus>
    <drawer default="false"/>
  </menus>

  <!-- 3D scene -->
  <plugin filename="MinimalScene" name="3D View">
    <gz-gui>
      <title>3D View</title>
      <property type="bool" key="showTitleBar">true</property>
      <property type="string" key="state">docked</property>
    </gz-gui>
    <engine>ogre2</engine>
    <scene>scene</scene>
    <ambient_light>0.6 0.6 0.7</ambient_light>
    <background_color>0.4 0.5 0.6</background_color>
    <camera_pose>90 -25 20 0 0.35 2.5</camera_pose>
  </plugin>

  <plugin filename="GzSceneManager" name="Scene Manager"/>
  <plugin filename="InteractiveViewControl" name="Interactive View Control"/>
  <plugin filename="CameraTracking" name="Camera Tracking"/>

  <!-- World control -->
  <plugin filename="WorldControl" name="World control">
    <gz-gui>
      <title>World control</title>
      <property type="bool" key="showTitleBar">false</property>
      <property type="bool" key="resizable">false</property>
      <property type="double" key="height">72</property>
      <property type="double" key="width">150</property>
      <property type="double" key="z">1</property>
      <property type="string" key="state">floating</property>
      <anchors target="3D View">
        <line own="left" target="left"/>
        <line own="bottom" target="bottom"/>
      </anchors>
    </gz-gui>
    <play_pause>true</play_pause>
    <step>true</step>
    <start_paused>false</start_paused>
  </plugin>

  <!-- World stats -->
  <plugin filename="WorldStats" name="World stats">
    <gz-gui>
      <title>World stats</title>
      <property type="bool" key="showTitleBar">false</property>
      <property type="bool" key="resizable">false</property>
      <property type="double" key="height">110</property>
      <property type="double" key="width">290</property>
      <property type="double" key="z">1</property>
      <property type="string" key="state">floating</property>
      <anchors target="3D View">
        <line own="right" target="right"/>
        <line own="bottom" target="bottom"/>
      </anchors>
    </gz-gui>
    <sim_time>true</sim_time>
    <real_time>true</real_time>
    <real_time_factor>true</real_time_factor>
    <iterations>false</iterations>
  </plugin>

  <!-- Landing Camera Display -->
  <plugin filename="ImageDisplay" name="Landing Camera">
    <gz-gui>
      <title>Landing Camera</title>
      <property type="bool" key="showTitleBar">true</property>
      <property type="string" key="state">floating</property>
      <property type="double" key="width">320</property>
      <property type="double" key="height">260</property>
      <property type="double" key="x">10</property>
      <property type="double" key="y">10</property>
    </gz-gui>
    <topic>/landing_camera/image</topic>
  </plugin>

  <!-- Gimbal Camera Display -->
  <plugin filename="ImageDisplay" name="Gimbal Camera">
    <gz-gui>
      <title>Gimbal Camera</title>
      <property type="bool" key="showTitleBar">true</property>
      <property type="string" key="state">floating</property>
      <property type="double" key="width">320</property>
      <property type="double" key="height">260</property>
      <property type="double" key="x">10</property>
      <property type="double" key="y">280</property>
    </gz-gui>
    <topic>/world/ship_landing_ardupilot/model/iris/model/gimbal/link/pitch_link/sensor/camera/image</topic>
  </plugin>

  <!-- Entity tree -->
  <plugin filename="EntityTree" name="Entity tree">
    <gz-gui>
      <title>Entity tree</title>
      <property type="bool" key="showTitleBar">true</property>
      <property type="string" key="state">floating</property>
      <property type="double" key="width">250</property>
      <property type="double" key="height">300</property>
      <property type="double" key="x">1340</property>
      <property type="double" key="y">10</property>
    </gz-gui>
  </plugin>

</window>
'''

    with open(config_file, 'w') as f:
        f.write(config_content)

    print(f"Created Gazebo GUI config: {config_file}")
    return config_file


class TelemetryState:
    """Shared telemetry state for video overlay"""
    def __init__(self):
        self.x = 0.0  # North
        self.y = 0.0  # East
        self.z = 0.0  # Down (negative = up)
        self.range = 0.0
        self.azimuth = 0.0
        self.elevation = 0.0
        self.altitude = 0.0
        self.flight_phase = "Initializing"
        self.lock = threading.Lock()

    def update(self, x, y, z):
        """Update position and calculate range/az/el from home"""
        with self.lock:
            self.x = x
            self.y = y
            self.z = z
            self.altitude = -z  # Convert down to up

            # Range (horizontal distance from home)
            self.range = math.sqrt(x**2 + y**2)

            # Azimuth (bearing from home, 0=North, 90=East)
            self.azimuth = math.degrees(math.atan2(y, x))
            if self.azimuth < 0:
                self.azimuth += 360

            # Elevation angle (angle from horizontal)
            horiz_dist = math.sqrt(x**2 + y**2)
            if horiz_dist > 0.1:
                self.elevation = math.degrees(math.atan2(-z, horiz_dist))
            else:
                self.elevation = 90.0 if z < 0 else -90.0

    def set_phase(self, phase):
        with self.lock:
            self.flight_phase = phase

    def get_overlay_text(self):
        """Get formatted text for video overlay"""
        with self.lock:
            return [
                f"Phase: {self.flight_phase}",
                f"Position: N={self.x:.1f}m E={self.y:.1f}m",
                f"Altitude: {self.altitude:.1f}m",
                f"Range: {self.range:.1f}m",
                f"Azimuth: {self.azimuth:.0f} deg",
                f"Elevation: {self.elevation:.1f} deg",
            ]


# Global telemetry state
telemetry = TelemetryState()


class ProcessManager:
    """Manages Gazebo and ArduPilot processes in visible terminals"""

    def __init__(self):
        self.gazebo_process = None
        self.ardupilot_process = None
        self.terminal_pids = []
        self.display = None  # Will be set by launch_gazebo

        # Register cleanup on exit
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        print("\nReceived interrupt signal, cleaning up...")
        self.cleanup()
        sys.exit(1)

    def _find_terminal(self):
        """Find available terminal emulator"""
        terminals = ['gnome-terminal', 'xterm', 'konsole', 'xfce4-terminal']
        for term in terminals:
            try:
                subprocess.run(['which', term], capture_output=True, check=True)
                return term
            except subprocess.CalledProcessError:
                continue
        return None

    def launch_gazebo(self, world_file, headless=False, gui_config=None):
        """Launch Gazebo with the ship world - directly with proper environment"""
        print("Launching Gazebo...")

        # Auto-detect display
        display = detect_display()
        self.display = display

        # Environment variables for Gazebo - CRITICAL for plugin loading
        # Prepend custom paths to existing ones (don't replace)
        env = os.environ.copy()

        custom_resource_path = '/home/john/gz_ws/src/ardupilot_gazebo/models:/home/john/gz_ws/src/ardupilot_gazebo/worlds:/home/john/github/shipboard_landing/gazebo/models'
        custom_plugin_path = '/home/john/gz_ws/src/ardupilot_gazebo/build'

        if 'GZ_SIM_RESOURCE_PATH' in env:
            env['GZ_SIM_RESOURCE_PATH'] = custom_resource_path + ':' + env['GZ_SIM_RESOURCE_PATH']
        else:
            env['GZ_SIM_RESOURCE_PATH'] = custom_resource_path

        if 'GZ_SIM_SYSTEM_PLUGIN_PATH' in env:
            env['GZ_SIM_SYSTEM_PLUGIN_PATH'] = custom_plugin_path + ':' + env['GZ_SIM_SYSTEM_PLUGIN_PATH']
        else:
            env['GZ_SIM_SYSTEM_PLUGIN_PATH'] = custom_plugin_path

        env['GZ_IP'] = '127.0.0.1'
        env['GZ_PARTITION'] = 'gazebo_default'
        env['LIBGL_DRI3_DISABLE'] = '1'
        env['QT_QPA_PLATFORM'] = 'xcb'
        env['DISPLAY'] = display

        print(f"  Display: {display}")
        print(f"  World: {world_file}")
        print(f"  GZ_SIM_SYSTEM_PLUGIN_PATH: {env['GZ_SIM_SYSTEM_PLUGIN_PATH']}")

        # Build command
        gz_args = ['gz', 'sim', '-r']
        if headless:
            gz_args.append('-s')  # Server only, no GUI
        if gui_config and not headless:
            gz_args.extend(['--gui-config', gui_config])
        gz_args.append(world_file)

        print(f"  Command: {' '.join(gz_args)}")

        # Launch Gazebo directly with proper environment
        # This ensures GZ_SIM_SYSTEM_PLUGIN_PATH is set correctly for ArduPilotPlugin
        self.gazebo_process = subprocess.Popen(
            gz_args,
            env=env,
            preexec_fn=os.setsid
        )

        # Wait for Gazebo to initialize
        # Note: ArduPilotPlugin uses UDP, not TCP on port 9002
        # The connection is established when ArduPilot SITL starts
        print("Waiting for Gazebo to initialize (15 seconds)...")
        time.sleep(15)
        print("Gazebo started")
        return True

    def launch_ardupilot(self, with_mavproxy=False):
        """Launch ArduPilot SITL directly (no terminal window)"""
        print("Launching ArduPilot SITL...")

        ardupilot_dir = '/home/john/ardupilot'

        # Launch sim_vehicle.py directly without terminal
        # Using --no-mavproxy since we'll connect directly via TCP
        cmd = [
            '/home/john/ardupilot/Tools/autotest/sim_vehicle.py',
            '-v', 'ArduCopter',
            '-f', 'JSON',
            '--model', 'JSON',
            '--no-mavproxy'
        ]

        print(f"  Command: {' '.join(cmd)}")

        self.ardupilot_process = subprocess.Popen(
            cmd,
            cwd=ardupilot_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )

        # Wait for ArduPilot to initialize (needs time for build + startup + connect to Gazebo)
        print("Waiting for ArduPilot SITL to initialize...")
        print("  (ArduPilot may need to build, this can take 30+ seconds)")
        print("  DEBUG: Monitoring ArduPilot output for 'JSON received' message")

        # Wait for ArduPilot to open its TCP ports AND receive JSON from Gazebo
        import socket
        port_ready = False
        json_ready = False

        for i in range(120):  # Wait up to 120 seconds
            # Check TCP port
            if not port_ready:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(('127.0.0.1', 5760))
                    sock.close()
                    if result == 0:
                        print(f"  DEBUG: TCP port 5760 ready at {i+1}s")
                        port_ready = True
                except:
                    pass

            # Check ArduPilot output for JSON connection
            if port_ready and self.ardupilot_process:
                try:
                    # Read any available output
                    import select
                    if hasattr(self.ardupilot_process, 'stdout') and self.ardupilot_process.stdout:
                        readable, _, _ = select.select([self.ardupilot_process.stdout], [], [], 0.1)
                        if readable:
                            line = self.ardupilot_process.stdout.readline()
                            if line:
                                line_str = line.decode('utf-8', errors='ignore').strip()
                                if 'JSON' in line_str:
                                    print(f"  DEBUG: {line_str}")
                                if 'JSON received' in line_str:
                                    json_ready = True
                                    print(f"  DEBUG: JSON connection established at {i+1}s")
                except:
                    pass

            # Both conditions met - give extra time for stabilization
            if port_ready and json_ready:
                print(f"  ArduPilot ready after {i+1} seconds (port + JSON)")
                time.sleep(5)  # Extra stabilization for heartbeat
                print("ArduPilot SITL started and connected to Gazebo")
                return True

            # Port ready but no JSON yet - still wait
            if port_ready and i >= 30 and i % 10 == 0:
                print(f"  DEBUG: Port ready, waiting for JSON sync... ({i}/120)")

            if not port_ready and i % 15 == 0:
                print(f"  Waiting for ArduPilot build/startup... ({i}/120)")

            time.sleep(1)

        # Timeout - return what we have
        if port_ready:
            print(f"  WARNING: Port ready but JSON sync not confirmed")
            print(f"  Proceeding anyway - connection may fail")
            time.sleep(5)
            return True

        print("  ERROR: ArduPilot ports not detected after 120 seconds")
        return False

    def cleanup(self):
        """Clean up all processes"""
        print("\nCleaning up processes...")

        # Kill terminal processes we started
        for name, proc in [('Gazebo', self.gazebo_process), ('ArduPilot', self.ardupilot_process)]:
            if proc and proc.poll() is None:
                print(f"  Stopping {name} terminal (pid {proc.pid})...")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=3)
                except Exception as e:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except:
                        pass

        # Kill any remaining processes (more comprehensive)
        print("Cleaning up remaining processes...")
        kill_patterns = [
            'gz sim', 'gzserver', 'gzclient', 'gz-sim',
            'arducopter', 'mavproxy', 'sim_vehicle',
            'ruby.*sim_vehicle'
        ]
        for pattern in kill_patterns:
            subprocess.run(['pkill', '-9', '-f', pattern], capture_output=True)

        # Also use killall for simple names
        for proc_name in ['gz', 'arducopter', 'ruby']:
            subprocess.run(['killall', '-9', proc_name], capture_output=True)

        # Try xdotool to close windows by name (wmctrl may not be installed)
        try:
            for title in ['Gazebo', 'ArduPilot', 'ArduCopter']:
                subprocess.run(['xdotool', 'search', '--name', title, 'windowclose'],
                             capture_output=True, timeout=2)
        except:
            pass  # xdotool may not be installed

        # Small delay to ensure cleanup
        time.sleep(1)
        print("Cleanup complete")


class CameraRecorder:
    """Records video with telemetry overlay"""

    def __init__(self, name, output_file, width=640, height=480, fps=15):
        self.name = name
        self.output_file = output_file
        self.width = width
        self.height = height
        self.fps = fps
        self.recording = False
        self.writer = None
        self.thread = None
        self.frame_count = 0

    def start(self):
        """Start recording with telemetry overlay"""
        if not HAS_OPENCV:
            print(f"OpenCV not available - skipping {self.name} recording")
            return

        self.recording = True
        print(f"Starting {self.name} recording -> {self.output_file}")

        self.thread = threading.Thread(target=self._record_loop)
        self.thread.daemon = True
        self.thread.start()

    def _record_loop(self):
        """Record frames with telemetry overlay"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_file, fourcc, self.fps,
            (self.width, self.height)
        )

        frame_interval = 1.0 / self.fps
        last_frame_time = time.time()
        start_time = time.time()

        while self.recording:
            current_time = time.time()
            if current_time - last_frame_time >= frame_interval:
                frame = self._create_telemetry_frame(current_time - start_time)
                self.writer.write(frame)
                self.frame_count += 1
                last_frame_time = current_time
            time.sleep(0.01)

        if self.writer:
            self.writer.release()

    def _create_telemetry_frame(self, elapsed_time):
        """Create a frame with telemetry HUD overlay"""
        # Dark background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (20, 20, 30)  # Dark blue-gray

        # Title
        cv2.putText(frame, self.name, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (self.width - 120, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # Elapsed time
        cv2.putText(frame, f"T+{elapsed_time:.1f}s", (self.width - 120, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Telemetry data
        overlay_lines = telemetry.get_overlay_text()
        y_start = 100
        for i, line in enumerate(overlay_lines):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)  # Phase in green
            cv2.putText(frame, line, (30, y_start + i * 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw compass rose for azimuth
        self._draw_compass(frame, 520, 350, 80)

        # Draw altitude bar
        self._draw_altitude_bar(frame, 580, 100, 200)

        # Recording indicator
        if int(elapsed_time * 2) % 2 == 0:
            cv2.circle(frame, (30, self.height - 30), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (45, self.height - 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (self.width - 150, self.height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        return frame

    def _draw_compass(self, frame, cx, cy, radius):
        """Draw a compass showing azimuth"""
        # Compass circle
        cv2.circle(frame, (cx, cy), radius, (100, 100, 100), 2)

        # Cardinal directions
        cv2.putText(frame, "N", (cx - 5, cy - radius - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "S", (cx - 5, cy + radius + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "E", (cx + radius + 5, cy + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "W", (cx - radius - 20, cy + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Azimuth pointer (from center toward target)
        with telemetry.lock:
            az_rad = math.radians(telemetry.azimuth - 90)  # Adjust for compass orientation
            range_val = telemetry.range

        # Scale range to fit in compass (max 50m = full radius)
        scaled_range = min(range_val / 50.0, 1.0) * (radius - 10)

        end_x = int(cx + scaled_range * math.cos(az_rad))
        end_y = int(cy + scaled_range * math.sin(az_rad))

        cv2.line(frame, (cx, cy), (end_x, end_y), (0, 255, 255), 2)
        cv2.circle(frame, (end_x, end_y), 5, (0, 255, 255), -1)

        # Home marker
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    def _draw_altitude_bar(self, frame, x, y, height):
        """Draw altitude bar indicator"""
        bar_width = 30
        max_alt = 30.0  # Max altitude for display

        # Background bar
        cv2.rectangle(frame, (x, y), (x + bar_width, y + height), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y), (x + bar_width, y + height), (100, 100, 100), 1)

        # Altitude fill
        with telemetry.lock:
            alt = telemetry.altitude

        fill_height = int(min(alt / max_alt, 1.0) * height)
        if fill_height > 0:
            cv2.rectangle(frame,
                         (x + 2, y + height - fill_height),
                         (x + bar_width - 2, y + height - 2),
                         (0, 200, 255), -1)

        # Label
        cv2.putText(frame, "ALT", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"{alt:.0f}m", (x - 5, y + height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def stop(self):
        """Stop recording"""
        self.recording = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.writer:
            self.writer.release()
        print(f"  {self.name}: Saved {self.frame_count} frames to {self.output_file}")


class ScreenRecorder:
    """Records the full screen (Gazebo + terminals) using ffmpeg"""

    def __init__(self, output_file, fps=15):
        self.output_file = output_file
        self.fps = fps
        self.process = None
        self.recording = False

    def start(self, display=None):
        """Start screen recording"""
        if display is None:
            display = os.environ.get('DISPLAY', ':10.0')

        print(f"Starting screen recording -> {self.output_file}")
        print(f"  Capturing display: {display}")

        # Use ffmpeg to record the entire screen
        cmd = [
            'ffmpeg', '-y',
            '-f', 'x11grab',
            '-framerate', str(self.fps),
            '-video_size', '1920x1080',  # Adjust to your screen size
            '-i', display,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            self.output_file
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.PIPE
            )
            self.recording = True
            print("  Screen recording started (captures Gazebo + terminal windows)")
        except FileNotFoundError:
            print("  ffmpeg not found - screen recording disabled")
            print("  Install with: sudo apt install ffmpeg")

    def stop(self):
        """Stop screen recording"""
        if self.process and self.recording:
            try:
                self.process.stdin.write(b'q')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except:
                    self.process.kill()

            self.recording = False
            print(f"  Screen recording saved: {self.output_file}")


class ShipLandingController:
    """Controls quadcopter for ship landing with MAVLink"""

    def __init__(self):
        self.master = None
        self.armed = False
        self.telemetry_thread = None
        self.running = False

    def connect(self, timeout=120):
        """Connect to ArduPilot via TCP port 5760 with debugging"""
        print("Connecting to ArduPilot on tcp:127.0.0.1:5760...")
        print("  DEBUG: Will retry with increasing delays until heartbeat received")

        start_time = time.time()
        attempt = 0

        while time.time() - start_time < timeout:
            attempt += 1
            elapsed = int(time.time() - start_time)

            try:
                print(f"  DEBUG: Connection attempt {attempt} at {elapsed}s...")
                self.master = mavutil.mavlink_connection('tcp:127.0.0.1:5760', source_system=255)

                # Wait for heartbeat with longer timeout
                print(f"  DEBUG: Waiting for heartbeat (15s timeout)...")
                msg = self.master.wait_heartbeat(timeout=15)

                if msg:
                    print(f"  DEBUG: Got heartbeat from system {self.master.target_system}")
                    if self.master.target_system == 1:
                        print(f"Connected to ArduPilot system 1 (attempt {attempt}, {elapsed}s)")

                        # Start telemetry thread
                        self.running = True
                        self.telemetry_thread = threading.Thread(target=self._telemetry_loop)
                        self.telemetry_thread.daemon = True
                        self.telemetry_thread.start()

                        return True
                    else:
                        print(f"  DEBUG: Wrong system ({self.master.target_system}), retrying...")
                else:
                    print(f"  DEBUG: No heartbeat received (attempt {attempt})")

            except Exception as e:
                print(f"  DEBUG: Connection error at {elapsed}s: {e}")

            # Increasing delay between attempts
            delay = min(5, 2 + attempt // 3)
            print(f"  DEBUG: Waiting {delay}s before next attempt...")
            time.sleep(delay)

        print(f"Failed to connect after {attempt} attempts ({timeout}s timeout)")
        return False

    def _telemetry_loop(self):
        """Background thread to update telemetry state"""
        while self.running:
            try:
                msg = self.master.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=0.5)
                if msg:
                    telemetry.update(msg.x, msg.y, msg.z)
            except:
                pass

    def wait_for_ekf(self, timeout=90):
        """Wait for EKF to be healthy"""
        print("Waiting for EKF to initialize...")
        telemetry.set_phase("EKF Init")
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Try to get EKF status
            msg = self.master.recv_match(type='EKF_STATUS_REPORT', blocking=True, timeout=3)
            if msg:
                flags = msg.flags
                # Check for healthy EKF (attitude + velocity good)
                if flags >= 128:  # Usually 831 when healthy
                    print(f"  EKF healthy (flags: {flags})")
                    return True
                print(f"  EKF initializing (flags: {flags})...")
            else:
                # No EKF message yet, check heartbeat
                hb = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
                if hb:
                    print(f"  Waiting for EKF... (system {hb.system_status})")
            time.sleep(2)

        print("EKF initialization timeout")
        return False

    def set_mode(self, mode):
        """Set flight mode"""
        mode_mapping = {
            'STABILIZE': 0, 'ACRO': 1, 'ALT_HOLD': 2, 'AUTO': 3,
            'GUIDED': 4, 'LOITER': 5, 'RTL': 6, 'CIRCLE': 7,
            'LAND': 9, 'DRIFT': 11, 'SPORT': 13, 'POSHOLD': 16,
        }
        mode_id = mode_mapping.get(mode.upper(), mode)
        self.master.set_mode(mode_id)
        time.sleep(0.5)
        print(f"Set mode: {mode}")

    def arm(self, max_attempts=10, force=True):
        """Arm the quadcopter with retries"""
        print(f"Arming (max {max_attempts} attempts)...")
        telemetry.set_phase("Arming")

        for attempt in range(1, max_attempts + 1):
            print(f"  Attempt {attempt}/{max_attempts}...")

            if force:
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0, 1, 21196, 0, 0, 0, 0, 0
                )
            else:
                self.master.arducopter_arm()

            time.sleep(1)

            msg = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=2)
            if msg and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                print(f"  Armed successfully on attempt {attempt}")
                self.armed = True
                return True

            ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=1)
            if ack and ack.result != 0:
                print(f"  Arm failed with result: {ack.result}")

        print("Failed to arm after all attempts")
        return False

    def takeoff(self, altitude=10):
        """Takeoff to specified altitude"""
        print(f"Taking off to {altitude}m...")
        telemetry.set_phase("Takeoff")

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude
        )

        start = time.time()
        while time.time() - start < 30:
            pos = self.get_position()
            if pos and pos['relative_alt'] > altitude - 1:
                print(f"  Reached {pos['relative_alt']:.1f}m")
                return True
            time.sleep(0.5)

        print("Takeoff timeout")
        return False

    def land(self):
        """Land the quadcopter"""
        print("Landing...")
        telemetry.set_phase("Landing")
        self.set_mode('LAND')

        start = time.time()
        while time.time() - start < 60:
            pos = self.get_position()
            if pos and pos['relative_alt'] < 0.5:
                print("  Landed")
                telemetry.set_phase("Landed")
                time.sleep(2)
                return True
            time.sleep(0.5)

        return False

    def goto_position_ned(self, north, east, down, phase_name=None):
        """Go to position relative to home (NED frame)"""
        if phase_name:
            telemetry.set_phase(phase_name)
        print(f"Going to NED: N={north}m, E={east}m, Alt={-down}m")

        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111111000,
            north, east, down,
            0, 0, 0, 0, 0, 0, 0, 0
        )

    def get_position(self):
        """Get current position"""
        msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
        if msg:
            return {
                'lat': msg.lat / 1e7,
                'lon': msg.lon / 1e7,
                'alt': msg.alt / 1000,
                'relative_alt': msg.relative_alt / 1000,
            }
        return None

    def get_local_position(self):
        """Get local position relative to home"""
        msg = self.master.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=2)
        if msg:
            return {'x': msg.x, 'y': msg.y, 'z': msg.z}
        return None

    def wait_for_position(self, target_n, target_e, target_d, tolerance=2, timeout=60):
        """Wait until reaching target position"""
        start = time.time()
        while time.time() - start < timeout:
            pos = self.get_local_position()
            if pos:
                dist = ((pos['x'] - target_n)**2 +
                        (pos['y'] - target_e)**2 +
                        (pos['z'] - target_d)**2)**0.5
                print(f"  Pos: N={pos['x']:.1f} E={pos['y']:.1f} Alt={-pos['z']:.1f}m | Dist: {dist:.1f}m")
                if dist < tolerance:
                    return True
            time.sleep(1)
        return False

    def stop(self):
        """Stop the controller"""
        self.running = False
        if self.telemetry_thread:
            self.telemetry_thread.join(timeout=1)


def aim_gimbal_down():
    """Aim the gimbal camera downward"""
    print("Aiming gimbal camera down...")
    subprocess.run([
        "gz", "topic", "-t", "/gimbal/cmd_pitch",
        "-m", "gz.msgs.Double", "-p", "data: -1.57"
    ], capture_output=True)


def run_demo(headless=False):
    """Run the ship landing demo"""
    print("=" * 70)
    print("  Ship Landing Demo - Automated Launch with Video Recording")
    print("=" * 70)

    # Configuration
    world_file = "/home/john/github/shipboard_landing/gazebo/worlds/ship_landing_ardupilot.sdf"
    output_dir = "/home/john/github/shipboard_landing/simulation"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output files
    screen_video = os.path.join(output_dir, f"screen_{timestamp}.mp4")
    gimbal_video = os.path.join(output_dir, f"gimbal_telemetry_{timestamp}.mp4")
    landing_video = os.path.join(output_dir, f"landing_telemetry_{timestamp}.mp4")

    # Initialize
    process_manager = ProcessManager()
    controller = ShipLandingController()
    recorders = []
    screen_recorder = None

    try:
        # Create Gazebo GUI config with camera displays
        gui_config = None
        if not headless:
            gui_config = create_gazebo_gui_config()

        # Launch Gazebo in visible terminal
        print("\n--- Launching Gazebo ---")
        process_manager.launch_gazebo(world_file, headless=headless, gui_config=gui_config)

        # Get the detected display for screen recording
        detected_display = getattr(process_manager, 'display', None)

        # Aim gimbal down
        time.sleep(2)
        aim_gimbal_down()

        # Launch ArduPilot SITL (without MAVProxy, connect directly via TCP)
        print("\n--- Launching ArduPilot SITL ---")
        process_manager.launch_ardupilot(with_mavproxy=False)

        # Connect to ArduPilot (via the second mavlink output)
        print("\n--- Connecting to ArduPilot ---")
        if not controller.connect(timeout=60):
            print("Failed to connect to ArduPilot")
            return False

        # Wait for EKF
        if not controller.wait_for_ekf(timeout=60):
            print("EKF failed to initialize")
            return False

        # Start video recording
        print("\n--- Starting Video Recording ---")

        # Screen recording (captures Gazebo window + terminal windows)
        if not headless:
            screen_recorder = ScreenRecorder(screen_video)
            screen_recorder.start(display=detected_display)

        # Camera recordings with telemetry overlay
        gimbal_recorder = CameraRecorder("Gimbal Camera", gimbal_video)
        gimbal_recorder.start()
        recorders.append(gimbal_recorder)

        landing_recorder = CameraRecorder("Landing Camera", landing_video)
        landing_recorder.start()
        recorders.append(landing_recorder)

        time.sleep(2)

        # Flight sequence
        print("\n--- Starting Flight ---")
        controller.set_mode('GUIDED')
        telemetry.set_phase("GUIDED Mode")
        time.sleep(1)

        # Arm
        if not controller.arm(max_attempts=10, force=True):
            print("Failed to arm")
            return False

        time.sleep(1)

        # Takeoff
        if not controller.takeoff(altitude=10):
            print("Takeoff failed")
            return False

        time.sleep(2)

        # Fly pattern
        print("\n--- Flying Pattern ---")

        print("Moving 15m North...")
        controller.goto_position_ned(15, 0, -10, "Flying North")
        controller.wait_for_position(15, 0, -10, tolerance=2, timeout=30)
        time.sleep(2)

        print("Moving 15m East...")
        controller.goto_position_ned(15, 15, -10, "Flying East")
        controller.wait_for_position(15, 15, -10, tolerance=2, timeout=30)
        time.sleep(2)

        print("Moving back South...")
        controller.goto_position_ned(0, 15, -10, "Flying South")
        controller.wait_for_position(0, 15, -10, tolerance=2, timeout=30)
        time.sleep(2)

        print("Returning to helipad...")
        controller.goto_position_ned(0, 0, -10, "Return to Home")
        controller.wait_for_position(0, 0, -10, tolerance=2, timeout=30)
        time.sleep(2)

        # Land
        print("\n--- Landing ---")
        controller.land()
        time.sleep(5)

        print("\n" + "=" * 70)
        print("  Demo Complete!")
        print("=" * 70)

        # Final position
        pos = controller.get_local_position()
        if pos:
            print(f"\nFinal position: N={pos['x']:.2f}m, E={pos['y']:.2f}m")
            accuracy = (pos['x']**2 + pos['y']**2)**0.5
            print(f"Landing accuracy: {accuracy:.2f}m from helipad center")

        print(f"\nVideo files saved to: {output_dir}")
        print(f"  - screen_{timestamp}.mp4 (Gazebo + terminals)")
        print(f"  - gimbal_telemetry_{timestamp}.mp4")
        print(f"  - landing_telemetry_{timestamp}.mp4")

        return True

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        if controller.master:
            controller.set_mode('LAND')
        return False

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        print("\n--- Stopping ---")

        # Stop telemetry
        controller.stop()

        # Stop recorders
        print("Stopping video recorders...")
        for recorder in recorders:
            recorder.stop()
        if screen_recorder:
            screen_recorder.stop()

        time.sleep(2)

        # Cleanup all processes
        print("Shutting down Gazebo and ArduPilot...")
        process_manager.cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ship Landing Demo')
    parser.add_argument('--headless', action='store_true',
                       help='Run without GUI (no screen recording)')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  SHIP LANDING DEMO")
    print("=" * 70)
    print("\nThis script will automatically:")
    print("  1. Detect display (remote :10 or local :0/:1)")
    print("  2. Launch Gazebo with ship world + camera displays")
    print("     (3D view + Landing Camera + Gimbal Camera windows)")
    print("  3. Launch ArduPilot SITL with MAVProxy (in terminal window)")
    print("  4. Arm, takeoff, fly a square pattern")
    print("  5. Return to helipad and land")
    print("  6. Record 3 videos:")
    print("     - Screen capture (Gazebo + terminal windows)")
    print("     - Gimbal camera with telemetry overlay")
    print("     - Landing camera with telemetry overlay")
    print("  7. Clean up all processes on exit")
    print("\nTelemetry overlay includes: Range, Azimuth, Elevation from home")
    print("=" * 70)

    if args.headless:
        print("\nRunning in HEADLESS mode (no GUI, no screen recording)")
    else:
        print("\nRunning with GUI (screen recording enabled)")

    print("\nStarting in 3 seconds... (Ctrl+C to abort)\n")
    time.sleep(3)

    success = run_demo(headless=args.headless)
    sys.exit(0 if success else 1)
