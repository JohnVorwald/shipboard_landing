#!/usr/bin/env python3
"""View Gazebo camera using GStreamer - receives UDP stream"""
import subprocess
import sys
import os

# The gimbal camera streams to UDP port 5600 by default
# Check if gstreamer is available
print("Attempting to view camera stream...")
print("The gimbal camera streams to UDP port 5600")

# Try using ffplay or gst-launch
try:
    # First check if there's a UDP stream
    cmd = ["gst-launch-1.0", "udpsrc", "port=5600", "!", 
           "application/x-rtp,encoding-name=H264", "!",
           "rtph264depay", "!", "avdec_h264", "!", 
           "videoconvert", "!", "autovideosink"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
except FileNotFoundError:
    print("gst-launch-1.0 not found, trying ffplay...")
    try:
        cmd = ["ffplay", "-i", "udp://127.0.0.1:5600"]
        subprocess.run(cmd)
    except FileNotFoundError:
        print("Neither gstreamer nor ffplay found.")
        print("\nAlternative: Use 'gz sim -g' to open Gazebo GUI")
        print("Then go to Plugins → Image Display")
