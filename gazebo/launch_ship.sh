#!/bin/bash
export GZ_SIM_RESOURCE_PATH=/home/john/gz_ws/src/ardupilot_gazebo/models:/home/john/gz_ws/src/ardupilot_gazebo/worlds:/home/john/github/shipboard_landing/gazebo/models
export GZ_SIM_SYSTEM_PLUGIN_PATH=/home/john/gz_ws/src/ardupilot_gazebo/build
export DISPLAY=:10.0
export GZ_IP=127.0.0.1
export GZ_PARTITION=gazebo_default
export LIBGL_DRI3_DISABLE=1
export QT_QPA_PLATFORM=xcb

# Run with default render engine (ogre2) - ogre1 has bounding box issues
gz sim -r /home/john/github/shipboard_landing/gazebo/worlds/ship_landing_ardupilot.sdf
