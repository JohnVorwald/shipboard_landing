cmake_minimum_required(VERSION 3.10.2)
project(gazebo_plugins)

# Find Gazebo Sim and its required components
find_package(gz-sim9 REQUIRED)

# --- Ship Motion Plugin ONLY ---
add_library(ship_motion_plugin SHARED ship_motion_plugin.cc)

# Link ONLY this plugin. This is the correct, modern way.
target_link_libraries(ship_motion_plugin PRIVATE gz-sim9::gz-sim9)
