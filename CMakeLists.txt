cmake_minimum_required(VERSION 3.0.2)
project(foundation_pose)

find_package(catkin REQUIRED)

catkin_package()

install(
  PROGRAMS scripts/pose_estimation.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
