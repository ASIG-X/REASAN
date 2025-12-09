cmake_minimum_required(VERSION 3.8)
project(lidar_processor)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(unitree_sdk2 REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_subdirectory(yaml-cpp)

# Add locomotion node executable
add_executable(locomotion_node locomotion_node.cpp)
target_include_directories(locomotion_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)
  
ament_target_dependencies(locomotion_node
  rclcpp
  geometry_msgs
  std_msgs
  Eigen3
)

target_link_libraries(locomotion_node 
  unitree_sdk2 
  onnxruntime
  ${ONNXRUNTIME_LIBRARIES}
  fmt::fmt
  yaml-cpp::yaml-cpp
)

# Add navigation node executable
add_executable(navigation_node navigation_node.cpp)
target_include_directories(navigation_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)

ament_target_dependencies(navigation_node
  rclcpp
  livox_ros_driver2
  visualization_msgs
  geometry_msgs
  nav_msgs
  std_msgs
  Eigen3
)

target_link_libraries(navigation_node
  onnxruntime
  ${ONNXRUNTIME_LIBRARIES}
  fmt::fmt
  yaml-cpp::yaml-cpp
  unitree_sdk2
)

# Add ray_estimation_node executable
add_executable(ray_estimation_node ray_estimation_node.cpp)
target_include_directories(ray_estimation_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)

ament_target_dependencies(ray_estimation_node
  rclcpp
  livox_ros_driver2
  std_msgs
  geometry_msgs
  visualization_msgs
  nav_msgs
  Eigen3
)

target_link_libraries(ray_estimation_node
  unitree_sdk2
  onnxruntime
  ${ONNXRUNTIME_LIBRARIES}
  fmt::fmt
  yaml-cpp::yaml-cpp
)

# Add filter_node executable
add_executable(filter_node filter_node.cpp)
target_include_directories(filter_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)

ament_target_dependencies(filter_node
  rclcpp
  std_msgs
  geometry_msgs
  Eigen3
)

target_link_libraries(filter_node
  unitree_sdk2
  onnxruntime
  ${ONNXRUNTIME_LIBRARIES}
  fmt::fmt
  yaml-cpp::yaml-cpp
)

# Install models directory
install(DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}/model
  DESTINATION lib/${PROJECT_NAME}
)

# Add mocap to unitree executable
add_executable(odom_node odom_node.cpp)
target_include_directories(odom_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)
  
ament_target_dependencies(odom_node
  rclcpp
  Eigen3
  nav_msgs
  std_msgs
)

target_link_libraries(odom_node
  unitree_sdk2
  fmt::fmt
)

# Install executable
install(TARGETS
  locomotion_node
  navigation_node
  ray_estimation_node
  filter_node
  odom_node
  DESTINATION lib/${PROJECT_NAME})

# Export package info
ament_package()
