cmake_minimum_required(VERSION 3.5)
project(livox_ros_driver2)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
#find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CustomPoint.msg"
  "msg/CustomMsg.msg"
  #DEPENDENCIES builtin_interfaces std_msgs
  DEPENDENCIES std_msgs  
  LIBRARY_NAME ${PROJECT_NAME}
 )
 

ament_export_dependencies(rosidl_default_runtime)

ament_package()
