cmake_minimum_required(VERSION 3.10)
project(go2_visualizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use GLVND OpenGL libraries (modern)
set(OpenGL_GL_PREFERENCE GLVND)

# Find required packages (all available via apt)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)
pkg_check_modules(TINYXML2 REQUIRED tinyxml2)

# Add executable
add_executable(go2_visualizer
    main.cpp
    camera.cpp
    renderer.cpp
    text_renderer.cpp
    zmq_receiver.cpp
    robot_model.cpp
)

target_include_directories(go2_visualizer PRIVATE
    ${OPENGL_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${ZMQ_INCLUDE_DIRS}
    ${TINYXML2_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(go2_visualizer
    ${OPENGL_LIBRARIES}
    glfw
    GLEW::GLEW
    ${ZMQ_LIBRARIES}
    ${TINYXML2_LIBRARIES}
    pthread
)

# Install
install(TARGETS go2_visualizer DESTINATION bin)
