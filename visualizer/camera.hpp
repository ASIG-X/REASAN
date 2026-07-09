#pragma once

#include "math_utils.hpp"
#include <GLFW/glfw3.h>

class Camera {
  public:
    Camera();

    void processMouseButton(int button, int action, double xpos, double ypos);
    void processMouseMove(double xpos, double ypos);
    void processScroll(double yoffset);
    void processKeyboard(GLFWwindow *window, float deltaTime);

    Mat4 getViewMatrix() const;
    Mat4 getProjectionMatrix(float aspectRatio) const;

    void reset();

    Vec3 getPosition() const { return position_; }
    Vec3 getTarget() const { return target_; }

  private:
    void updateCameraVectors();

    Vec3 position_;
    Vec3 target_;
    Vec3 up_;

    // Orbit camera parameters
    float distance_;
    float yaw_;
    float pitch_;

    // Mouse state
    bool leftButtonDown_;
    bool middleButtonDown_;
    bool rightButtonDown_;
    double lastMouseX_;
    double lastMouseY_;

    // Camera settings
    float mouseSensitivity_;
    float scrollSensitivity_;
    float movementSpeed_;
    float fov_;
    float nearPlane_;
    float farPlane_;
};
