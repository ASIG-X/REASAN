#include "camera.hpp"
#include <algorithm>
#include <cmath>

Camera::Camera()
    : target_(0.0f, 0.0f, 0.0f)
    , up_(0.0f, 0.0f, 1.0f)
    , distance_(5.0f)
    , yaw_(0.0f)
    , pitch_(0.5f)
    , leftButtonDown_(false)
    , middleButtonDown_(false)
    , rightButtonDown_(false)
    , lastMouseX_(0.0)
    , lastMouseY_(0.0)
    , mouseSensitivity_(0.005f)
    , scrollSensitivity_(0.3f)
    , movementSpeed_(2.0f)
    , fov_(45.0f * M_PI / 180.0f)
    , nearPlane_(0.1f)
    , farPlane_(100.0f) {
    updateCameraVectors();
}

void Camera::updateCameraVectors() {
    // Calculate position based on spherical coordinates around target
    position_.x = target_.x + distance_ * std::cos(pitch_) * std::cos(yaw_);
    position_.y = target_.y + distance_ * std::cos(pitch_) * std::sin(yaw_);
    position_.z = target_.z + distance_ * std::sin(pitch_);
}

void Camera::processMouseButton(int button, int action, double xpos, double ypos) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        leftButtonDown_ = (action == GLFW_PRESS);
    } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        middleButtonDown_ = (action == GLFW_PRESS);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        rightButtonDown_ = (action == GLFW_PRESS);
    }

    lastMouseX_ = xpos;
    lastMouseY_ = ypos;
}

void Camera::processMouseMove(double xpos, double ypos) {
    double deltaX = xpos - lastMouseX_;
    double deltaY = ypos - lastMouseY_;

    if (leftButtonDown_) {
        // Orbit rotation
        yaw_ -= static_cast<float>(deltaX) * mouseSensitivity_;
        pitch_ += static_cast<float>(deltaY) * mouseSensitivity_;

        // Clamp pitch to avoid flipping
        pitch_ = std::clamp(pitch_, -1.5f, 1.5f);

        updateCameraVectors();
    } else if (middleButtonDown_ || rightButtonDown_) {
        // Pan
        Vec3 forward = (target_ - position_).normalized();
        Vec3 right = forward.cross(up_).normalized();
        Vec3 cameraUp = right.cross(forward).normalized();

        float panSpeed = distance_ * 0.002f;
        target_ -= right * static_cast<float>(deltaX) * panSpeed;
        target_ += cameraUp * static_cast<float>(deltaY) * panSpeed;

        updateCameraVectors();
    }

    lastMouseX_ = xpos;
    lastMouseY_ = ypos;
}

void Camera::processScroll(double yoffset) {
    distance_ -= static_cast<float>(yoffset) * scrollSensitivity_;
    distance_ = std::clamp(distance_, 0.5f, 50.0f);
    updateCameraVectors();
}

void Camera::processKeyboard(GLFWwindow *window, float deltaTime) {
    float velocity = movementSpeed_ * deltaTime;

    Vec3 forward = (target_ - position_).normalized();
    Vec3 right = forward.cross(up_).normalized();

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        target_ += forward * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        target_ -= forward * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        target_ -= right * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        target_ += right * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        target_.z += velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        target_.z -= velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        reset();
    }

    updateCameraVectors();
}

Mat4 Camera::getViewMatrix() const { return Mat4::lookAt(position_, target_, up_); }

Mat4 Camera::getProjectionMatrix(float aspectRatio) const {
    return Mat4::perspective(fov_, aspectRatio, nearPlane_, farPlane_);
}

void Camera::reset() {
    target_ = Vec3(0.0f, 0.0f, 0.0f);
    distance_ = 5.0f;
    yaw_ = 0.0f;
    pitch_ = 0.5f;
    updateCameraVectors();
}
