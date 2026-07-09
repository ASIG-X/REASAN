#pragma once

#include "math_utils.hpp"
#include <GL/glew.h>
#include <string>
#include <vector>

class Renderer {
  public:
    Renderer();
    ~Renderer();

    bool initialize();
    void cleanup();

    void beginFrame(const Mat4 &view, const Mat4 &projection);
    void endFrame();

    // Draw functions - points is XYZI format (4 floats per point)
    void drawPointCloud(
        const std::vector<float> &points,
        const Vec3 &color,
        float pointSize = 2.0f,
        float maxDistance = 5.0f,
        float minAlpha = 0.1f,
        float maxAlpha = 0.8f,
        float yaw = 0.0f,
        bool useIntensityColor = true); // Use rainbow colormap based on intensity
    void drawRays(
        const std::vector<float> &rays,
        const Vec3 &origin,
        const Vec3 &color,
        float lineWidth = 1.0f,
        float yaw = 0.0f);
    void drawRayEndpoints(
        const std::vector<float> &rays,
        const Vec3 &origin,
        const Vec3 &color,
        float sphereSize = 6.0f,
        float yaw = 0.0f);
    void drawGrid(float size, int divisions, const Vec3 &color);
    void drawAxes(float length);
    void drawRobot(const Vec3 &position, float yaw);

    // Velocity visualization
    void drawVelocityArrow(
        const Vec3 &origin,
        float vx,
        float vy,
        float robotYaw,
        const Vec3 &color,
        float scale = 0.5f,
        float lineWidth = 3.0f,
        float arrowHeadSize = 0.08f);
    void drawAngularVelocityArc(
        const Vec3 &origin,
        float angularVel,
        float robotYaw,
        const Vec3 &color,
        float radius = 0.4f,
        float lineWidth = 3.0f,
        float arcHeadSize = 0.06f);

  private:
    bool compileShader(GLuint &shader, GLenum type, const char *source);
    bool linkProgram(GLuint &program, GLuint vertexShader, GLuint fragmentShader);

    GLuint shaderProgram_;
    GLuint pointCloudShaderProgram_;
    GLuint vao_;
    GLuint vbo_;

    // Main shader uniforms
    GLint mvpLocation_;
    GLint colorLocation_;

    // Point cloud shader uniforms
    GLint pcMvpLocation_;
    GLint pcColorLocation_;
    GLint pcPointSizeLocation_;
    GLint pcMaxDistanceLocation_;
    GLint pcMinAlphaLocation_;
    GLint pcMaxAlphaLocation_;
    GLint pcUseIntensityColorLocation_;

    Mat4 currentView_;
    Mat4 currentProjection_;
};
