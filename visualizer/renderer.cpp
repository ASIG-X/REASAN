#include "renderer.hpp"
#include <cmath>
#include <cstdio>

// Simple vertex shader
static const char *vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat4 uMVP;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)";

// Simple fragment shader
static const char *fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

uniform vec3 uColor;

void main() {
    FragColor = vec4(uColor, 1.0);
}
)";

// Point cloud vertex shader with intensity input
static const char *pointCloudVertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec4 aPosIntensity;  // xyz = position, w = intensity

uniform mat4 uMVP;
uniform float uPointSize;
uniform float uMaxDistance;

out float vDistance;
out float vIntensity;

void main() {
    vec3 pos = aPosIntensity.xyz;
    gl_Position = uMVP * vec4(pos, 1.0);
    gl_PointSize = uPointSize;
    vDistance = length(pos.xy);  // Distance from origin in XY plane
    vIntensity = aPosIntensity.w;
}
)";

// Point cloud fragment shader with intensity-based rainbow coloring
static const char *pointCloudFragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

uniform float uMaxDistance;
uniform float uMinAlpha;
uniform float uMaxAlpha;
uniform int uUseIntensityColor;  // 1 = rainbow colormap, 0 = uniform color
uniform vec3 uColor;             // Used when uUseIntensityColor = 0

in float vDistance;
in float vIntensity;

// Rainbow colormap function (RViz style: red -> yellow -> green -> cyan -> blue)
vec3 rainbowColormap(float t) {
    // Clamp input to [0, 1]
    t = clamp(t, 0.0, 1.0);
    
    // RViz rainbow: Red (low) -> Yellow -> Green -> Cyan -> Blue (high)
    vec3 color;
    
    if (t < 0.25) {
        // Red to Yellow (0.0 - 0.25)
        float s = t / 0.25;
        color = vec3(1.0, s, 0.0);
    } else if (t < 0.5) {
        // Yellow to Green (0.25 - 0.5)
        float s = (t - 0.25) / 0.25;
        color = vec3(1.0 - s, 1.0, 0.0);
    } else if (t < 0.75) {
        // Green to Cyan (0.5 - 0.75)
        float s = (t - 0.5) / 0.25;
        color = vec3(0.0, 1.0, s);
    } else {
        // Cyan to Blue (0.75 - 1.0)
        float s = (t - 0.75) / 0.25;
        color = vec3(0.0, 1.0 - s, 1.0);
    }
    
    return color;
}

void main() {
    // Make points circular by discarding pixels outside radius
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) {
        discard;
    }
    
    // Get color based on mode
    vec3 color;
    if (uUseIntensityColor == 1) {
        color = rainbowColormap(vIntensity);
    } else {
        color = uColor;
    }
    
    // Alpha decreases with distance from origin
    float normalizedDist = clamp(vDistance / uMaxDistance, 0.0, 1.0);
    float alpha = mix(uMaxAlpha, uMinAlpha, normalizedDist);
    FragColor = vec4(color, alpha);
}
)";

Renderer::Renderer()
    : shaderProgram_(0)
    , pointCloudShaderProgram_(0)
    , vao_(0)
    , vbo_(0)
    , mvpLocation_(-1)
    , colorLocation_(-1)
    , pcMvpLocation_(-1)
    , pcColorLocation_(-1)
    , pcPointSizeLocation_(-1)
    , pcMaxDistanceLocation_(-1)
    , pcMinAlphaLocation_(-1)
    , pcMaxAlphaLocation_(-1)
    , pcUseIntensityColorLocation_(-1) {}

Renderer::~Renderer() { cleanup(); }

bool Renderer::initialize() {
    // Compile shaders
    GLuint vertexShader, fragmentShader;

    if (!compileShader(vertexShader, GL_VERTEX_SHADER, vertexShaderSource)) {
        return false;
    }

    if (!compileShader(fragmentShader, GL_FRAGMENT_SHADER, fragmentShaderSource)) {
        glDeleteShader(vertexShader);
        return false;
    }

    if (!linkProgram(shaderProgram_, vertexShader, fragmentShader)) {
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return false;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Get uniform locations for main shader
    mvpLocation_ = glGetUniformLocation(shaderProgram_, "uMVP");
    colorLocation_ = glGetUniformLocation(shaderProgram_, "uColor");

    // Compile point cloud shaders
    GLuint pcVertexShader, pcFragmentShader;

    if (!compileShader(pcVertexShader, GL_VERTEX_SHADER, pointCloudVertexShaderSource)) {
        return false;
    }

    if (!compileShader(pcFragmentShader, GL_FRAGMENT_SHADER, pointCloudFragmentShaderSource)) {
        glDeleteShader(pcVertexShader);
        return false;
    }

    if (!linkProgram(pointCloudShaderProgram_, pcVertexShader, pcFragmentShader)) {
        glDeleteShader(pcVertexShader);
        glDeleteShader(pcFragmentShader);
        return false;
    }

    glDeleteShader(pcVertexShader);
    glDeleteShader(pcFragmentShader);

    // Get uniform locations for point cloud shader
    pcMvpLocation_ = glGetUniformLocation(pointCloudShaderProgram_, "uMVP");
    pcColorLocation_ = glGetUniformLocation(pointCloudShaderProgram_, "uColor");
    pcPointSizeLocation_ = glGetUniformLocation(pointCloudShaderProgram_, "uPointSize");
    pcMaxDistanceLocation_ = glGetUniformLocation(pointCloudShaderProgram_, "uMaxDistance");
    pcMinAlphaLocation_ = glGetUniformLocation(pointCloudShaderProgram_, "uMinAlpha");
    pcMaxAlphaLocation_ = glGetUniformLocation(pointCloudShaderProgram_, "uMaxAlpha");
    pcUseIntensityColorLocation_ =
        glGetUniformLocation(pointCloudShaderProgram_, "uUseIntensityColor");

    // Create VAO and VBO
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    // Note: Vertex attribute setup will be done per-draw call since we have different formats
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    // Enable depth testing and point size
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    return true;
}

void Renderer::cleanup() {
    if (vbo_) {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }
    if (vao_) {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
    if (shaderProgram_) {
        glDeleteProgram(shaderProgram_);
        shaderProgram_ = 0;
    }
    if (pointCloudShaderProgram_) {
        glDeleteProgram(pointCloudShaderProgram_);
        pointCloudShaderProgram_ = 0;
    }
}

bool Renderer::compileShader(GLuint &shader, GLenum type, const char *source) {
    shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        fprintf(stderr, "Shader compilation error: %s\n", infoLog);
        return false;
    }
    return true;
}

bool Renderer::linkProgram(GLuint &program, GLuint vertexShader, GLuint fragmentShader) {
    program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        fprintf(stderr, "Program linking error: %s\n", infoLog);
        return false;
    }
    return true;
}

void Renderer::beginFrame(const Mat4 &view, const Mat4 &projection) {
    currentView_ = view;
    currentProjection_ = projection;

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaderProgram_);
}

void Renderer::endFrame() { glUseProgram(0); }

void Renderer::drawPointCloud(
    const std::vector<float> &points,
    const Vec3 &color,
    float pointSize,
    float maxDistance,
    float minAlpha,
    float maxAlpha,
    float yaw,
    bool useIntensityColor) {
    if (points.empty())
        return;

    // Points are in XYZI format (4 floats per point)
    size_t numPoints = points.size() / 4;
    if (numPoints == 0)
        return;

    // Rotate points by yaw if needed (preserve intensity)
    std::vector<float> rotatedPoints;
    if (std::fabs(yaw) > 0.001f) {
        float cosYaw = std::cos(yaw);
        float sinYaw = std::sin(yaw);
        rotatedPoints.resize(points.size());
        for (size_t i = 0; i < numPoints; i++) {
            size_t idx = i * 4;
            float x = points[idx];
            float y = points[idx + 1];
            rotatedPoints[idx] = x * cosYaw - y * sinYaw;
            rotatedPoints[idx + 1] = x * sinYaw + y * cosYaw;
            rotatedPoints[idx + 2] = points[idx + 2]; // z
            rotatedPoints[idx + 3] = points[idx + 3]; // intensity
        }
    }
    const std::vector<float> &pointsToRender = (std::fabs(yaw) > 0.001f) ? rotatedPoints : points;

    // Use point cloud shader
    glUseProgram(pointCloudShaderProgram_);

    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Disable depth write for transparent points (but keep depth test)
    glDepthMask(GL_FALSE);

    Mat4 mvp = currentProjection_ * currentView_;
    glUniformMatrix4fv(pcMvpLocation_, 1, GL_FALSE, mvp.data());
    glUniform3f(pcColorLocation_, color.x, color.y, color.z);
    glUniform1f(pcPointSizeLocation_, pointSize);
    glUniform1f(pcMaxDistanceLocation_, maxDistance);
    glUniform1f(pcMinAlphaLocation_, minAlpha);
    glUniform1f(pcMaxAlphaLocation_, maxAlpha);
    glUniform1i(pcUseIntensityColorLocation_, useIntensityColor ? 1 : 0);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(
        GL_ARRAY_BUFFER, pointsToRender.size() * sizeof(float), pointsToRender.data(),
        GL_DYNAMIC_DRAW);

    // Set vertex attribute for vec4 (XYZI)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glDrawArrays(GL_POINTS, 0, numPoints);

    glBindVertexArray(0);

    // Restore state
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    // Switch back to main shader
    glUseProgram(shaderProgram_);
}

void Renderer::drawRays(
    const std::vector<float> &rays,
    const Vec3 &origin,
    const Vec3 &color,
    float lineWidth,
    float yaw) {
    if (rays.empty())
        return;

    // Build line vertices: origin to ray endpoint for each ray
    std::vector<float> lineVertices;
    int numRays = rays.size();
    lineVertices.reserve(numRays * 6); // 2 vertices per ray, 3 floats each

    for (int i = 0; i < numRays; i++) {
        // Base angle in body frame + yaw offset for global frame
        float angle = -M_PI + (2.0f * M_PI * i) / numRays + yaw;
        float distance = rays[i];

        // Origin
        lineVertices.push_back(origin.x);
        lineVertices.push_back(origin.y);
        lineVertices.push_back(origin.z);

        // Endpoint
        lineVertices.push_back(origin.x + distance * std::cos(angle));
        lineVertices.push_back(origin.y + distance * std::sin(angle));
        lineVertices.push_back(origin.z);
    }

    Mat4 mvp = currentProjection_ * currentView_;
    glUniformMatrix4fv(mvpLocation_, 1, GL_FALSE, mvp.data());
    glUniform3f(colorLocation_, color.x, color.y, color.z);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(
        GL_ARRAY_BUFFER, lineVertices.size() * sizeof(float), lineVertices.data(), GL_DYNAMIC_DRAW);

    // Set vertex attribute for vec3 (XYZ only)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);

    glLineWidth(lineWidth);
    glDrawArrays(GL_LINES, 0, lineVertices.size() / 3);

    glBindVertexArray(0);
}

void Renderer::drawRayEndpoints(
    const std::vector<float> &rays,
    const Vec3 &origin,
    const Vec3 &color,
    float sphereSize,
    float yaw) {
    if (rays.empty())
        return;

    // Build endpoint vertices
    std::vector<float> endpoints;
    int numRays = rays.size();
    endpoints.reserve(numRays * 3);

    for (int i = 0; i < numRays; i++) {
        float angle = -M_PI + (2.0f * M_PI * i) / numRays + yaw;
        float distance = rays[i];

        // Endpoint position
        endpoints.push_back(origin.x + distance * std::cos(angle));
        endpoints.push_back(origin.y + distance * std::sin(angle));
        endpoints.push_back(origin.z);
    }

    // Use point cloud shader for round points
    glUseProgram(pointCloudShaderProgram_);

    Mat4 mvp = currentProjection_ * currentView_;
    glUniformMatrix4fv(pcMvpLocation_, 1, GL_FALSE, mvp.data());
    glUniform3f(pcColorLocation_, color.x, color.y, color.z);
    glUniform1f(pcPointSizeLocation_, sphereSize);
    glUniform1f(pcMaxDistanceLocation_, 10.0f); // Large value so alpha is ~1
    glUniform1f(pcMinAlphaLocation_, 1.0f);
    glUniform1f(pcMaxAlphaLocation_, 1.0f);
    glUniform1i(pcUseIntensityColorLocation_, 0); // Use uniform color, not intensity

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(
        GL_ARRAY_BUFFER, endpoints.size() * sizeof(float), endpoints.data(), GL_DYNAMIC_DRAW);

    // Set vertex attribute for vec3 (XYZ only, intensity will be undefined but not used)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glDrawArrays(GL_POINTS, 0, endpoints.size() / 3);

    glBindVertexArray(0);

    // Switch back to main shader
    glUseProgram(shaderProgram_);
}

void Renderer::drawGrid(float size, int divisions, const Vec3 &color) {
    std::vector<float> gridVertices;
    float step = size / divisions;
    float halfSize = size / 2.0f;

    // Lines parallel to Y axis
    for (int i = 0; i <= divisions; i++) {
        float x = -halfSize + i * step;
        gridVertices.push_back(x);
        gridVertices.push_back(-halfSize);
        gridVertices.push_back(0.0f);
        gridVertices.push_back(x);
        gridVertices.push_back(halfSize);
        gridVertices.push_back(0.0f);
    }

    // Lines parallel to X axis
    for (int i = 0; i <= divisions; i++) {
        float y = -halfSize + i * step;
        gridVertices.push_back(-halfSize);
        gridVertices.push_back(y);
        gridVertices.push_back(0.0f);
        gridVertices.push_back(halfSize);
        gridVertices.push_back(y);
        gridVertices.push_back(0.0f);
    }

    Mat4 mvp = currentProjection_ * currentView_;
    glUniformMatrix4fv(mvpLocation_, 1, GL_FALSE, mvp.data());
    glUniform3f(colorLocation_, color.x, color.y, color.z);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(
        GL_ARRAY_BUFFER, gridVertices.size() * sizeof(float), gridVertices.data(), GL_DYNAMIC_DRAW);

    // Set vertex attribute for vec3
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);

    glLineWidth(1.0f);
    glDrawArrays(GL_LINES, 0, gridVertices.size() / 3);

    glBindVertexArray(0);
}

void Renderer::drawAxes(float length) {
    std::vector<float> axesVertices = {
        // X axis (red)
        0.0f,
        0.0f,
        0.0f,
        length,
        0.0f,
        0.0f,
        // Y axis (green)
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        length,
        0.0f,
        // Z axis (blue)
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        length,
    };

    Mat4 mvp = currentProjection_ * currentView_;
    glUniformMatrix4fv(mvpLocation_, 1, GL_FALSE, mvp.data());

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(
        GL_ARRAY_BUFFER, axesVertices.size() * sizeof(float), axesVertices.data(), GL_DYNAMIC_DRAW);

    // Set vertex attribute for vec3
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);

    glLineWidth(2.0f);

    // Draw X axis (red)
    glUniform3f(colorLocation_, 1.0f, 0.2f, 0.2f);
    glDrawArrays(GL_LINES, 0, 2);

    // Draw Y axis (green)
    glUniform3f(colorLocation_, 0.2f, 1.0f, 0.2f);
    glDrawArrays(GL_LINES, 2, 2);

    // Draw Z axis (blue)
    glUniform3f(colorLocation_, 0.2f, 0.2f, 1.0f);
    glDrawArrays(GL_LINES, 4, 2);

    glBindVertexArray(0);
}

void Renderer::drawRobot(const Vec3 &position, float yaw) {
    // Simple robot representation: filled arrow pointing in direction of yaw
    float robotSize = 0.25f; // Smaller triangle

    // Triangle pointing in yaw direction
    float tipX = position.x + robotSize * std::cos(yaw);
    float tipY = position.y + robotSize * std::sin(yaw);

    float backAngle1 = yaw + 2.5f;
    float backAngle2 = yaw - 2.5f;
    float backDist = robotSize * 0.6f;

    float back1X = position.x + backDist * std::cos(backAngle1);
    float back1Y = position.y + backDist * std::sin(backAngle1);
    float back2X = position.x + backDist * std::cos(backAngle2);
    float back2Y = position.y + backDist * std::sin(backAngle2);

    // Filled triangle (3 vertices) - elevated above ground
    float elevation = 0.05f;
    std::vector<float> fillVertices = {tipX,   tipY,   position.z + elevation,
                                       back1X, back1Y, position.z + elevation,
                                       back2X, back2Y, position.z + elevation};

    Mat4 mvp = currentProjection_ * currentView_;
    glUniformMatrix4fv(mvpLocation_, 1, GL_FALSE, mvp.data());

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);

    // Set vertex attribute for vec3
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);

    // Draw filled triangle (yellow)
    glUniform3f(colorLocation_, 1.0f, 0.8f, 0.0f);
    glBufferData(
        GL_ARRAY_BUFFER, fillVertices.size() * sizeof(float), fillVertices.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindVertexArray(0);
}

void Renderer::drawVelocityArrow(
    const Vec3 &origin,
    float vx,
    float vy,
    float robotYaw,
    const Vec3 &color,
    float scale,
    float lineWidth,
    float arrowHeadSize) {
    // Skip if velocity is too small
    float magnitude = std::sqrt(vx * vx + vy * vy);
    if (magnitude < 0.01f)
        return;

    // Transform velocity from robot frame to world frame
    float cosYaw = std::cos(robotYaw);
    float sinYaw = std::sin(robotYaw);
    float worldVx = vx * cosYaw - vy * sinYaw;
    float worldVy = vx * sinYaw + vy * cosYaw;

    // Scale the arrow
    float endX = origin.x + worldVx * scale;
    float endY = origin.y + worldVy * scale;

    // Create arrow head
    float arrowAngle = std::atan2(worldVy, worldVx);
    float head1Angle = arrowAngle + M_PI - 0.4f;
    float head2Angle = arrowAngle + M_PI + 0.4f;

    float head1X = endX + arrowHeadSize * std::cos(head1Angle);
    float head1Y = endY + arrowHeadSize * std::sin(head1Angle);
    float head2X = endX + arrowHeadSize * std::cos(head2Angle);
    float head2Y = endY + arrowHeadSize * std::sin(head2Angle);

    std::vector<float> arrowVertices = {
        // Main line
        origin.x,
        origin.y,
        origin.z + 0.05f,
        endX,
        endY,
        origin.z + 0.05f,
        // Arrow head
        endX,
        endY,
        origin.z + 0.05f,
        head1X,
        head1Y,
        origin.z + 0.05f,
        endX,
        endY,
        origin.z + 0.05f,
        head2X,
        head2Y,
        origin.z + 0.05f,
    };

    Mat4 mvp = currentProjection_ * currentView_;
    glUniformMatrix4fv(mvpLocation_, 1, GL_FALSE, mvp.data());
    glUniform3f(colorLocation_, color.x, color.y, color.z);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(
        GL_ARRAY_BUFFER, arrowVertices.size() * sizeof(float), arrowVertices.data(),
        GL_DYNAMIC_DRAW);

    // Set vertex attribute for vec3
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);

    glLineWidth(lineWidth);
    glDrawArrays(GL_LINES, 0, arrowVertices.size() / 3);

    glBindVertexArray(0);
}

void Renderer::drawAngularVelocityArc(
    const Vec3 &origin,
    float angularVel,
    float robotYaw,
    const Vec3 &color,
    float radius,
    float lineWidth,
    float arcHeadSize) {
    // Skip if angular velocity is too small
    if (std::fabs(angularVel) < 0.05f)
        return;

    // Clamp angular velocity for visualization
    float clampedAngVel = std::max(-3.0f, std::min(3.0f, angularVel));

    // Arc spans proportional to angular velocity (max 180 degrees at max speed)
    float arcSpan = (clampedAngVel / 3.0f) * M_PI * 0.5f;

    // Start angle is robot's forward direction
    float startAngle = robotYaw;
    float endAngle = robotYaw + arcSpan;

    // Generate arc vertices
    std::vector<float> arcVertices;
    int segments = 16;
    float angleStep = arcSpan / segments;

    for (int i = 0; i < segments; ++i) {
        float a1 = startAngle + i * angleStep;
        float a2 = startAngle + (i + 1) * angleStep;

        arcVertices.push_back(origin.x + radius * std::cos(a1));
        arcVertices.push_back(origin.y + radius * std::sin(a1));
        arcVertices.push_back(origin.z + 0.05f);

        arcVertices.push_back(origin.x + radius * std::cos(a2));
        arcVertices.push_back(origin.y + radius * std::sin(a2));
        arcVertices.push_back(origin.z + 0.05f);
    }

    // Add arrowhead at the end of the arc
    float arrowDir = (angularVel > 0) ? (endAngle + M_PI / 2) : (endAngle - M_PI / 2);
    float head1Angle = arrowDir - 0.5f;
    float head2Angle = arrowDir + 0.5f;

    float endX = origin.x + radius * std::cos(endAngle);
    float endY = origin.y + radius * std::sin(endAngle);
    float head1X = endX + arcHeadSize * std::cos(head1Angle);
    float head1Y = endY + arcHeadSize * std::sin(head1Angle);
    float head2X = endX + arcHeadSize * std::cos(head2Angle);
    float head2Y = endY + arcHeadSize * std::sin(head2Angle);

    arcVertices.push_back(endX);
    arcVertices.push_back(endY);
    arcVertices.push_back(origin.z + 0.05f);
    arcVertices.push_back(head1X);
    arcVertices.push_back(head1Y);
    arcVertices.push_back(origin.z + 0.05f);

    arcVertices.push_back(endX);
    arcVertices.push_back(endY);
    arcVertices.push_back(origin.z + 0.05f);
    arcVertices.push_back(head2X);
    arcVertices.push_back(head2Y);
    arcVertices.push_back(origin.z + 0.05f);

    Mat4 mvp = currentProjection_ * currentView_;
    glUniformMatrix4fv(mvpLocation_, 1, GL_FALSE, mvp.data());
    glUniform3f(colorLocation_, color.x, color.y, color.z);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(
        GL_ARRAY_BUFFER, arcVertices.size() * sizeof(float), arcVertices.data(), GL_DYNAMIC_DRAW);

    // Set vertex attribute for vec3
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);

    glLineWidth(lineWidth);
    glDrawArrays(GL_LINES, 0, arcVertices.size() / 3);

    glBindVertexArray(0);
}
