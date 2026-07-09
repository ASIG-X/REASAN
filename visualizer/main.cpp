#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "camera.hpp"
#include "renderer.hpp"
#include "robot_model.hpp"
#include "text_renderer.hpp"
#include "zmq_receiver.hpp"

// Global variables for callbacks
Camera *g_camera = nullptr;

// Yaw offset calibration
float g_yawOffset = 0.0f;       // Stored yaw offset
float g_yawOffsetDelta = 0.0f;  // Temporary offset while dragging
bool g_calibratingYaw = false;  // Currently calibrating
double g_calibrateStartX = 0.0; // Mouse X when calibration started

// Frame mode: false = body frame (default), true = global frame
bool g_globalFrame = false;

void errorCallback(int error, const char *description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    // Check for Ctrl+Left click for yaw calibration
    if (button == GLFW_MOUSE_BUTTON_LEFT && (mods & GLFW_MOD_CONTROL)) {
        if (action == GLFW_PRESS) {
            g_calibratingYaw = true;
            g_calibrateStartX = xpos;
            g_yawOffsetDelta = 0.0f;
            printf("Yaw calibration started. Drag left/right to adjust.\n");
        } else if (action == GLFW_RELEASE) {
            g_calibratingYaw = false;
            g_yawOffset += g_yawOffsetDelta;
            g_yawOffsetDelta = 0.0f;
            printf("Yaw calibration finished. Offset: %.2f degrees\n", g_yawOffset * 180.0f / M_PI);
        }
        return; // Don't pass to camera
    }

    if (g_camera) {
        g_camera->processMouseButton(button, action, xpos, ypos);
    }
}

void cursorPosCallback(GLFWwindow *window, double xpos, double ypos) {
    // Handle yaw calibration drag
    if (g_calibratingYaw) {
        double deltaX = xpos - g_calibrateStartX;
        // Sensitivity: 200 pixels = 90 degrees
        g_yawOffsetDelta = static_cast<float>(deltaX / 200.0 * M_PI / 2.0);
        return; // Don't pass to camera while calibrating
    }

    if (g_camera) {
        g_camera->processMouseMove(xpos, ypos);
    }
}

void scrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
    if (g_camera) {
        g_camera->processScroll(yoffset);
    }
}

void printUsage(const char *programName) {
    printf("Usage: %s [options]\n", programName);
    printf("Options:\n");
    printf("  -a, --address <addr>  ZMQ server address (default: tcp://localhost:5555)\n");
    printf("  -w, --width <px>      Window width (default: 1280)\n");
    printf("  -h, --height <px>     Window height (default: 720)\n");
    printf("  -g, --global          Display in global frame (default: body frame)\n");
    printf("  -f, --font <path>     TTF font file path\n");
    printf("  -s, --fontsize <pt>   Font size in points (default: 18)\n");
    printf("  -p, --pointsize <px>  Point cloud point size (default: 4.0)\n");
    printf("  -d, --fadedist <m>    Distance at which points fade to transparent (default: 3.0)\n");
    printf("  -m, --model <path>    Robot model XML file path\n");
    printf("  --arrow-width <px>    Velocity arrow line width (default: 3.0)\n");
    printf("  --arrow-scale <m>     Velocity arrow scale factor (default: 0.5)\n");
    printf("  --arrow-head <m>      Velocity arrow head size (default: 0.08)\n");
    printf("  --arc-radius <m>      Angular velocity arc radius (default: 0.6)\n");
    printf("  --arc-gap <m>         Gap between inner and outer arc (default: 0.15)\n");
    printf("  --arc-head <m>        Angular velocity arc arrow head size (default: 0.1)\n");
    printf("  --robot-height <m>    Robot z-offset above ground (default: 0.3)\n");
    printf("  --robot-scale <f>     Robot model scale factor (default: 1.0)\n");
    printf("  --help                Show this help message\n");
    printf("\nControls:\n");
    printf("  Left mouse drag     - Rotate camera\n");
    printf("  Right/Middle drag   - Pan camera\n");
    printf("  Scroll wheel        - Zoom in/out\n");
    printf("  Ctrl + Left drag    - Calibrate yaw offset\n");
    printf("  W/A/S/D             - Move target point\n");
    printf("  Q/E                 - Move target up/down\n");
    printf("  R                   - Reset camera view\n");
    printf("  ESC                 - Exit\n");
}

int main(int argc, char **argv) {
    // Parse command line arguments
    std::string zmqAddress = "tcp://localhost:5555";
    std::string fontPath = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf";
    std::string modelPath = "go2_model/go2.xml"; // Default to robot model
    int windowWidth = 1280;
    int windowHeight = 720;
    float fontSize = 18.0f;
    float pointSize = 4.0f;
    float fadeDistance = 3.0f;
    float arrowWidth = 3.0f;
    float arrowScale = 0.5f;
    float arrowHeadSize = 0.08f;
    float arcRadius = 0.6f;
    float arcGap = 0.15f;
    float arcHeadSize = 0.1f;
    float robotHeight = 0.3f;
    float robotScale = 1.0f;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-a" || arg == "--address") && i + 1 < argc) {
            zmqAddress = argv[++i];
        } else if ((arg == "-w" || arg == "--width") && i + 1 < argc) {
            windowWidth = std::atoi(argv[++i]);
        } else if ((arg == "-h" || arg == "--height") && i + 1 < argc) {
            windowHeight = std::atoi(argv[++i]);
        } else if (arg == "-g" || arg == "--global") {
            g_globalFrame = true;
        } else if ((arg == "-f" || arg == "--font") && i + 1 < argc) {
            fontPath = argv[++i];
        } else if ((arg == "-s" || arg == "--fontsize") && i + 1 < argc) {
            fontSize = std::atof(argv[++i]);
        } else if ((arg == "-p" || arg == "--pointsize") && i + 1 < argc) {
            pointSize = std::atof(argv[++i]);
        } else if ((arg == "-d" || arg == "--fadedist") && i + 1 < argc) {
            fadeDistance = std::atof(argv[++i]);
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (arg == "--arrow-width" && i + 1 < argc) {
            arrowWidth = std::atof(argv[++i]);
        } else if (arg == "--arrow-scale" && i + 1 < argc) {
            arrowScale = std::atof(argv[++i]);
        } else if (arg == "--arrow-head" && i + 1 < argc) {
            arrowHeadSize = std::atof(argv[++i]);
        } else if (arg == "--arc-radius" && i + 1 < argc) {
            arcRadius = std::atof(argv[++i]);
        } else if (arg == "--arc-gap" && i + 1 < argc) {
            arcGap = std::atof(argv[++i]);
        } else if (arg == "--arc-head" && i + 1 < argc) {
            arcHeadSize = std::atof(argv[++i]);
        } else if (arg == "--robot-height" && i + 1 < argc) {
            robotHeight = std::atof(argv[++i]);
        } else if (arg == "--robot-scale" && i + 1 < argc) {
            robotScale = std::atof(argv[++i]);
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }

    printf("Go2 Visualizer\n");
    printf("==============\n");
    printf("Connecting to: %s\n", zmqAddress.c_str());
    printf("Window size: %dx%d\n", windowWidth, windowHeight);
    printf("Font size: %.1f pt\n", fontSize);
    printf("Frame mode: %s\n", g_globalFrame ? "GLOBAL" : "BODY");
    if (!modelPath.empty()) {
        printf("Robot model: %s\n", modelPath.c_str());
    }
    printf("\n");

    // Initialize GLFW
    glfwSetErrorCallback(errorCallback);
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    // Set OpenGL version
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4); // Anti-aliasing

    // Create window
    GLFWwindow *window =
        glfwCreateWindow(windowWidth, windowHeight, "Go2 Visualizer", nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // VSync

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    printf("OpenGL Version: %s\n", glGetString(GL_VERSION));
    printf("GLSL Version: %s\n\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    // Set callbacks
    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);

    // Enable multisampling
    glEnable(GL_MULTISAMPLE);

    // Create camera and renderer
    Camera camera;
    g_camera = &camera;

    Renderer renderer;
    if (!renderer.initialize()) {
        fprintf(stderr, "Failed to initialize renderer\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Load robot model (optional)
    RobotModel robotModel;
    bool useRobotModel = false;
    if (!modelPath.empty()) {
        if (robotModel.load(modelPath)) {
            if (robotModel.initGL()) {
                useRobotModel = true;
                printf("Robot model loaded successfully\n");
            } else {
                fprintf(stderr, "Failed to initialize robot model GL resources\n");
            }
        } else {
            fprintf(stderr, "Failed to load robot model: %s\n", modelPath.c_str());
        }
    }

    // Create text renderer
    TextRenderer textRenderer;
    if (!textRenderer.initialize(fontPath, fontSize)) {
        fprintf(stderr, "Warning: Failed to initialize text renderer. Text display disabled.\n");
    }

    // Create ZMQ receiver
    ZMQReceiver zmqReceiver(zmqAddress);
    zmqReceiver.start();

    // Timing
    double lastTime = glfwGetTime();
    int frameCount = 0;
    double fpsTime = 0.0;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Calculate delta time
        double currentTime = glfwGetTime();
        float deltaTime = static_cast<float>(currentTime - lastTime);
        lastTime = currentTime;

        // FPS counter
        frameCount++;
        fpsTime += deltaTime;
        if (fpsTime >= 1.0) {
            char title[128];
            snprintf(
                title, sizeof(title), "Go2 Visualizer - FPS: %d | %s", frameCount,
                zmqReceiver.isConnected() ? "Connected" : "Disconnected");
            glfwSetWindowTitle(window, title);
            frameCount = 0;
            fpsTime = 0.0;
        }

        // Process input
        glfwPollEvents();
        camera.processKeyboard(window, deltaTime);

        // Get window size
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);

        float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

        // Get data from ZMQ
        std::vector<float> rays = zmqReceiver.getEstimatedRays();
        std::vector<float> pointCloud = zmqReceiver.getPointCloud();
        IMUData imuData = zmqReceiver.getIMUData();
        ControlData controlData = zmqReceiver.getControlData();
        BatteryData batteryData = zmqReceiver.getBatteryData();
        std::vector<MotorData> motorData = zmqReceiver.getMotorData();
        int16_t footForces[4] = {
            zmqReceiver.getFootForce(0), zmqReceiver.getFootForce(1), zmqReceiver.getFootForce(2),
            zmqReceiver.getFootForce(3)};

        // Render
        Mat4 view = camera.getViewMatrix();
        Mat4 projection = camera.getProjectionMatrix(aspectRatio);

        renderer.beginFrame(view, projection);

        // Draw grid (commented out)
        // renderer.drawGrid(10.0f, 20, Vec3(0.3f, 0.3f, 0.3f));

        // Calculate world yaw (IMU yaw + calibration offset)
        float worldYaw = imuData.rpy[2] + g_yawOffset + g_yawOffsetDelta;
        // Normalize yaw to [-π, π]
        while (worldYaw > M_PI)
            worldYaw -= 2.0f * M_PI;
        while (worldYaw < -M_PI)
            worldYaw += 2.0f * M_PI;

        // Select yaw based on frame mode
        float displayYaw = g_globalFrame ? worldYaw : 0.0f;

        // Draw robot (3D model or triangle) - elevated above ground
        if (useRobotModel) {
            // Set joint angles from motor data
            std::vector<float> jointAngles(12);
            for (int i = 0; i < 12 && i < (int)motorData.size(); i++) {
                jointAngles[i] = motorData[i].q;
            }
            robotModel.setJointAngles(jointAngles);
            robotModel.render(view, projection, displayYaw, Vec3(0, 0, robotHeight), robotScale);
        } else {
            renderer.drawRobot(Vec3(0, 0, robotHeight), displayYaw);
        }

        // Draw point cloud (intensity-colored, larger points, distance-based transparency)
        if (!pointCloud.empty()) {
            renderer.drawPointCloud(
                pointCloud, Vec3(0.6f, 0.6f, 0.6f), // Fallback gray color
                pointSize,                          // Point size
                fadeDistance,                       // Max distance for alpha falloff
                0.05f,      // Min alpha (at max distance) - very transparent
                1.0f,       // Max alpha (at origin) - fully opaque
                displayYaw, // Yaw rotation
                true);      // Use intensity-based rainbow colormap
        }

        // Draw estimated rays (blue color) with spheres at endpoints - at origin (z=0)
        if (!rays.empty()) {
            renderer.drawRays(rays, Vec3(0, 0, 0), Vec3(0.5f, 0.7f, 0.9f), 1.5f, displayYaw);
            renderer.drawRayEndpoints(
                rays, Vec3(0, 0, 0), Vec3(0.6f, 0.8f, 1.0f), 8.0f, displayYaw);
        }

        // Draw velocity commands
        // controlData.mode: 0 = DIRECT_CONTROL, 1 = FILTER_POLICY
        // controlData.cmd: command sent to locomotion [vx, vy, wz]
        // controlData.nav_cmd: command input to filter (wireless input in filter mode)

        // Always draw the command going to locomotion (green arrow)
        renderer.drawVelocityArrow(
            Vec3(0, 0, 0), controlData.cmd[0], controlData.cmd[1], displayYaw,
            Vec3(0.2f, 1.0f, 0.2f), arrowScale, arrowWidth,
            arrowHeadSize); // Green for loco command
        renderer.drawAngularVelocityArc(
            Vec3(0, 0, 0), controlData.cmd[2], displayYaw, Vec3(0.2f, 1.0f, 0.2f), arcRadius,
            arrowWidth, arcHeadSize); // Green for loco angular

        // Draw input command (red arrow)
        // In direct mode: cmd is the input (same as loco output)
        // In filter mode: nav_cmd is the input to the filter
        if (controlData.mode == 0) {
            // Direct control mode: draw input in red (same values as cmd)
            renderer.drawVelocityArrow(
                Vec3(0, 0, 0), controlData.cmd[0], controlData.cmd[1], displayYaw,
                Vec3(1.0f, 0.3f, 0.3f), arrowScale, arrowWidth - 1.0f,
                arrowHeadSize); // Red for input
            renderer.drawAngularVelocityArc(
                Vec3(0, 0, 0), controlData.cmd[2], displayYaw, Vec3(1.0f, 0.3f, 0.3f),
                arcRadius + arcGap, arrowWidth - 1.0f, arcHeadSize); // Red for input angular
        } else {
            // Filter policy mode: draw filter input in red
            renderer.drawVelocityArrow(
                Vec3(0, 0, 0), controlData.nav_cmd[0], controlData.nav_cmd[1], displayYaw,
                Vec3(1.0f, 0.3f, 0.3f), arrowScale, arrowWidth - 1.0f,
                arrowHeadSize); // Red for filter input
            renderer.drawAngularVelocityArc(
                Vec3(0, 0, 0), controlData.nav_cmd[2], displayYaw, Vec3(1.0f, 0.3f, 0.3f),
                arcRadius + arcGap, arrowWidth - 1.0f, arcHeadSize); // Red for filter input angular
        }

        renderer.endFrame();

        // Render text overlay (2D)
        textRenderer.setScreenSize(width, height);

        float lineH = textRenderer.getLineHeight();
        char buf[256];

        // ==================== LEFT PANEL ====================
        float leftX = 10.0f;
        float lineY = 10.0f;

        // --- Control Mode ---
        const char *modeStr = (controlData.mode == 0) ? "DIRECT" : "FILTER";
        snprintf(buf, sizeof(buf), "Mode: %s", modeStr);
        textRenderer.renderText(buf, leftX, lineY, 1.0f, 1.0f, 0.5f);
        lineY += lineH * 1.5f;

        // --- IMU Section ---
        textRenderer.renderText("=== IMU ===", leftX, lineY, 0.6f, 0.8f, 1.0f);
        lineY += lineH * 1.1f;

        snprintf(buf, sizeof(buf), "Roll:   %7.2f deg", imuData.rpy[0] * 180.0f / M_PI);
        textRenderer.renderText(buf, leftX, lineY, 0.8f, 0.8f, 1.0f);
        lineY += lineH;

        snprintf(buf, sizeof(buf), "Pitch:  %7.2f deg", imuData.rpy[1] * 180.0f / M_PI);
        textRenderer.renderText(buf, leftX, lineY, 0.8f, 0.8f, 1.0f);
        lineY += lineH;

        snprintf(buf, sizeof(buf), "Yaw:    %7.2f deg", worldYaw * 180.0f / M_PI);
        textRenderer.renderText(buf, leftX, lineY, 0.8f, 0.8f, 1.0f);
        lineY += lineH * 1.2f;

        snprintf(
            buf, sizeof(buf), "Quat:   [%6.3f, %6.3f, %6.3f, %6.3f]", imuData.quaternion[0],
            imuData.quaternion[1], imuData.quaternion[2], imuData.quaternion[3]);
        textRenderer.renderText(buf, leftX, lineY, 0.6f, 0.6f, 0.8f);
        lineY += lineH * 1.2f;

        snprintf(
            buf, sizeof(buf), "Gyro:   [%7.2f, %7.2f, %7.2f]", imuData.gyroscope[0],
            imuData.gyroscope[1], imuData.gyroscope[2]);
        textRenderer.renderText(buf, leftX, lineY, 0.7f, 0.7f, 0.7f);
        lineY += lineH;

        snprintf(
            buf, sizeof(buf), "Accel:  [%7.2f, %7.2f, %7.2f]", imuData.accelerometer[0],
            imuData.accelerometer[1], imuData.accelerometer[2]);
        textRenderer.renderText(buf, leftX, lineY, 0.7f, 0.7f, 0.7f);
        lineY += lineH;

        snprintf(
            buf, sizeof(buf), "Grav:   [%7.2f, %7.2f, %7.2f]", imuData.gravity[0],
            imuData.gravity[1], imuData.gravity[2]);
        textRenderer.renderText(buf, leftX, lineY, 0.7f, 0.7f, 0.7f);
        lineY += lineH * 1.2f;

        snprintf(buf, sizeof(buf), "IMU Temp: %3d C", imuData.temperature);
        textRenderer.renderText(buf, leftX, lineY, 0.6f, 0.6f, 0.6f);
        lineY += lineH * 1.8f;

        // --- Commands Section ---
        textRenderer.renderText("=== Commands ===", leftX, lineY, 0.6f, 1.0f, 0.6f);
        lineY += lineH * 1.1f;

        snprintf(
            buf, sizeof(buf), "Loco:   vx=%6.2f  vy=%6.2f  wz=%6.2f", controlData.cmd[0],
            controlData.cmd[1], controlData.cmd[2]);
        textRenderer.renderText(buf, leftX, lineY, 0.3f, 1.0f, 0.3f);
        lineY += lineH;

        if (controlData.mode == 0) {
            snprintf(
                buf, sizeof(buf), "Input:  vx=%6.2f  vy=%6.2f  wz=%6.2f", controlData.cmd[0],
                controlData.cmd[1], controlData.cmd[2]);
        } else {
            snprintf(
                buf, sizeof(buf), "Input:  vx=%6.2f  vy=%6.2f  wz=%6.2f", controlData.nav_cmd[0],
                controlData.nav_cmd[1], controlData.nav_cmd[2]);
        }
        textRenderer.renderText(buf, leftX, lineY, 1.0f, 0.3f, 0.3f);
        lineY += lineH * 1.8f;

        // --- Foot Forces ---
        textRenderer.renderText("=== Foot Forces ===", leftX, lineY, 0.8f, 0.6f, 1.0f);
        lineY += lineH * 1.1f;

        snprintf(buf, sizeof(buf), "FL: %5d    FR: %5d", footForces[0], footForces[1]);
        textRenderer.renderText(buf, leftX, lineY, 0.7f, 0.7f, 0.9f);
        lineY += lineH;

        snprintf(buf, sizeof(buf), "RL: %5d    RR: %5d", footForces[2], footForces[3]);
        textRenderer.renderText(buf, leftX, lineY, 0.7f, 0.7f, 0.9f);
        lineY += lineH * 1.8f;

        // --- Battery Section ---
        textRenderer.renderText("=== Battery ===", leftX, lineY, 1.0f, 0.8f, 0.3f);
        lineY += lineH * 1.1f;

        float battR = (batteryData.soc < 20) ? 1.0f : 0.3f;
        float battG = (batteryData.soc > 50) ? 1.0f : 0.5f;
        snprintf(buf, sizeof(buf), "SOC:     %3d %%", batteryData.soc);
        textRenderer.renderText(buf, leftX, lineY, battR, battG, 0.3f);
        lineY += lineH;

        snprintf(buf, sizeof(buf), "Current: %6d mA", batteryData.current);
        textRenderer.renderText(buf, leftX, lineY, 0.7f, 0.7f, 0.7f);
        lineY += lineH;

        snprintf(buf, sizeof(buf), "Cycles:  %5d", batteryData.cycle);
        textRenderer.renderText(buf, leftX, lineY, 0.6f, 0.6f, 0.6f);
        lineY += lineH;

        snprintf(
            buf, sizeof(buf), "Status:  0x%02X   Ver: %d.%d", batteryData.status,
            batteryData.version_high, batteryData.version_low);
        textRenderer.renderText(buf, leftX, lineY, 0.5f, 0.5f, 0.5f);
        lineY += lineH;

        // ==================== RIGHT PANEL (Motors) ====================
        // Calculate right panel width based on longest line
        const char *motorHeader = "Name         q        dq      tau    T";
        float motorPanelWidth = textRenderer.getTextWidth(motorHeader) + 20.0f;
        float rightX = width - motorPanelWidth;
        lineY = 10.0f;

        textRenderer.renderText("=== Motors ===", rightX, lineY, 1.0f, 0.6f, 0.6f);
        lineY += lineH * 1.1f;

        // Motor names for Go2 (standard order)
        const char *motorNames[12] = {"FR_hip  ", "FR_thigh", "FR_calf ", "FL_hip  ",
                                      "FL_thigh", "FL_calf ", "RR_hip  ", "RR_thigh",
                                      "RR_calf ", "RL_hip  ", "RL_thigh", "RL_calf "};

        if (motorData.size() >= 12) {
            // Header with fixed column positions
            //          Name       q        dq       tau      T
            textRenderer.renderText(motorHeader, rightX, lineY, 0.5f, 0.5f, 0.5f);
            lineY += lineH * 1.1f;

            for (int i = 0; i < 12; i++) {
                const MotorData &m = motorData[i];

                // Color based on temperature
                float tempR, tempG, tempB;
                if (m.temperature > 70) {
                    tempR = 1.0f;
                    tempG = 0.2f;
                    tempB = 0.2f; // Hot - red
                } else if (m.temperature > 55) {
                    tempR = 1.0f;
                    tempG = 0.6f;
                    tempB = 0.2f; // Warm - orange
                } else {
                    tempR = 0.7f;
                    tempG = 0.8f;
                    tempB = 0.7f; // Normal - greenish
                }

                // Fixed-width format: name(8) + q(8) + dq(8) + tau(7) + temp(4)
                snprintf(
                    buf, sizeof(buf), "%s %7.2f  %7.2f  %6.1f  %3d", motorNames[i], m.q, m.dq,
                    m.tau_est, m.temperature);
                textRenderer.renderText(buf, rightX, lineY, tempR, tempG, tempB);
                lineY += lineH;

                // Add spacing between leg groups (every 3 motors)
                if ((i + 1) % 3 == 0 && i < 11) {
                    lineY += lineH * 0.3f;
                }
            }
        } else {
            textRenderer.renderText("No motor data", rightX, lineY, 0.5f, 0.5f, 0.5f);
            lineY += lineH;
        }

        // ==================== BOTTOM INFO ====================
        // Frame mode indicator (bottom right)
        snprintf(buf, sizeof(buf), "[%s FRAME]", g_globalFrame ? "GLOBAL" : "BODY");
        float frameTextWidth = textRenderer.getTextWidth(buf);
        textRenderer.renderText(
            buf, width - frameTextWidth - 10, height - lineH - 10, 0.6f, 0.6f, 0.6f);

        // Point cloud and ray count (bottom left)
        snprintf(buf, sizeof(buf), "Points: %zu  Rays: %zu", pointCloud.size() / 4, rays.size());
        textRenderer.renderText(buf, 10, height - lineH - 10, 0.5f, 0.5f, 0.5f);

        // Swap buffers
        glfwSwapBuffers(window);
    }

    // Cleanup
    zmqReceiver.stop();
    textRenderer.cleanup();
    renderer.cleanup();
    g_camera = nullptr;

    glfwDestroyWindow(window);
    glfwTerminate();

    printf("Visualizer closed.\n");
    return 0;
}
