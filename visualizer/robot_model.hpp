#pragma once

#include "math_utils.hpp"
#include <GL/glew.h>
#include <array>
#include <map>
#include <string>
#include <vector>

// Mesh data for a single OBJ file
struct Mesh {
    std::vector<float> vertices; // x,y,z interleaved
    std::vector<float> normals;  // nx,ny,nz interleaved
    std::vector<unsigned int> indices;
    Vec3 color;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint nbo = 0;
    GLuint ebo = 0;
};

// A body/link in the robot
struct Body {
    std::string name;
    std::string parentName;
    Vec3 pos;                                 // Position relative to parent
    std::array<float, 4> quat = {1, 0, 0, 0}; // Quaternion (w,x,y,z) relative to parent
    std::vector<std::string> meshNames;       // Meshes attached to this body
    std::vector<Vec3> meshColors;             // Color for each mesh
    std::vector<std::array<float, 4>>
        meshQuats; // Rotation for each mesh (some have local rotation)

    // Joint info (if this body has a joint)
    std::string jointName;
    Vec3 jointAxis;          // Rotation axis
    float jointAngle = 0.0f; // Current joint angle
};

class RobotModel {
  public:
    RobotModel();
    ~RobotModel();

    // Load MuJoCo XML and OBJ meshes
    bool load(const std::string &xmlPath);

    // Initialize OpenGL resources
    bool initGL();
    void cleanupGL();

    // Set joint angles (12 joints for Go2)
    // Order: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
    //        RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
    void setJointAngles(const std::vector<float> &angles);

    // Render the robot
    // view, projection: camera matrices
    // baseYaw: yaw rotation of base_link
    // position: base position offset (x, y, z)
    // scale: uniform scale factor for the robot model
    void render(
        const Mat4 &view,
        const Mat4 &projection,
        float baseYaw = 0.0f,
        const Vec3 &position = Vec3(0, 0, 0),
        float scale = 1.0f);

    bool isLoaded() const { return loaded_; }

  private:
    bool parseMujocoXML(const std::string &xmlPath);
    bool loadMesh(const std::string &objPath, const std::string &name);
    void parseBody(void *bodyElement, const std::string &parentName);
    Mat4 computeBodyTransform(const std::string &bodyName, float baseYaw);
    Mat4 quatToMat4(float w, float x, float y, float z);

    std::string meshDir_;
    std::map<std::string, Mesh> meshes_;
    std::map<std::string, Body> bodies_;
    std::vector<std::string> bodyOrder_; // For traversal

    // Joint name to body name mapping
    std::map<std::string, std::string> jointToBody_;

    // Shader for robot rendering
    GLuint shaderProgram_ = 0;
    GLint mvpLocation_ = -1;
    GLint modelLocation_ = -1;
    GLint colorLocation_ = -1;
    GLint lightDirLocation_ = -1;

    bool loaded_ = false;
    bool glInitialized_ = false;
};
