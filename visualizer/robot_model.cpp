#include "robot_model.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <tinyxml2.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace tinyxml2;

// Vertex shader with lighting
static const char *robotVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uMVP;
uniform mat4 uModel;

out vec3 vNormal;
out vec3 vFragPos;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vFragPos = vec3(uModel * vec4(aPos, 1.0));
    vNormal = mat3(transpose(inverse(uModel))) * aNormal;
}
)";

// Fragment shader with simple lighting
static const char *robotFragmentShader = R"(
#version 330 core
out vec4 FragColor;

in vec3 vNormal;
in vec3 vFragPos;

uniform vec3 uColor;
uniform vec3 uLightDir;

void main() {
    // Ambient
    float ambient = 0.3;
    
    // Diffuse
    vec3 norm = normalize(vNormal);
    float diff = max(dot(norm, normalize(uLightDir)), 0.0);
    
    // Combine
    vec3 result = (ambient + 0.7 * diff) * uColor;
    FragColor = vec4(result, 1.0);
}
)";

RobotModel::RobotModel() {}

RobotModel::~RobotModel() { cleanupGL(); }

bool RobotModel::load(const std::string &xmlPath) {
    if (!parseMujocoXML(xmlPath)) {
        return false;
    }

    // Set default standing pose
    // Joint order: FR (hip, thigh, calf), FL, RR, RL
    std::vector<float> defaultPose = {
        -0.1f, 0.8f, -1.5f, // FR: hip, thigh, calf
        0.1f,  0.8f, -1.5f, // FL: hip, thigh, calf
        -0.1f, 1.0f, -1.5f, // RR: hip, thigh, calf
        0.1f,  1.0f, -1.5f  // RL: hip, thigh, calf
    };
    setJointAngles(defaultPose);

    loaded_ = true;
    return true;
}

bool RobotModel::parseMujocoXML(const std::string &xmlPath) {
    XMLDocument doc;
    if (doc.LoadFile(xmlPath.c_str()) != XML_SUCCESS) {
        fprintf(stderr, "Failed to load XML: %s\n", xmlPath.c_str());
        return false;
    }

    // Get mesh directory from compiler element
    XMLElement *mujoco = doc.FirstChildElement("mujoco");
    if (!mujoco) {
        fprintf(stderr, "No <mujoco> element found\n");
        return false;
    }

    XMLElement *compiler = mujoco->FirstChildElement("compiler");
    if (compiler && compiler->Attribute("meshdir")) {
        meshDir_ = compiler->Attribute("meshdir");
    } else {
        meshDir_ = "assets";
    }

    // Construct full mesh path
    size_t lastSlash = xmlPath.find_last_of("/\\");
    std::string basePath = (lastSlash != std::string::npos) ? xmlPath.substr(0, lastSlash + 1) : "";
    meshDir_ = basePath + meshDir_ + "/";

    printf("Mesh directory: %s\n", meshDir_.c_str());

    // Parse materials for colors
    std::map<std::string, Vec3> materials;
    XMLElement *asset = mujoco->FirstChildElement("asset");
    if (asset) {
        for (XMLElement *mat = asset->FirstChildElement("material"); mat;
             mat = mat->NextSiblingElement("material")) {
            const char *name = mat->Attribute("name");
            const char *rgba = mat->Attribute("rgba");
            if (name && rgba) {
                float r, g, b, a;
                sscanf(rgba, "%f %f %f %f", &r, &g, &b, &a);
                materials[name] = Vec3(r, g, b);
            }
        }
    }

    // Load all meshes referenced in asset
    if (asset) {
        for (XMLElement *mesh = asset->FirstChildElement("mesh"); mesh;
             mesh = mesh->NextSiblingElement("mesh")) {
            const char *file = mesh->Attribute("file");
            if (file) {
                std::string meshName = file;
                // Remove .obj extension for name
                size_t dotPos = meshName.find_last_of('.');
                if (dotPos != std::string::npos) {
                    meshName = meshName.substr(0, dotPos);
                }

                std::string fullPath = meshDir_ + file;
                if (!loadMesh(fullPath, meshName)) {
                    fprintf(stderr, "Warning: Failed to load mesh %s\n", fullPath.c_str());
                }
            }
        }
    }

    // Parse worldbody
    XMLElement *worldbody = mujoco->FirstChildElement("worldbody");
    if (!worldbody) {
        fprintf(stderr, "No <worldbody> element found\n");
        return false;
    }

    // Parse body hierarchy recursively
    XMLElement *baseBody = worldbody->FirstChildElement("body");
    if (baseBody) {
        parseBody(baseBody, "");
    }

    // Set up material colors for bodies
    for (auto &[bodyName, body] : bodies_) {
        // Default color
        for (size_t i = 0; i < body.meshNames.size(); i++) {
            if (i < body.meshColors.size()) {
                // Already set from parsing
            }
        }
    }

    printf("Loaded %zu meshes, %zu bodies\n", meshes_.size(), bodies_.size());
    return true;
}

void RobotModel::parseBody(void *element, const std::string &parentName) {
    XMLElement *bodyElem = static_cast<XMLElement *>(element);

    Body body;
    body.name = bodyElem->Attribute("name") ? bodyElem->Attribute("name") : "";
    body.parentName = parentName;
    body.pos = Vec3(0, 0, 0);
    body.quat[0] = 1;
    body.quat[1] = 0;
    body.quat[2] = 0;
    body.quat[3] = 0;

    // Parse position
    const char *posAttr = bodyElem->Attribute("pos");
    if (posAttr) {
        sscanf(posAttr, "%f %f %f", &body.pos.x, &body.pos.y, &body.pos.z);
    }

    // Parse joint (if any)
    XMLElement *joint = bodyElem->FirstChildElement("joint");
    if (joint) {
        body.jointName = joint->Attribute("name") ? joint->Attribute("name") : "";
        jointToBody_[body.jointName] = body.name;

        // Get axis - default is Y axis based on MuJoCo defaults in the file
        body.jointAxis = Vec3(0, 1, 0);
        const char *axisAttr = joint->Attribute("axis");
        if (axisAttr) {
            sscanf(axisAttr, "%f %f %f", &body.jointAxis.x, &body.jointAxis.y, &body.jointAxis.z);
        }

        // Check class for axis override
        const char *classAttr = joint->Attribute("class");
        if (classAttr && strcmp(classAttr, "abduction") == 0) {
            body.jointAxis = Vec3(1, 0, 0); // X axis for hip abduction
        }
    }

    // Parse visual geoms (meshes)
    for (XMLElement *geom = bodyElem->FirstChildElement("geom"); geom;
         geom = geom->NextSiblingElement("geom")) {

        const char *classAttr = geom->Attribute("class");
        if (classAttr && strcmp(classAttr, "visual") == 0) {
            const char *meshAttr = geom->Attribute("mesh");
            if (meshAttr) {
                body.meshNames.push_back(meshAttr);

                // Get color from material
                Vec3 color(0.5f, 0.5f, 0.5f); // Default gray
                const char *matAttr = geom->Attribute("material");
                if (matAttr) {
                    if (strcmp(matAttr, "black") == 0) {
                        color = Vec3(0.1f, 0.1f, 0.1f);
                    } else if (strcmp(matAttr, "white") == 0) {
                        color = Vec3(0.95f, 0.95f, 0.95f);
                    } else if (strcmp(matAttr, "gray") == 0) {
                        color = Vec3(0.67f, 0.69f, 0.77f);
                    } else if (strcmp(matAttr, "metal") == 0) {
                        color = Vec3(0.9f, 0.95f, 0.95f);
                    }
                }
                body.meshColors.push_back(color);

                // Get mesh rotation (quat)
                float meshQuat[4] = {1, 0, 0, 0};
                const char *quatAttr = geom->Attribute("quat");
                if (quatAttr) {
                    sscanf(
                        quatAttr, "%f %f %f %f", &meshQuat[0], &meshQuat[1], &meshQuat[2],
                        &meshQuat[3]);
                }
                // Store in a simpler way - we'll handle this during rendering
            }
        }
    }

    bodies_[body.name] = body;
    bodyOrder_.push_back(body.name);

    // Recursively parse child bodies
    for (XMLElement *child = bodyElem->FirstChildElement("body"); child;
         child = child->NextSiblingElement("body")) {
        parseBody(child, body.name);
    }
}

bool RobotModel::loadMesh(const std::string &objPath, const std::string &name) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objPath.c_str());

    if (!ret) {
        fprintf(stderr, "TinyOBJ error: %s\n", err.c_str());
        return false;
    }

    Mesh mesh;
    mesh.color = Vec3(0.5f, 0.5f, 0.5f);

    // Process all shapes
    for (const auto &shape : shapes) {
        for (size_t f = 0; f < shape.mesh.indices.size(); f++) {
            tinyobj::index_t idx = shape.mesh.indices[f];

            mesh.vertices.push_back(attrib.vertices[3 * idx.vertex_index + 0]);
            mesh.vertices.push_back(attrib.vertices[3 * idx.vertex_index + 1]);
            mesh.vertices.push_back(attrib.vertices[3 * idx.vertex_index + 2]);

            if (idx.normal_index >= 0) {
                mesh.normals.push_back(attrib.normals[3 * idx.normal_index + 0]);
                mesh.normals.push_back(attrib.normals[3 * idx.normal_index + 1]);
                mesh.normals.push_back(attrib.normals[3 * idx.normal_index + 2]);
            } else {
                // Default normal (will calculate later if needed)
                mesh.normals.push_back(0);
                mesh.normals.push_back(0);
                mesh.normals.push_back(1);
            }

            mesh.indices.push_back(mesh.indices.size());
        }
    }

    meshes_[name] = std::move(mesh);
    printf("Loaded mesh: %s (%zu vertices)\n", name.c_str(), meshes_[name].vertices.size() / 3);
    return true;
}

bool RobotModel::initGL() {
    if (glInitialized_)
        return true;

    // Compile shaders
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &robotVertexShader, nullptr);
    glCompileShader(vs);

    GLint success;
    glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(vs, 512, nullptr, log);
        fprintf(stderr, "Vertex shader error: %s\n", log);
        return false;
    }

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &robotFragmentShader, nullptr);
    glCompileShader(fs);

    glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(fs, 512, nullptr, log);
        fprintf(stderr, "Fragment shader error: %s\n", log);
        return false;
    }

    shaderProgram_ = glCreateProgram();
    glAttachShader(shaderProgram_, vs);
    glAttachShader(shaderProgram_, fs);
    glLinkProgram(shaderProgram_);

    glGetProgramiv(shaderProgram_, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(shaderProgram_, 512, nullptr, log);
        fprintf(stderr, "Shader link error: %s\n", log);
        return false;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    mvpLocation_ = glGetUniformLocation(shaderProgram_, "uMVP");
    modelLocation_ = glGetUniformLocation(shaderProgram_, "uModel");
    colorLocation_ = glGetUniformLocation(shaderProgram_, "uColor");
    lightDirLocation_ = glGetUniformLocation(shaderProgram_, "uLightDir");

    // Create VAO/VBO for each mesh
    for (auto &[name, mesh] : meshes_) {
        glGenVertexArrays(1, &mesh.vao);
        glGenBuffers(1, &mesh.vbo);
        glGenBuffers(1, &mesh.nbo);

        glBindVertexArray(mesh.vao);

        // Vertices
        glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
        glBufferData(
            GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(float), mesh.vertices.data(),
            GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);

        // Normals
        glBindBuffer(GL_ARRAY_BUFFER, mesh.nbo);
        glBufferData(
            GL_ARRAY_BUFFER, mesh.normals.size() * sizeof(float), mesh.normals.data(),
            GL_STATIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);
    }

    glInitialized_ = true;
    printf("Robot model GL initialized\n");
    return true;
}

void RobotModel::cleanupGL() {
    if (!glInitialized_)
        return;

    for (auto &[name, mesh] : meshes_) {
        if (mesh.vao)
            glDeleteVertexArrays(1, &mesh.vao);
        if (mesh.vbo)
            glDeleteBuffers(1, &mesh.vbo);
        if (mesh.nbo)
            glDeleteBuffers(1, &mesh.nbo);
        if (mesh.ebo)
            glDeleteBuffers(1, &mesh.ebo);
    }

    if (shaderProgram_)
        glDeleteProgram(shaderProgram_);

    glInitialized_ = false;
}

void RobotModel::setJointAngles(const std::vector<float> &angles) {
    // Go2 joint order in motorData:
    // FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
    // RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf

    const char *jointNames[12] = {"FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                                  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                                  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"};

    for (int i = 0; i < 12 && i < (int)angles.size(); i++) {
        auto it = jointToBody_.find(jointNames[i]);
        if (it != jointToBody_.end()) {
            auto bodyIt = bodies_.find(it->second);
            if (bodyIt != bodies_.end()) {
                bodyIt->second.jointAngle = angles[i];
            }
        }
    }
}

Mat4 RobotModel::quatToMat4(float w, float x, float y, float z) {
    Mat4 m;

    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;

    m.m[0] = 1 - 2 * (yy + zz);
    m.m[4] = 2 * (xy - wz);
    m.m[8] = 2 * (xz + wy);
    m.m[12] = 0;
    m.m[1] = 2 * (xy + wz);
    m.m[5] = 1 - 2 * (xx + zz);
    m.m[9] = 2 * (yz - wx);
    m.m[13] = 0;
    m.m[2] = 2 * (xz - wy);
    m.m[6] = 2 * (yz + wx);
    m.m[10] = 1 - 2 * (xx + yy);
    m.m[14] = 0;
    m.m[3] = 0;
    m.m[7] = 0;
    m.m[11] = 0;
    m.m[15] = 1;

    return m;
}

Mat4 RobotModel::computeBodyTransform(const std::string &bodyName, float baseYaw) {
    auto it = bodies_.find(bodyName);
    if (it == bodies_.end()) {
        return Mat4::identity();
    }

    const Body &body = it->second;

    // Start with parent transform
    Mat4 parentTransform = Mat4::identity();
    if (!body.parentName.empty()) {
        parentTransform = computeBodyTransform(body.parentName, baseYaw);
    } else {
        // Base link - apply base yaw rotation
        float c = std::cos(baseYaw);
        float s = std::sin(baseYaw);
        parentTransform.m[0] = c;
        parentTransform.m[4] = -s;
        parentTransform.m[1] = s;
        parentTransform.m[5] = c;
    }

    // Translation
    Mat4 translation = Mat4::identity();
    translation.m[12] = body.pos.x;
    translation.m[13] = body.pos.y;
    translation.m[14] = body.pos.z;

    // Joint rotation (if any)
    Mat4 jointRotation = Mat4::identity();
    if (!body.jointName.empty()) {
        float angle = body.jointAngle;
        float c = std::cos(angle);
        float s = std::sin(angle);

        // Rotation around joint axis
        Vec3 ax = body.jointAxis;
        float x = ax.x, y = ax.y, z = ax.z;

        jointRotation.m[0] = c + x * x * (1 - c);
        jointRotation.m[4] = x * y * (1 - c) - z * s;
        jointRotation.m[8] = x * z * (1 - c) + y * s;
        jointRotation.m[1] = y * x * (1 - c) + z * s;
        jointRotation.m[5] = c + y * y * (1 - c);
        jointRotation.m[9] = y * z * (1 - c) - x * s;
        jointRotation.m[2] = z * x * (1 - c) - y * s;
        jointRotation.m[6] = z * y * (1 - c) + x * s;
        jointRotation.m[10] = c + z * z * (1 - c);
    }

    return parentTransform * translation * jointRotation;
}

void RobotModel::render(
    const Mat4 &view, const Mat4 &projection, float baseYaw, const Vec3 &position, float scale) {
    if (!loaded_ || !glInitialized_)
        return;

    glUseProgram(shaderProgram_);

    // Light direction (from top-right-front)
    glUniform3f(lightDirLocation_, 0.5f, 0.3f, 1.0f);

    // Create base translation matrix for position offset
    Mat4 baseTranslation = Mat4::identity();
    baseTranslation.m[12] = position.x;
    baseTranslation.m[13] = position.y;
    baseTranslation.m[14] = position.z;

    // Create scale matrix
    Mat4 scaleMatrix = Mat4::identity();
    scaleMatrix.m[0] = scale;
    scaleMatrix.m[5] = scale;
    scaleMatrix.m[10] = scale;

    // Render each body's meshes
    for (const auto &bodyName : bodyOrder_) {
        auto bodyIt = bodies_.find(bodyName);
        if (bodyIt == bodies_.end())
            continue;

        const Body &body = bodyIt->second;
        Mat4 modelMatrix = baseTranslation * scaleMatrix * computeBodyTransform(bodyName, baseYaw);

        for (size_t i = 0; i < body.meshNames.size(); i++) {
            const std::string &meshName = body.meshNames[i];
            auto meshIt = meshes_.find(meshName);
            if (meshIt == meshes_.end())
                continue;

            const Mesh &mesh = meshIt->second;

            Mat4 mvp = projection * view * modelMatrix;
            glUniformMatrix4fv(mvpLocation_, 1, GL_FALSE, mvp.data());
            glUniformMatrix4fv(modelLocation_, 1, GL_FALSE, modelMatrix.data());

            // Set color
            Vec3 color = (i < body.meshColors.size()) ? body.meshColors[i] : Vec3(0.5f, 0.5f, 0.5f);
            glUniform3f(colorLocation_, color.x, color.y, color.z);

            glBindVertexArray(mesh.vao);
            glDrawArrays(GL_TRIANGLES, 0, mesh.vertices.size() / 3);
        }
    }

    glBindVertexArray(0);
}
