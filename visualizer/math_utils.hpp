#pragma once

#include <cmath>

// Simple 3D vector
struct Vec3 {
    float x, y, z;

    Vec3()
        : x(0)
        , y(0)
        , z(0) {}
    Vec3(float x, float y, float z)
        : x(x)
        , y(y)
        , z(z) {}

    Vec3 operator+(const Vec3 &v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3 &v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }

    Vec3 &operator+=(const Vec3 &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    Vec3 &operator-=(const Vec3 &v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    float dot(const Vec3 &v) const { return x * v.x + y * v.y + z * v.z; }

    Vec3 cross(const Vec3 &v) const {
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }

    float length() const { return std::sqrt(x * x + y * y + z * z); }

    Vec3 normalized() const {
        float len = length();
        if (len > 0.0001f) {
            return *this / len;
        }
        return Vec3(0, 0, 1);
    }
};

// 4x4 Matrix (column-major for OpenGL)
struct Mat4 {
    float m[16];

    Mat4() {
        for (int i = 0; i < 16; i++)
            m[i] = 0;
        m[0] = m[5] = m[10] = m[15] = 1; // Identity
    }

    static Mat4 identity() { return Mat4(); }

    static Mat4 perspective(float fov, float aspect, float near, float far) {
        Mat4 result;
        float tanHalfFov = std::tan(fov / 2.0f);

        for (int i = 0; i < 16; i++)
            result.m[i] = 0;

        result.m[0] = 1.0f / (aspect * tanHalfFov);
        result.m[5] = 1.0f / tanHalfFov;
        result.m[10] = -(far + near) / (far - near);
        result.m[11] = -1.0f;
        result.m[14] = -(2.0f * far * near) / (far - near);

        return result;
    }

    static Mat4 lookAt(const Vec3 &eye, const Vec3 &center, const Vec3 &up) {
        Vec3 f = (center - eye).normalized();
        Vec3 s = f.cross(up).normalized();
        Vec3 u = s.cross(f);

        Mat4 result;
        result.m[0] = s.x;
        result.m[4] = s.y;
        result.m[8] = s.z;
        result.m[1] = u.x;
        result.m[5] = u.y;
        result.m[9] = u.z;
        result.m[2] = -f.x;
        result.m[6] = -f.y;
        result.m[10] = -f.z;
        result.m[12] = -s.dot(eye);
        result.m[13] = -u.dot(eye);
        result.m[14] = f.dot(eye);
        result.m[15] = 1.0f;

        return result;
    }

    Mat4 operator*(const Mat4 &other) const {
        Mat4 result;
        for (int i = 0; i < 16; i++)
            result.m[i] = 0;

        for (int col = 0; col < 4; col++) {
            for (int row = 0; row < 4; row++) {
                for (int k = 0; k < 4; k++) {
                    result.m[col * 4 + row] += m[k * 4 + row] * other.m[col * 4 + k];
                }
            }
        }
        return result;
    }

    const float *data() const { return m; }
};
