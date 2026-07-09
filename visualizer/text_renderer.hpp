#pragma once

#include <GL/glew.h>
#include <string>
#include <vector>

class TextRenderer {
  public:
    TextRenderer();
    ~TextRenderer();

    // Initialize with a TTF font file and desired font size
    bool initialize(const std::string &fontPath, float fontSize);
    void cleanup();

    // Set screen dimensions (call when window resizes)
    void setScreenSize(int width, int height);

    // Render text at screen position (pixels from top-left)
    void renderText(const std::string &text, float x, float y, float r, float g, float b);

    // Convenience overloads
    void renderText(const std::string &text, float x, float y) {
        renderText(text, x, y, 1.0f, 1.0f, 1.0f); // White
    }

    // Get text dimensions
    float getTextWidth(const std::string &text);
    float getLineHeight() const { return lineHeight_; }

  private:
    bool initialized_;
    int screenWidth_;
    int screenHeight_;
    float fontSize_;
    float lineHeight_;

    // OpenGL resources
    GLuint shaderProgram_;
    GLuint vao_;
    GLuint vbo_;
    GLuint fontTexture_;

    // Shader uniform locations
    GLint projectionLocation_;
    GLint textColorLocation_;
    GLint textureSamplerLocation_;

    // Character info (ASCII 32-126)
    static const int FIRST_CHAR = 32;
    static const int NUM_CHARS = 95;
    static const int ATLAS_SIZE = 512;

    struct CharInfo {
        float x0, y0, x1, y1; // Texture coordinates
        float xoff, yoff;     // Offset from cursor
        float xadvance;       // How much to advance cursor
        float width, height;  // Character size in pixels
    };
    CharInfo charInfo_[NUM_CHARS];

    bool compileShader(GLuint &shader, GLenum type, const char *source);
    bool linkProgram(GLuint &program, GLuint vertexShader, GLuint fragmentShader);
};
