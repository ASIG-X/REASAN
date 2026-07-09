#define STB_TRUETYPE_IMPLEMENTATION
#include "../stb_truetype.h"

#include "text_renderer.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

// Text vertex shader
static const char *textVertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>

out vec2 TexCoords;

uniform mat4 uProjection;

void main() {
    gl_Position = uProjection * vec4(vertex.xy, 0.0, 1.0);
    TexCoords = vertex.zw;
}
)";

// Text fragment shader
static const char *textFragmentShaderSource = R"(
#version 330 core
in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D uTexture;
uniform vec3 uTextColor;

void main() {
    float alpha = texture(uTexture, TexCoords).r;
    FragColor = vec4(uTextColor, alpha);
}
)";

TextRenderer::TextRenderer()
    : initialized_(false)
    , screenWidth_(1280)
    , screenHeight_(720)
    , fontSize_(16.0f)
    , lineHeight_(20.0f)
    , shaderProgram_(0)
    , vao_(0)
    , vbo_(0)
    , fontTexture_(0)
    , projectionLocation_(-1)
    , textColorLocation_(-1)
    , textureSamplerLocation_(-1) {
    std::memset(charInfo_, 0, sizeof(charInfo_));
}

TextRenderer::~TextRenderer() { cleanup(); }

bool TextRenderer::compileShader(GLuint &shader, GLenum type, const char *source) {
    shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        fprintf(stderr, "Text shader compilation error: %s\n", infoLog);
        return false;
    }
    return true;
}

bool TextRenderer::linkProgram(GLuint &program, GLuint vertexShader, GLuint fragmentShader) {
    program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        fprintf(stderr, "Text program linking error: %s\n", infoLog);
        return false;
    }
    return true;
}

bool TextRenderer::initialize(const std::string &fontPath, float fontSize) {
    fontSize_ = fontSize;

    // Load font file
    std::ifstream fontFile(fontPath, std::ios::binary | std::ios::ate);
    if (!fontFile.is_open()) {
        fprintf(stderr, "Failed to open font file: %s\n", fontPath.c_str());
        return false;
    }

    std::streamsize size = fontFile.tellg();
    fontFile.seekg(0, std::ios::beg);

    std::vector<unsigned char> fontBuffer(size);
    if (!fontFile.read(reinterpret_cast<char *>(fontBuffer.data()), size)) {
        fprintf(stderr, "Failed to read font file: %s\n", fontPath.c_str());
        return false;
    }

    // Initialize stb_truetype
    stbtt_fontinfo fontInfo;
    if (!stbtt_InitFont(&fontInfo, fontBuffer.data(), 0)) {
        fprintf(stderr, "Failed to initialize font\n");
        return false;
    }

    // Get font metrics
    float scale = stbtt_ScaleForPixelHeight(&fontInfo, fontSize_);
    int ascent, descent, lineGap;
    stbtt_GetFontVMetrics(&fontInfo, &ascent, &descent, &lineGap);
    lineHeight_ = (ascent - descent + lineGap) * scale;

    // Create font atlas
    std::vector<unsigned char> atlasData(ATLAS_SIZE * ATLAS_SIZE, 0);

    // Pack characters into atlas
    int x = 1, y = 1;
    int rowHeight = 0;

    for (int c = FIRST_CHAR; c < FIRST_CHAR + NUM_CHARS; c++) {
        int idx = c - FIRST_CHAR;

        // Always get advance width first (needed for space and other chars)
        int advanceWidth, leftSideBearing;
        stbtt_GetCodepointHMetrics(&fontInfo, c, &advanceWidth, &leftSideBearing);
        charInfo_[idx].xadvance = advanceWidth * scale;

        int w, h, xoff, yoff;
        unsigned char *bitmap =
            stbtt_GetCodepointBitmap(&fontInfo, 0, scale, c, &w, &h, &xoff, &yoff);

        if (bitmap && w > 0 && h > 0) {
            // Check if we need to move to next row
            if (x + w + 1 >= ATLAS_SIZE) {
                x = 1;
                y += rowHeight + 1;
                rowHeight = 0;
            }

            // Check if we've run out of space
            if (y + h + 1 >= ATLAS_SIZE) {
                fprintf(stderr, "Font atlas too small!\n");
                stbtt_FreeBitmap(bitmap, nullptr);
                break;
            }

            // Copy bitmap to atlas
            for (int row = 0; row < h; row++) {
                for (int col = 0; col < w; col++) {
                    atlasData[(y + row) * ATLAS_SIZE + (x + col)] = bitmap[row * w + col];
                }
            }

            // Store character info
            charInfo_[idx].x0 = (float)x / ATLAS_SIZE;
            charInfo_[idx].y0 = (float)y / ATLAS_SIZE;
            charInfo_[idx].x1 = (float)(x + w) / ATLAS_SIZE;
            charInfo_[idx].y1 = (float)(y + h) / ATLAS_SIZE;
            charInfo_[idx].xoff = (float)xoff;
            charInfo_[idx].yoff = (float)yoff;
            charInfo_[idx].width = (float)w;
            charInfo_[idx].height = (float)h;

            x += w + 1;
            if (h > rowHeight)
                rowHeight = h;

            stbtt_FreeBitmap(bitmap, nullptr);
        } else {
            // Character has no bitmap (like space) - just set zero size
            charInfo_[idx].x0 = 0;
            charInfo_[idx].y0 = 0;
            charInfo_[idx].x1 = 0;
            charInfo_[idx].y1 = 0;
            charInfo_[idx].xoff = 0;
            charInfo_[idx].yoff = 0;
            charInfo_[idx].width = 0;
            charInfo_[idx].height = 0;
            // xadvance already set above

            if (bitmap) {
                stbtt_FreeBitmap(bitmap, nullptr);
            }
        }
    }

    // Create OpenGL texture
    glGenTextures(1, &fontTexture_);
    glBindTexture(GL_TEXTURE_2D, fontTexture_);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RED, ATLAS_SIZE, ATLAS_SIZE, 0, GL_RED, GL_UNSIGNED_BYTE,
        atlasData.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Compile shaders
    GLuint vertexShader, fragmentShader;
    if (!compileShader(vertexShader, GL_VERTEX_SHADER, textVertexShaderSource)) {
        return false;
    }
    if (!compileShader(fragmentShader, GL_FRAGMENT_SHADER, textFragmentShaderSource)) {
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

    // Get uniform locations
    projectionLocation_ = glGetUniformLocation(shaderProgram_, "uProjection");
    textColorLocation_ = glGetUniformLocation(shaderProgram_, "uTextColor");
    textureSamplerLocation_ = glGetUniformLocation(shaderProgram_, "uTexture");

    // Create VAO and VBO
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    initialized_ = true;
    printf("Text renderer initialized with font: %s (size: %.0f)\n", fontPath.c_str(), fontSize_);
    return true;
}

void TextRenderer::cleanup() {
    if (vbo_) {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }
    if (vao_) {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
    if (fontTexture_) {
        glDeleteTextures(1, &fontTexture_);
        fontTexture_ = 0;
    }
    if (shaderProgram_) {
        glDeleteProgram(shaderProgram_);
        shaderProgram_ = 0;
    }
    initialized_ = false;
}

void TextRenderer::setScreenSize(int width, int height) {
    screenWidth_ = width;
    screenHeight_ = height;
}

void TextRenderer::renderText(
    const std::string &text, float x, float y, float r, float g, float b) {
    if (!initialized_)
        return;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    glUseProgram(shaderProgram_);

    // Create orthographic projection matrix
    float projection[16] = {
        2.0f / screenWidth_,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        -2.0f / screenHeight_,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        -1.0f,
        0.0f,
        -1.0f,
        1.0f,
        0.0f,
        1.0f};

    glUniformMatrix4fv(projectionLocation_, 1, GL_FALSE, projection);
    glUniform3f(textColorLocation_, r, g, b);
    glUniform1i(textureSamplerLocation_, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, fontTexture_);

    glBindVertexArray(vao_);

    float cursorX = x;
    float cursorY = y + fontSize_; // Baseline

    for (char c : text) {
        if (c == '\n') {
            cursorX = x;
            cursorY += lineHeight_;
            continue;
        }

        if (c < FIRST_CHAR || c >= FIRST_CHAR + NUM_CHARS)
            continue;

        int idx = c - FIRST_CHAR;
        const CharInfo &ch = charInfo_[idx];

        float xpos = cursorX + ch.xoff;
        float ypos = cursorY + ch.yoff;
        float w = ch.width;
        float h = ch.height;

        // Generate 6 vertices (2 triangles)
        float vertices[6][4] = {
            {xpos, ypos, ch.x0, ch.y0},         {xpos, ypos + h, ch.x0, ch.y1},
            {xpos + w, ypos + h, ch.x1, ch.y1},

            {xpos, ypos, ch.x0, ch.y0},         {xpos + w, ypos + h, ch.x1, ch.y1},
            {xpos + w, ypos, ch.x1, ch.y0},
        };

        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        cursorX += ch.xadvance;
    }

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
}

float TextRenderer::getTextWidth(const std::string &text) {
    float width = 0.0f;
    float maxWidth = 0.0f;

    for (char c : text) {
        if (c == '\n') {
            if (width > maxWidth)
                maxWidth = width;
            width = 0.0f;
            continue;
        }

        if (c < FIRST_CHAR || c >= FIRST_CHAR + NUM_CHARS)
            continue;

        int idx = c - FIRST_CHAR;
        width += charInfo_[idx].xadvance;
    }

    return (width > maxWidth) ? width : maxWidth;
}
