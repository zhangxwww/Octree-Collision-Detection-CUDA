#pragma once

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class KeyboardController
{
public:
    KeyboardController() {};
    void callback(GLFWwindow* window, int key, int scancode, int action, int mode);
    bool shouldClose() const;
    void step(float delta);
    glm::vec3 getEyePos() const;

private:
    bool goLeft = false;
    bool goRight = false;
    bool goStraight = false;
    bool goBack = false;
    bool goUp = false;
    bool goDown = false;

    float theta = 0.0f;
    float radius = 20.0f;
    float height = 0.0f;

    bool _shouldClose = false;
};

