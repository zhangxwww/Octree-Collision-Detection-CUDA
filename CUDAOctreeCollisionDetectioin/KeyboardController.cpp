#include "KeyboardController.h"

void KeyboardController::callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        _shouldClose = true;
    }
    else if (key == GLFW_KEY_D) {
        if (action == GLFW_PRESS) {
            goLeft = true;
        }
        else if (action == GLFW_RELEASE) {
            goLeft = false;
        }
    }
    else if (key == GLFW_KEY_A) {
        if (action == GLFW_PRESS) {
            goRight = true;
        }
        else if (action == GLFW_RELEASE) {
            goRight = false;
        }
    }
    else if (key == GLFW_KEY_W) {
        if (action == GLFW_PRESS) {
            goStraight = true;
        }
        else if (action == GLFW_RELEASE) {
            goStraight = false;
        }
    }
    else if (key == GLFW_KEY_S) {
        if (action == GLFW_PRESS) {
            goBack = true;
        }
        else if (action == GLFW_RELEASE) {
            goBack = false;
        }
    }
    else if (key == GLFW_KEY_Q) {
        if (action == GLFW_PRESS) {
            goDown = true;
        }
        else if (action == GLFW_RELEASE) {
            goDown = false;
        }
    }
    else if (key == GLFW_KEY_E) {
        if (action == GLFW_PRESS) {
            goUp = true;
        }
        else if (action == GLFW_RELEASE) {
            goUp = false;
        }
    }
}

bool KeyboardController::shouldClose() const {
    return _shouldClose;
}

void KeyboardController::step(float delta) {
    if (goLeft) {
        theta -= 1.0f * delta;
    }
    else if (goRight) {
        theta += 1.0f * delta;
    }
    else if (goStraight) {
        radius -= 5.0f * delta;
        radius = radius > 0 ? radius : 0;
    }
    else if (goBack) {
        radius += 5.0f * delta;
    }
    else if (goUp) {
        height += 5.0f * delta;
    }
    else if (goDown) {
        height -= 5.0f * delta;
    }
}

glm::vec3 KeyboardController::getEyePos() const {
    float xPos = radius * cos(theta);
    float zPos = radius * sin(theta);
    return glm::vec3(xPos, height, zPos);
}
