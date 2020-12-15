#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// window size
const int WIDTH = 800;
const int HEIGHT = 600;

// descrete of the ball
const int X_SEGMENTS = 50;
const int Y_SEGMENTS = 50;

const float PI = 3.14159265358979323846f;

const float GRAVITY = 10.0f;

// size of the box
__constant__
const float BOX_SIZE = 10.0f;

// used to update the world
const float TIME_BETWEEN_UPDATES = 0.01f;
const int TIMER_MS = 25;

// configuration of the octree
const int MAX_OCTREE_DEPTH = 6;
const int MIN_BALLS_PER_OCTREE = 3;
const int MAX_BALLS_PER_OCTREE = 16;

// size of the scene
const float SCENE_MAX_X = BOX_SIZE / 2;
const float SCENE_MAX_Y = BOX_SIZE / 2;
const float SCENE_MAX_Z = BOX_SIZE / 2;

const int SCENE_MIN_X = -SCENE_MAX_X;
const int SCENE_MIN_Y = -SCENE_MAX_Y;
const int SCENE_MIN_Z = -SCENE_MAX_Z;

// radius of the ball
const float MAX_RADIUS = 0.3;
const float MIN_RADIUS = 0.1;

// mass of the ball
const float MAX_MASS = 5;
const float MIN_MASS = 1;

// COR of the ball
const float MAX_E = 1.0f;
const float MIN_E = 0.5f;

// max initial velocity of the ball
const float MAX_VELOCITY = 3;

// vertices, indices, model matrix, color of the wall
struct Wall {
    const float vertex[12] = {
        0.5f, 0.0f, 0.5f,
        0.5f, 0.0f, -0.5f,
        -0.5f, 0.0f, -0.5f,
        -0.5f, 0.0f, 0.5f
    };
    const unsigned int indices[6] = {
        0, 1, 3,
        1, 2, 3
    };
    const glm::mat4 floorTrans = glm::scale(
        glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, SCENE_MIN_Y, 0.0f)),
        glm::vec3(BOX_SIZE)
    );
    const glm::mat4 ceilTrans = glm::scale(
        glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, SCENE_MAX_Y, 0.0f)),
        glm::vec3(BOX_SIZE)
    );
    const glm::vec3 color = glm::vec3(0.6f, 0.6f, 0.6f);
};

const Wall wall;

// constrains of the world
const int MAX_BALLS = 10000;
const int MAX_COLLISIONS = 10000000;
