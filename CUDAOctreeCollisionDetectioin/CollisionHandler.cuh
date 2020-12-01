#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <glm/glm.hpp>

#include <vector>

#include "Octree.h"
#include "Constant.cuh"



void handleBallBallCollisionsCuda(std::vector<BallPair> bps);


__global__
void _handleBallBallCollisions(
    glm::vec3* b_pos_1, glm::vec3* b_pos_2,
    glm::vec3* b_v_1, glm::vec3* b_v_2,
    glm::vec3* b_v_out_1, glm::vec3* b_v_out_2,
    int n
);

/*
__global__
void _handleBallWallCollisions() {};
*/


bool _checkDevice();
bool _mallocForBallBallCollisions(
    glm::vec3** p1, glm::vec3** p2,
    glm::vec3** p3, glm::vec3** p4,
    glm::vec3** p5, glm::vec3** p6,
    int n
);
void _initForBallBallCollisions(
    glm::vec3* b_pos_1, glm::vec3* b_pos_2,
    glm::vec3* b_v_1, glm::vec3* b_v_2,
    std::vector<BallPair> bps, int n
);
void _updateVelocity(
    std::vector<BallPair> bps, int n,
    glm::vec3* b_v_out_1, glm::vec3* b_v_out_2
);

