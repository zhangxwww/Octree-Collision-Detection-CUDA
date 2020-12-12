#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <glm/glm.hpp>

#include <vector>

#include "Octree.h"
#include "Constant.cuh"


void updateBallsInfo(std::vector<Ball*>& balls, int n);
void handleBallBallCollisionsCuda(std::vector<BallIndexPair>& bips, std::vector<Ball*>& balls);
void handleBallWallCollisionsCuda(std::vector<BallWallIndexPair>& bwips, std::vector<Ball*>& balls);
void updateVelocity( std::vector<Ball*>& balls, int n);

__global__
void _handleBallBallCollisions(int n);

__global__
void _handleBallWallCollisions(int n);

__device__
glm::vec3 getWallDir(int w);

void _initBallInfo(std::vector<Ball*>& balls, int n);
void _updateBallBallCollisionInfo(std::vector<BallIndexPair>& bips, int m);
void _updateBallWallCollisionInfo(std::vector<BallWallIndexPair>& bwips, int m);

