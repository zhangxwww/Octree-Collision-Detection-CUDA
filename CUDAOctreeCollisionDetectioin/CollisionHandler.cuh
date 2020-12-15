#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <glm/glm.hpp>

#include <vector>

#include "Octree.h"
#include "Constant.cuh"

// update the position and velocity for each ball in device
void updateBallsInfo(std::vector<Ball*>& balls, const int n);

void handleBallBallCollisionsCuda(std::vector<BallIndexPair>& bips, std::vector<Ball*>& balls);
void handleBallWallCollisionsCuda(std::vector<BallWallIndexPair>& bwips, std::vector<Ball*>& balls);

// update velocity in host
void updateVelocity( std::vector<Ball*>& balls, const int n);

// n: # pairs of balls
__global__
void _handleBallBallCollisions(int n);

// n: # of ball-wall pairs
__global__
void _handleBallWallCollisions(int n);

// get the normal vector of each wall
__device__
glm::vec3 getWallDir(const int w);

// initialize the mass, radius, COR of each ball in device
void _initBallInfo(std::vector<Ball*>& balls, const int n);

// update the index of the entries in device
void _updateBallBallCollisionInfo(std::vector<BallIndexPair>& bips, const int m);
void _updateBallWallCollisionInfo(std::vector<BallWallIndexPair>& bwips, const int m);

