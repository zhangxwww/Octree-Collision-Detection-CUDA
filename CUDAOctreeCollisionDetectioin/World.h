#pragma once

#include <glm/glm.hpp>

#include <vector>

#include "Octree.h"
#include "Constant.cuh"
#include "CollisionHandler.cuh"


class World {
public:
    World();
    ~World();

    void addBalls(const int n);
    void step(float t, float& timeUntilUpdate);

    std::vector<Ball*>& getBalls();

private:
    void move(float dt);
    void applyGravity();

    bool testBallBallCollision(BallPair bp);
    bool testBallWallCollision(BallWall bw);

    void handleBallBallCollisions();
    void handleBallWallCollisions();

    glm::vec3 wallDirection(WALL_TYPE w) const;

    void performUpdate();


    Octree* root;
    std::vector<Ball*> balls;
};

