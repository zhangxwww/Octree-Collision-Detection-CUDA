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

    // cpu version, not used
    bool testBallBallCollision(BallIndexPair bip);
    bool testBallWallCollision(BallWallIndexPair bwip);

    void handleBallBallCollisions();
    void handleBallWallCollisions();

    // cpu version, not used
    void handleBallBallCollisionsCpu(std::vector<BallIndexPair>& bips);
    void handleBallWallCollisionsCpu(std::vector<BallWallIndexPair>& bwips);

    glm::vec3 wallDirection(WALL_TYPE w) const;

    void performUpdate();


    Octree* root;
    std::vector<Ball*> balls;
};

