#pragma once

#include <vector>
#include <set>

#include <glm/glm.hpp>

#include "Constant.cuh"


struct Ball {
    glm::vec3 v;
    glm::vec3 pos;
    float radius;
    glm::vec3 color;
};

enum class WALL_TYPE {
    LEFT,
    RIGHT,
    FAR,
    NEAR,
    TOP,
    BOTTOM
};

struct BallPair {
    Ball* b1;
    Ball* b2;
};

struct BallWall {
    Ball* b;
    WALL_TYPE w;
};

class Octree {
public:
    Octree(glm::vec3 min, glm::vec3 max, int depth_);
    ~Octree();

    void add(Ball* ball);
    void remove(Ball* ball);
    void move(Ball* ball, glm::vec3 old);
    void potentialBallWallCollisions(std::vector<BallWall>& collisions);
    void potentialBallBallCollisions(std::vector<BallPair>& collisions);

private:

    void createChildren();
    void destroyChildren();
    void addOrRemove(Ball* ball, glm::vec3 pos, bool addOrRemove);
    void collectBalls(std::set<Ball*>& bset);
    void remove(Ball* ball, glm::vec3 pos);
    void potentialBallWallCollisions(std::vector<BallWall>& collisions, WALL_TYPE w, char coord, int dir);

    glm::vec3 lowerLeft;
    glm::vec3 upperRight;
    glm::vec3 center;

    Octree* children[2][2][2];
    bool has_children;
    std::set<Ball*> balls;
    int depth;
    int numBalls;
};

