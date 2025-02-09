#pragma once

#include <vector>
#include <set>

#include <glm/glm.hpp>

#include "Constant.cuh"


struct Ball {
    glm::vec3 v;
    glm::vec3 pos;
    float radius;
    float m;
    float e;
    glm::vec3 color;
    int id;
};


enum class WALL_TYPE {
    LEFT = 0,
    RIGHT = 1,
    FAR = 2,
    NEAR = 3,
    TOP = 4,
    BOTTOM = 5
};

struct BallIndexPair {
    int id1;
    int id2;
};

struct BallWallIndexPair {
    int bid;
    int wid;
};

class Octree {
public:
    Octree(const glm::vec3 & min, const glm::vec3 & max, const int depth_);
    ~Octree();

    // add the ball into the tree
    void add(Ball* ball);
    // remove the ball from the tree
    void remove(Ball* ball);
    // move the ball from the old position to the new one
    void move(Ball* ball, glm::vec3 old);

    // find potential collisions and push them into the vector
    void potentialBallWallCollisions(std::vector<BallWallIndexPair>& collisions);
    void potentialBallBallCollisions(std::vector<BallIndexPair>& collisions);

private:

    void createChildren();
    void destroyChildren();
    void addOrRemove(Ball* ball, glm::vec3 pos, bool addOrRemove);
    void collectBalls(std::set<Ball*>& bset);
    void remove(Ball* ball, glm::vec3 pos);
    void potentialBallWallCollisions(std::vector<BallWallIndexPair>& collisions, 
        const WALL_TYPE w, const char coord, const int dir);

    glm::vec3 lowerLeft;
    glm::vec3 upperRight;
    glm::vec3 center;

    Octree* children[2][2][2];
    bool has_children;
    std::set<Ball*> balls;
    int depth;
    int numBalls;
};

