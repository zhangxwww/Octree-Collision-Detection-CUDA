#include "Octree.h"

Octree::Octree(glm::vec3 min, glm::vec3 max, int depth_):
    lowerLeft(min), upperRight(max), 
    center((min + max) / glm::vec3(2.0f, 2.0f, 2.0f)),
    numBalls(0), depth(depth_), has_children(false) {
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                children[i][j][k] = nullptr;
            }
        }
    }
}

Octree::~Octree() {
    if (has_children) {
        destroyChildren();
    }
}

void Octree::add(Ball* ball) {
    numBalls++;
    if (!has_children && depth < MAX_OCTREE_DEPTH && numBalls > MAX_BALLS_PER_OCTREE) {
        createChildren();
    }
    if (has_children) {
        addOrRemove(ball, ball->pos, true);
    }
    else {
        balls.insert(ball);
    }
}

void Octree::remove(Ball* ball) {
    remove(ball, ball->pos);
}

void Octree::move(Ball* ball, glm::vec3 old) {
    remove(ball, old);
    add(ball);
}

void Octree::potentialBallWallCollisions(std::vector<BallWall>& collisions) {
    potentialBallWallCollisions(collisions, WALL_TYPE::LEFT, 'x', 0);
    potentialBallWallCollisions(collisions, WALL_TYPE::RIGHT, 'x', 1);
    potentialBallWallCollisions(collisions, WALL_TYPE::BOTTOM, 'y', 0);
    potentialBallWallCollisions(collisions, WALL_TYPE::TOP, 'y', 1);
    potentialBallWallCollisions(collisions, WALL_TYPE::FAR, 'z', 0);
    potentialBallWallCollisions(collisions, WALL_TYPE::NEAR, 'z', 1);
}

void Octree::potentialBallBallCollisions(std::vector<BallPair>& collisions) {
    if (has_children) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    children[i][j][k]->potentialBallBallCollisions(collisions);
                }
            }
        }
    }
    else {
        for (auto it1 = balls.begin(); it1 != balls.end(); it1++) {
            for (auto it2 = balls.begin(); it2 != balls.end(); it2++) {
                if (*it1 < *it2) {
                    BallPair bp;
                    bp.b1 = *it1;
                    bp.b2 = *it2;
                    collisions.push_back(bp);
                }
            }
        }
    }
}

void Octree::createChildren() {
    for (int i = 0; i < 2; i++) {
        float minX = i == 0 ? lowerLeft.x : center.x;
        float maxX = i == 0 ? center.x : upperRight.x;

        for (int j = 0; j < 2; j++) {
            float minY = j == 0 ? lowerLeft.y : center.y;
            float maxY = j == 0 ? center.y : upperRight.y;

            for (int k = 0; k < 2; k++) {
                float minZ = k == 0 ? lowerLeft.z : center.z;
                float maxZ = k == 0 ? center.z : upperRight.z;

                children[i][j][k] = new Octree(
                    glm::vec3(minX, minY, minZ),
                    glm::vec3(maxX, maxY, maxZ),
                    depth + 1);
            }
        }
    }
    for (Ball* b : balls) {
        addOrRemove(b, b->pos, true);
    }
    balls.clear();
    has_children = true;
}

void Octree::destroyChildren() {
    collectBalls(balls);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                delete children[i][j][k];
            }
        }
    }
    has_children = false;
}

void Octree::addOrRemove(Ball* ball, glm::vec3 pos, bool isAdd) {
    for (int i = 0; i < 2; i++) {
        if (i == 0) {
            if (pos.x - ball->radius > center.x) {
                continue;
            }
        }
        else if (pos.x + ball->radius < center.x) {
            continue;
        }

        for (int j = 0; j < 2; j++) {
            if (j == 0) {
                if (pos.y - ball->radius > center.y) {
                    continue;
                }
            }
            else if (pos.y + ball->radius < center.y) {
                continue;
            }

            for (int k = 0; k < 2; k++) {
                if (k == 0) {
                    if (pos.z - ball->radius > center.z) {
                        continue;
                    }
                }
                else if (pos.z + ball->radius < center.z) {
                    continue;
                }
                if (isAdd) {
                    children[i][j][k]->add(ball);
                }
                else {
                    children[i][j][k]->remove(ball, pos);
                }
            }
        }
    }
}

void Octree::collectBalls(std::set<Ball*>& bset) {
    if (has_children) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    children[i][j][k]->collectBalls(bset);
                }
            }
        }
    }
    else {
        for (Ball* b : balls) {
            bset.insert(b);
        }
    }
}

void Octree::remove(Ball* ball, glm::vec3 pos) {
    numBalls--;
    if (has_children && numBalls < MIN_BALLS_PER_OCTREE) {
        destroyChildren();
    }
    if (has_children) {
        addOrRemove(ball, pos, false);
    }
    else {
        balls.erase(ball);
    }
}

void Octree::potentialBallWallCollisions(std::vector<BallWall>& collisions, WALL_TYPE w, char coord, int dir) {
    if (has_children) {
        for (int d2 = 0; d2 < 2; d2++) {
            for (int d3 = 0; d3 < 2; d3++) {
                Octree* child = nullptr;
                switch (coord) {
                case 'x':
                    child = children[dir][d2][d3];
                    break;
                case 'y':
                    child = children[d2][dir][d3];
                    break;
                case 'z':
                    child = children[d2][d3][dir];
                    break;
                default:
                    break;
                }
                child->potentialBallWallCollisions(collisions, w, coord, dir);
            }
        }
    }
    else {
        for (Ball* b : balls) {
            BallWall bw;
            bw.w = w;
            bw.b = b;
            collisions.push_back(bw);
        }
    }
}
