#include "World.h"

float randFloat() {
    return (float)rand() / ((float)RAND_MAX + 1);
}


World::World() {
    root = new Octree(
        glm::vec3(SCENE_MIN_X, SCENE_MIN_Y, SCENE_MIN_Z),
        glm::vec3(SCENE_MAX_X, SCENE_MAX_Y, SCENE_MAX_Z),
        0
    );
}

World::~World() {
    delete root;
}

void World::addBalls(const int n) {
    for (int i = 0; i < n; i++) {
        Ball* b = new Ball();
        b->pos = glm::vec3(
            SCENE_MAX_X * randFloat(),
            SCENE_MAX_Y * randFloat(),
            SCENE_MAX_Z * randFloat()
        );
        b->v = glm::vec3(
            MAX_VELOCITY * randFloat(),
            MAX_VELOCITY * randFloat(),
            MAX_VELOCITY * randFloat()
        );
        b->radius = RADIUS;
        b->color = glm::vec3(
            0.6f * randFloat() + 0.2f,
            0.6f * randFloat() + 0.2f,
            0.6f * randFloat() + 0.2f
        );
        balls.push_back(b);
        root->add(b);
    }
}

void World::step(float t, float& timeUntilUpdate) {
    while (t > 0) {
        if (timeUntilUpdate <= t) {
            move(timeUntilUpdate);
            performUpdate();
            t -= timeUntilUpdate;
            timeUntilUpdate = TIME_BETWEEN_UPDATES;
        }
        else {
            move(t);
            timeUntilUpdate -= t;
            t = 0;
        }
    }
}

void World::move(float dt) {
    for (Ball* b : balls) {
        glm::vec3 old = b->pos;
        b->pos += b->v * dt;
        root->move(b, old);
    }
}

void World::applyGravity() {
    for (Ball* b : balls) {
        b->v.y -= GRAVITY * TIME_BETWEEN_UPDATES;
    }
}

bool World::testBallBallCollision(BallPair bp)
{
    float r = 2 * RADIUS;
    glm::vec3 displacement = bp.b1->pos - bp.b2->pos;
    if (glm::dot(displacement, displacement) < r * r) {
        glm::vec3 dv = bp.b1->v - bp.b2->v;
        return glm::dot(dv, displacement) < 0; 
    }
    return false;
}

bool World::testBallWallCollision(BallWall bw) {
    glm::vec3 dir = wallDirection(bw.w);
    return glm::dot(bw.b->pos, dir) + bw.b->radius > BOX_SIZE / 2
        && glm::dot(bw.b->v, dir) > 0;
}

void World::handleBallBallCollisions() {
    std::vector<BallPair> bps;
    root->potentialBallBallCollisions(bps);
    for (BallPair bp : bps) {
        if (testBallBallCollision(bp)) {
            Ball* b1 = bp.b1;
            Ball* b2 = bp.b2;
            glm::vec3 displacement = glm::normalize(b1->pos - b2->pos);
            b1->v -= glm::vec3(2.0f) * displacement * glm::dot(b1->v, displacement);
            b2->v -= glm::vec3(2.0f) * displacement * glm::dot(b2->v, displacement);
        }
    }
}

void World::handleBallWallCollisions() {
    std::vector<BallWall> bws;
    root->potentialBallWallCollisions(bws);
    for (BallWall bw : bws) {
        if (testBallWallCollision(bw)) {
            glm::vec3 dir = glm::normalize(wallDirection(bw.w));
            bw.b->v -= glm::vec3(2.0f) * dir * glm::dot(bw.b->v, dir);
        }
    }
}

glm::vec3 World::wallDirection(WALL_TYPE w) const {
    switch (w)
    {
    case WALL_TYPE::LEFT:
        return glm::vec3(-1.0f, 0.0f, 0.0f);
    case WALL_TYPE::RIGHT:
        return glm::vec3(1.0f, 0.0f, 0.0f);
    case WALL_TYPE::FAR:
        return glm::vec3(0.0f, 0.0f, 1.0f);
    case WALL_TYPE::NEAR:
        return glm::vec3(0.0f, 0.0f, -1.0f);
    case WALL_TYPE::TOP:
        return glm::vec3(0.0f, 1.0f, 0.0f);
    case WALL_TYPE::BOTTOM:
        return glm::vec3(0.0f, -1.0f, 0.0f);
    default:
        return glm::vec3(0.0f, 0.0f, 0.0f);
    }
}

void World::performUpdate() {
    applyGravity();
    handleBallBallCollisions();
    handleBallWallCollisions();
}
