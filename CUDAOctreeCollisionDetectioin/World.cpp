#include "World.h"

float randFloat() {
    return (float)rand() / ((float)RAND_MAX + 1);
}

glm::vec3 randVec3() {
    return glm::vec3(randFloat(), randFloat(), randFloat());
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
        b->pos = randVec3() * BOX_SIZE - glm::vec3(SCENE_MAX_X, SCENE_MAX_Y, SCENE_MAX_Z);
        b->v = randVec3() * 2.0f - glm::vec3(MAX_VELOCITY);
        b->radius = randFloat() * (MAX_RADIUS - MIN_RADIUS) + MIN_RADIUS;
        b->color = randVec3() * 0.6f + 0.2f;
        b->m = randFloat() * (MAX_MASS - MIN_MASS) + MIN_MASS;
        b->e = randFloat();
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

std::vector<Ball*>& World::getBalls() {
    return balls;
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
    glm::vec3 displacement = bp.b1->pos - bp.b2->pos;
    float r = bp.b1->radius + bp.b2->radius;
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
   //  handleBallBallCollisionsCpu(bps);
    handleBallBallCollisionsCuda(bps);
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

void World::handleBallBallCollisionsCpu(std::vector<BallPair> bps) {
    for (BallPair bp : bps) {
        if (testBallBallCollision(bp)) {
            Ball* b1 = bp.b1;
            Ball* b2 = bp.b2;
            glm::vec3 displacement = glm::normalize(b1->pos - b2->pos);
            float e = std::min(b1->e, b2->e);
            glm::vec3 vr1 = glm::dot(b1->v, displacement) * displacement;
            glm::vec3 vr2 = glm::dot(b2->v, displacement) * displacement;
            glm::vec3 dvr1 = ((1 + e) * b2->m * (vr2 - vr1)) / (b1->m + b2->m);
            glm::vec3 dvr2 = ((1 + e) * b1->m * (vr1 - vr2)) / (b1->m + b2->m);
            b1->v += dvr1;
            b2->v += dvr2;
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
        return glm::vec3(0.0f, 0.0f, -1.0f);
    case WALL_TYPE::NEAR:
        return glm::vec3(0.0f, 0.0f, 1.0f);
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
