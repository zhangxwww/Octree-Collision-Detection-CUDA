#include "CollisionHandler.cuh"

#include <stdio.h>

glm::vec3 bp[MAX_BALLS], bv[MAX_BALLS];
float mass[MAX_BALLS], r[MAX_BALLS], e[MAX_BALLS];
int id1[MAX_COLLISIONS], id2[MAX_COLLISIONS], id3[MAX_COLLISIONS], id4[MAX_COLLISIONS];

__device__ glm::vec3 d_bp[MAX_BALLS], d_bv[MAX_BALLS];
__device__ float d_mass[MAX_BALLS], d_r[MAX_BALLS], d_e[MAX_BALLS];
__device__ int d_id1[MAX_COLLISIONS], d_id2[MAX_COLLISIONS], d_id3[MAX_COLLISIONS], d_id4[MAX_COLLISIONS];


void updateBallsInfo(std::vector<Ball*>& balls, int n) {
    for (int i = 0; i < n; i++) {
        bp[i] = balls[i]->pos;
        bv[i] = balls[i]->v;
    }

    int v_size = n * sizeof(glm::vec3);
    cudaMemcpyToSymbol(d_bp, bp, v_size, 0);
    cudaMemcpyToSymbol(d_bv, bv, v_size, 0);
}

void handleBallBallCollisionsCuda(std::vector<BallIndexPair>& bips, std::vector<Ball*>& balls) {

    int n = balls.size();
    int m = bips.size();

    _updateBallBallCollisionInfo(bips, m);

    dim3 blockSize(64);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    _handleBallBallCollisions << <gridSize, blockSize >> > (m);
}

void handleBallWallCollisionsCuda(std::vector<BallWallIndexPair>& bwips, std::vector<Ball*>& balls) {

    int n = balls.size();
    int m = bwips.size();

    _updateBallWallCollisionInfo(bwips, m);

    dim3 blockSize(64);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    _handleBallWallCollisions << <gridSize, blockSize >> > (m);
}

__global__
void _handleBallBallCollisions(int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        int index1 = d_id1[i];
        int index2 = d_id2[i];
        glm::vec3 p1 = d_bp[index1];
        glm::vec3 p2 = d_bp[index2];
        float r = d_r[index1] + d_r[index2];
        glm::vec3 displacement = p1 - p2;
        if (glm::dot(displacement, displacement) < r * r)  {
            glm::vec3 v1 = d_bv[index1];
            glm::vec3 v2 = d_bv[index2];
            glm::vec3 dv = v1 - v2;
            if (glm::dot(dv, displacement) < 0) {
                float e1 = d_e[index1];
                float e2 = d_e[index2];
                float e = e1 < e2 ? e1 : e2;

                float m1 = d_mass[index1];
                float m2 = d_mass[index2];

                glm::vec3 dis = glm::normalize(displacement);
                glm::vec3 vr1 = glm::dot(v1, dis) * dis;
                glm::vec3 vr2 = glm::dot(v2, dis) * dis;
                glm::vec3 dvr1 = ((1 + e) * m2 * (vr2 - vr1)) / (m1 + m2);
                glm::vec3 dvr2 = ((1 + e) * m1 * (vr1 - vr2)) / (m1 + m2);
                d_bv[index1] += dvr1;
                d_bv[index2] += dvr2;
            }
        }
    }
}


__global__
void _handleBallWallCollisions(int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        int bid = d_id3[i];
        int wid = d_id4[i];
        glm::vec3 dir = getWallDir(wid);
        glm::vec3 p = d_bp[bid];
        glm::vec3 v = d_bv[bid];
        float r = d_r[bid];
        if (glm::dot(p, dir) + r > BOX_SIZE / 2
            && glm::dot(v, dir) > 0) {

            float e = d_e[bid];
            d_bv[bid] -= (1 + e) * dir * glm::dot(v, dir);
        }
    }
}

__device__ 
glm::vec3 getWallDir(int w) {
    switch (w)
    {
    case 0:
        return glm::vec3(-1.0f, 0.0f, 0.0f);
    case 1:
        return glm::vec3(1.0f, 0.0f, 0.0f);
    case 2:
        return glm::vec3(0.0f, 0.0f, -1.0f);
    case 3:
        return glm::vec3(0.0f, 0.0f, 1.0f);
    case 4:
        return glm::vec3(0.0f, 1.0f, 0.0f);
    case 5:
        return glm::vec3(0.0f, -1.0f, 0.0f);
    default:
        return glm::vec3(0.0f);
    }
}

void _initBallInfo(std::vector<Ball*>& balls, int n) {
    for (int i = 0; i < n; i++) {
        bp[i] = balls[i]->pos;
        bv[i] = balls[i]->v;
        mass[i] = balls[i]->m;
        r[i] = balls[i]->radius;
        e[i] = balls[i]->e;
    }

    int v_size = n * sizeof(glm::vec3);
    int f_size = n * sizeof(float);

    cudaMemcpyToSymbol(d_bp, bp, v_size, 0);
    cudaMemcpyToSymbol(d_bv, bv, v_size, 0);
    cudaMemcpyToSymbol(d_mass, mass, f_size, 0);
    cudaMemcpyToSymbol(d_r, r, f_size, 0);
    cudaMemcpyToSymbol(d_e, e, f_size, 0);
}

void _updateBallBallCollisionInfo(std::vector<BallIndexPair>& bips, int m) {
    for (int i = 0; i < m && i < MAX_COLLISIONS; i++) {
        id1[i] = bips[i].id1;
        id2[i] = bips[i].id2;
    }
    int i_size = m * sizeof(int);
    cudaMemcpyToSymbol(d_id1, id1, i_size, 0);
    cudaMemcpyToSymbol(d_id2, id2, i_size, 0);
}

void _updateBallWallCollisionInfo(std::vector<BallWallIndexPair>& bwips, int m) {
    for (int i = 0; i < m && i < MAX_COLLISIONS; i++) {
        id3[i] = bwips[i].bid;
        id4[i] = bwips[i].wid;
    }
    int i_size = m * sizeof(int);
    cudaMemcpyToSymbol(d_id3, id3, i_size, 0);
    cudaMemcpyToSymbol(d_id4, id4, i_size, 0);
}

void updateVelocity(std::vector<Ball*>& balls, int n) {
    cudaMemcpyFromSymbol(bv, d_bv, n * sizeof(glm::vec3));
    for (int i = 0; i < n; i++) {
        balls[i]->v = bv[i];
    }
}

