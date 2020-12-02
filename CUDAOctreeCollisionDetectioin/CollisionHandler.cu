#include "CollisionHandler.cuh"

#include <stdio.h>


void handleBallBallCollisionsCuda(std::vector<BallPair> bps) {

    int n = bps.size();

    glm::vec3* bp1, * bp2, * bv1, * bv2, * bvo1, * bvo2;
    float* m1, * m2, * r1, * r2, * e1, * e2;
    
    if (!_checkDevice()) {
        goto ErrorDevice;
    }
    if (!_mallocForBallBallCollisions(&bp1, &bp2, &bv1, &bv2, &bvo1, &bvo2, &m1, &m2, &r1, &r2, &e1, &e2, n)) {
        goto ErrorMalloc;
    }
    _initForBallBallCollisions(bp1, bp2, bv1, bv2, m1, m2, r1, r2, e1, e2, bps, n);
    
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    
    _handleBallBallCollisions << <gridSize, blockSize >> > (
        bp1, bp2, bv1, bv2, bvo1, bvo2, n
    );

    cudaDeviceSynchronize();

    _updateVelocity(bps, n, bvo1, bvo2);

    goto Return;

ErrorDevice:
    fprintf(stderr, "cudaSetDevice failed!");
    goto Return;

ErrorMalloc:
    fprintf(stderr, "cudaMalloc failed!");
    goto Return;

Return:
    cudaFree(bv1);
    cudaFree(bv2);
    cudaFree(bp1);
    cudaFree(bp2);
    cudaFree(bvo1);
    cudaFree(bvo2);
    cudaFree(m1);
    cudaFree(m2);
    cudaFree(r1);
    cudaFree(r2);
    cudaFree(e1);
    cudaFree(e2);

    return;
}

__global__
void _handleBallBallCollisions(
    glm::vec3* b_pos_1, glm::vec3* b_pos_2, 
    glm::vec3* b_v_1, glm::vec3* b_v_2, 
    glm::vec3* b_v_out_1, glm::vec3* b_v_out_2, 
    int n
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        glm::vec3 p1 = b_pos_1[i];
        glm::vec3 p2 = b_pos_2[i];
        glm::vec3 v1 = b_v_1[i];
        glm::vec3 v2 = b_v_2[i];
        glm::vec3 displacement = p1 - p2;
        if (glm::dot(displacement, displacement) < 0.16) {
            glm::vec3 dv = v1 - v2;
            if (glm::dot(dv, displacement) < 0) {
                glm::vec3 dis = glm::normalize(displacement);
                b_v_out_1[i] = v1 - glm::vec3(2.0f) * dis * glm::dot(v1, dis);
                b_v_out_2[i] = v2 - glm::vec3(2.0f) * dis * glm::dot(v2, dis);
            }
        }
    }
}

bool _checkDevice() {
    return cudaSetDevice(0) == cudaSuccess ? true : false;
}

bool _mallocForBallBallCollisions(
    glm::vec3** p1, glm::vec3** p2, 
    glm::vec3** p3, glm::vec3** p4, 
    glm::vec3** p5, glm::vec3** p6,
    float** m1, float** m2,
    float** r1, float** r2,
    float** e1, float** e2,
    int n) {

    int v_size = n * sizeof(glm::vec3);
    int f_size = n * sizeof(float);
    cudaError_t cudaStatus;
    cudaStatus = cudaMallocManaged(p1, v_size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    cudaStatus = cudaMallocManaged(p2, v_size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    cudaStatus = cudaMallocManaged(p3, v_size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    cudaStatus = cudaMallocManaged(p4, v_size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    cudaStatus = cudaMallocManaged(p5, v_size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    cudaStatus = cudaMallocManaged(p6, v_size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    cudaStatus = cudaMallocManaged(m1, f_size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    cudaStatus = cudaMallocManaged(m2, f_size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    cudaStatus = cudaMallocManaged(r1, f_size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    cudaStatus = cudaMallocManaged(r2, f_size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    cudaStatus = cudaMallocManaged(e1, f_size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    cudaStatus = cudaMallocManaged(e2, f_size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    return true;
}

void _initForBallBallCollisions(
    glm::vec3* b_pos_1, glm::vec3* b_pos_2, 
    glm::vec3* b_v_1, glm::vec3* b_v_2, 
    float* m1, float* m2,
    float* r1, float* r2,
    float* e1, float* e2,
    std::vector<BallPair> bps, int n) {

    for (int i = 0; i < n; i++) {
        Ball* b1 = bps[i].b1;
        Ball* b2 = bps[i].b2;
        b_pos_1[i] = b1->pos;
        b_pos_2[i] = b2->pos;
        b_v_1[i] = b1->v;
        b_v_2[i] = b2->v;
        m1[i] = b1->m;
        m2[i] = b2->m;
        r1[i] = b1->radius;
        r2[i] = b2->radius;
        e1[i] = b1->e;
        e2[i] = b2->e;
    }
}

void _updateVelocity(
    std::vector<BallPair> bps, int n,
    glm::vec3* b_v_out_1, glm::vec3* b_v_out_2) {

    for (int i = 0; i < n; i++) {
        bps[i].b1->v = b_v_out_1[i];
        bps[i].b2->v = b_v_out_2[i];
    }
}

