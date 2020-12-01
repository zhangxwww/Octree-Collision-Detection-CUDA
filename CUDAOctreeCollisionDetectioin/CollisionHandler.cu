#include "CollisionHandler.cuh"

#include <stdio.h>


void handleBallBallCollisionsCuda(std::vector<BallPair> bps) {
    int n = bps.size();
    glm::vec3* bp1, * bp2, * bv1, * bv2, * bvo1, * bvo2;
    
    if (!_checkDevice()) {
        goto ErrorDevice;
    }
    if (!_mallocForBallBallCollisions(&bp1, &bp2, &bv1, &bv2, &bvo1, &bvo2, n)) {
        goto ErrorMalloc;
    }
    _initForBallBallCollisions(bp1, bp2, bv1, bv2, bps, n);
    
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    
    _handleBallBallCollisions << <gridSize, blockSize >> > (
        bp1, bp2, bv1, bv2, bvo1, bvo2, n
    );

    cudaDeviceSynchronize();

    _updateVelocity(bps, n, bvo1, bvo2);

    return;

ErrorDevice:
    fprintf(stderr, "cudaSetDevice failed!");
    goto Error;

ErrorMalloc:
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;

Error:
    cudaFree(bv1);
    cudaFree(bv2);
    cudaFree(bp1);
    cudaFree(bp2);
    cudaFree(bvo1);
    cudaFree(bvo2);

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
        if (glm::dot(displacement, displacement) < RADIUS_SQUARE_4) {
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
    glm::vec3** p5, glm::vec3** p6, int n) {

    int size = n * sizeof(glm::vec3);
    cudaError_t cudaStatus;
    cudaStatus = cudaMallocManaged(&p1, size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    cudaStatus = cudaMallocManaged(&p2, size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    cudaStatus = cudaMallocManaged(&p3, size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    cudaStatus = cudaMallocManaged(&p4, size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    cudaStatus = cudaMallocManaged(&p5, size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    cudaStatus = cudaMallocManaged(&p6, size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }
    return true;
}

void _initForBallBallCollisions(
    glm::vec3* b_pos_1, glm::vec3* b_pos_2, 
    glm::vec3* b_v_1, glm::vec3* b_v_2, 
    std::vector<BallPair> bps, int n) {

    for (int i = 0; i < n; i++) {
        Ball* b1 = bps[i].b1;
        Ball* b2 = bps[i].b2;
        b_pos_1[i] = b1->pos;
        b_pos_2[i] = b2->pos;
        b_v_1[i] = b1->v;
        b_v_2[i] = b2->v;
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

