#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 9

__global__ void MatAdd(float* MatA, float* MatB, float* MatC) {

}


__global__ void MatElementWiseMul(float* MatA, float* MatB, float* MatC){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = i * N + j; // Flattened 2D indexing

    if (i < N && j < N) {
        MatC[idx] = MatA[idx] * MatB[idx];
    }
}

int main() {
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks((N + threadsPerBlock.x -1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y -1) / threadsPerBlock.y);

    float* MatA = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    float* MatB = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    float* MatC = malloc(sizeof(MatA));
    size_t size = N * N * sizeof(float);


    
    cudaMemcpy((void**)&MatA, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)&MatB, N * sizeof(float), cudaMemcpyHostToDevice);
    //launch kernela
    cudaMemcpy((void**)&MatC, N * sizeof(float), cudaMemcpyDeviceToHost);
    

    return 0;
}
