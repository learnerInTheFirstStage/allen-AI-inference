#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define TILE_SIZE 32

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int K, int N)
{
    // Get shared memory (for each Block)
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++ t)
    {
        // 1. Move the data from VRAM to SHARED MEMORY
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            s_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        
        if (col < N && (t * TILE_SIZE + threadIdx.y) < K)
            s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // 2. Make calculation in SHARED MEMORY
        for (int k = 0; k < TILE_SIZE; ++ k)
        {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();

        if (row < M && col < N) C[row * N + col] = sum;
    }
}

/**
 * @brief Simple Matrix Multiplication Kernel
 * Each thread computes one element of the output matrix C.
 */
__global__ void matmul_simple_kernel(const float* A, const float* B, float* C, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// C wrapper function to be called from C++ code
extern "C" void launch_matmul_kernel(const float* h_A, const float* h_B, float* h_C, int M, int K, int N)
{
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // 1. Allocate Device Memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 2. Copy data from Host to Device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 3. Define Grid and Block dimensions
    // dim3 threadsPerBlock(16, 16);
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    // 4. Launch Kernel
    matmul_tiled_kernel <<< numBlocks, threadsPerBlock >>>  (d_A, d_B, d_C, M, K, N);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "[CUDA Internal] Pure Kernel Execution: " << kernel_time / 1000.0 << " ms" << std::endl;
    
    // 5. Copy result back to Host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // 6. Free Device Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}