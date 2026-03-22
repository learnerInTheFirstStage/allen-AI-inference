#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

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
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    // 4. Launch Kernel
    matmul_simple_kernel <<< numBlocks, threadsPerBlock >>>  (d_A, d_B, d_C, M, K, N);

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