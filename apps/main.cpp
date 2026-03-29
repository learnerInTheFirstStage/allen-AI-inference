#include <cuda_runtime.h>
#include "MyTenson.hpp"
#include <chrono>
#include <iostream>

extern "C" void launch_matmul_kernel(const float* h_A, const float* h_B, float* h_C, int M, int K, int N);
extern "C" void launch_fused_matmul_kernel(const float* h_A, const float* h_B, float* h_C, int M, int K, int N, float bias);
extern "C" void launch_fused_matmul_kernel_pooled(const float* h_A, const float* h_B, float* h_C, int M, int K, int N, float bias);
extern "C" void launch_pipelined_matmul(
    float* h_A, float* h_B, float* h_C,
    float* d_A, float* d_B, float* d_C,
    int M, int K, int N, float bias);

/**
 * @brief Simple RAII Timer to measure execution time.
 */
class Timer {
  public:
    Timer(const std::string &name)
        : name_(name), start_(std::chrono::high_resolution_clock::now()) {
    }

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start_)
                .count();
        std::cout << "[Timer] " << name_ << ": " << duration << " ms"
                  << std::endl;
    }

  private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};


int main() {
    try {
        const size_t N = 4096;
        std::cout << "Initializing two " << N << "x" << N << " matrices..."
                  << std::endl;

        MyTensor<float> A(N, N);
        A.fill(1.1f);
        MyTensor<float> B(N, N);
        B.fill(2.2f);

        MyTensor<float> C_gpu(N, N);

        const size_t size = N * N * sizeof(float);
        float *d_A, *d_B, *d_C;
        
        float *h_A_pinned, *h_B_pinned, *h_C_pinned;
        cudaMallocHost(&h_A_pinned, size);
        cudaMallocHost(&h_B_pinned, size);
        cudaMallocHost(&h_C_pinned, size);

        std::copy(A.data().begin(), A.data().end(), h_A_pinned);
        std::copy(B.data().begin(), B.data().end(), h_B_pinned);

        // ------- 1. Memory pooling initialization (Outside Timer) -----------
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        {
            // ----------------------  2. Inference Stage ------------------------
            Timer t("CUDA 4070 Ti Pooled Inference (4096x4096)");

            cudaMemcpy(d_A, A.data().data(), size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, B.data().data(), size, cudaMemcpyHostToDevice);
            
            float bias_val = -0.5f;
            launch_fused_matmul_kernel_pooled(d_A, d_B, d_C, N, N, N, bias_val);

            cudaMemcpy(const_cast<float*>(C_gpu.data().data()), d_C, size, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }

        {
            Timer t("CUDA 4070 Ti Pipelined (Async)");
            launch_pipelined_matmul(h_A_pinned, h_B_pinned, h_C_pinned, d_A, d_B, d_C, N, N, N, -0.5f);
        }

        // ----------------3. Free (Outsize Timer) -------------------------------
        cudaFreeHost(h_A_pinned);
        cudaFreeHost(h_B_pinned);
        cudaFreeHost(h_C_pinned);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // std::cout << "Testing on M1 (Release Mode)..." << std::endl;
        // {
        //     Timer t("Matrix Multiplication with optimization (1024x1024)");
        //     auto C = MyTensor<float>::matmul(A, B);
        // }

        // {
        //     Timer t("Matrix Multiplication (1024x1024)");
        //     auto C = MyTensor<float>::matmul_no_op(A, B);
        // }

        // {
        //     Timer t("Manual NEON Kernel");
        //     auto C = MyTensor<float>::matmul_neon(A, B);
        // }

        // float bias{-0.5f}; // Some bias to test ReLU

        // std::cout << "--- AI Infra Benchmark (MacBook M1) ---" << std::endl;
        // {
        //     Timer t("Fused MatMul + Bias + ReLU (Manual NEON)");
        //     auto C = MyTensor<float>::matmul_fused_neon(A, B, bias);
        // };

        // {
        //     Timer t("CUDA 4070 Ti Kernel (Naive)");           
        //     launch_matmul_kernel(A.data().data(), B.data().data(), const_cast<float*>(C_gpu.data().data()), N, N, N);
        // }

        // {
        //     Timer t("CUDA 4070 Ti kernel (Fused)");
        //     float bias_val = -0.5f;
        //     launch_fused_matmul_kernel(A.data().data(), B.data().data(), const_cast<float*>(C_gpu.data().data()), N, N, N, bias_val);
        // }

        std::cout << "Done." << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
