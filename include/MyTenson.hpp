#pragma once

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define ALLEN_HAS_NEON
#endif

#include <algorithm>
#include <concepts>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

/**
 * @brief A lightweight Tensor class for AI inference.
 * Using C++20 concepts to restrict T to numeric types.
 */
template <typename T>
concept Numeric = std::floating_point<T> || std::integral<T>;

template <typename T = float>
    requires Numeric<T>
class MyTensor {
  public:
    struct Shape {
        size_t rows;
        size_t cols;
    };

    // Constructor using member initailizer list
    MyTensor(size_t r, size_t c) : shape_{r, c}, data_(r * c, T{}) {
    }

    // Use std::span for safe, non-owning view of data
    std::span<const T> data() const {
        return data_;
    }
    Shape shape() const {
        return shape_;
    }

    void fill(T value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    // Static factory method for Matrix Multiplication: C = A * B
    // Optimization: i-k-j loop order for better cache locality
    static MyTensor matmul(const MyTensor &A, const MyTensor &B) {
        if (A.shape_.cols != B.shape_.rows) {
            throw std::invalid_argument("Incompatibal dimensions for matmul");
        }

        const size_t M = A.shape_.rows;
        const size_t K = A.shape_.cols;
        const size_t N = B.shape_.cols;

        MyTensor result(M, N);

        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                T temp = A.data_[i * K + k]; // Access A once for the inner loop
                for (size_t j = 0; j < N; ++j) {
                    result.data_[i * N + j] += temp * B.data_[k * N + j];
                }
            }
        }

        return result;
    }

    // No optimization
    static MyTensor matmul_no_op(const MyTensor &A, const MyTensor &B) {
        if (A.shape_.cols != B.shape_.rows) {
            throw std::invalid_argument("Incompatibal dimensions for matmul");
        }

        const size_t M = A.shape_.rows;
        const size_t K = A.shape_.cols;
        const size_t N = B.shape_.cols;

        MyTensor result(M, N);

        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                for (size_t k = 0; k < K; ++k) {
                    result.data_[i * N + j] +=
                        A.data_[i * K + k] * B.data_[k * N + j];
                }
            }
        }

        return result;
    }

    static MyTensor matmul_neon(const MyTensor &A, const MyTensor &B) {
    #ifdef ALLEN_HAS_NEON
        if (A.shape_.cols != B.shape_.rows) {
            throw std::invalid_argument("Incompatibal dimensions for matmul");
        }

        const size_t M = A.shape_.rows;
        const size_t K = A.shape_.cols;
        const size_t N = B.shape_.cols;

        MyTensor result(M, N);

        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                float temp = A.data_[i * K + k];
                // Broadcast 'temp' into all 4 slots of a NEON register
                float32x4_t a_vec = vdupq_n_f32(temp);

                size_t j = 0;
                // Process 4 elements at a time
                for (; j + 3 < N; j += 4) {
                    // Load 4 floats from B
                    float32x4_t b_vec = vld1q_f32(&B.data_[k * N + j]);
                    // Load 4 current floats from result
                    float32x4_t res_vec = vld1q_f32(&result.data_[i * N + j]);
                    // multiply-accumulate: res = res + (a * b)
                    res_vec = vmlaq_f32(res_vec, a_vec, b_vec);

                    // Store back 4 floats to result
                    vst1q_f32(&result.data_[i * N + j], res_vec);
                }

                // Clean up the remainder (if N is not multiple of 4)
                for (; j < N; ++j) {
                    result.data_[i * N + j] += temp * B.data_[k * N + j];
                }
            }
        }

        return result;
    #else
        // Fallback for Windows/Intel/AMD
        return matmul(A, B);
    #endif
    }

    static MyTensor matmul_fused_neon(const MyTensor &A, const MyTensor &B,
                                      float bias) {
    #ifdef ALLEN_HAS_NEON
        if (A.shape_.cols != B.shape_.rows) {
            throw std::invalid_argument("Incompatibal dimensions for matmul");
        }

        const size_t M = A.shape_.rows;
        const size_t K = A.shape_.cols;
        const size_t N = B.shape_.cols;

        MyTensor result(M, N);

        float32x4_t v_bias = vdupq_n_f32(bias);
        float32x4_t v_zero = vdupq_n_f32(0.0f);

        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                float32x4_t v_a = vdupq_n_f32(A.data_[i * K + k]);
                size_t j = 0;
                for (; j + 3 < N; j += 4) {
                    float32x4_t v_b = vld1q_f32(&B.data_[k * N + j]);
                    float32x4_t v_res = vld1q_f32(&result.data_[i * N + j]);

                    // Fused Multiply-Add
                    v_res = vmlaq_f32(v_res, v_a, v_b);

                    vst1q_f32(&result.data_[i * N + j], v_res);
                }

                // Remainder...
                for (; j < N; ++j) {
                    result.data_[i * N + j] +=
                        A.data_[i * K + k] * B.data_[k * N + j];
                }
            }

            // --- FUSION POINT ---
            // After finishing one row of MatMul, apply Bias and ReLU in one
            // pass
            size_t j = 0;
            for (; j + 3 < N; j += 4) {
                float32x4_t v_res = vld1q_f32(&result.data_[i * N + j]);
                v_res = vaddq_f32(v_res, v_bias); // Add Bias
                v_res = vmaxq_f32(v_res, v_zero); // ReLU: max(0, x)
                vst1q_f32(&result.data_[i * N + j], v_res);
            }

            for (; j < N; ++j) {
                result.data_[i * N + j] =
                    std::max(0.0f, result.data_[i * N + j] + bias);
            }
        }

        return result;
    #else
        // Windows / WSL2 (x86)
        const size_t M = A.shape_.rows;
        const size_t K = A.shape_.cols;
        const size_t N = B.shape_.cols;
        MyTensor result(M, N);
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                float temp = A.data_[i * K + k];
                for (size_t j = 0; j < N; ++j) {
                    result.data_[i * N + j] += temp * B.data_[k * N + j];
                }
            }
            // Fusion: Bias + ReLU
            for (size_t j = 0; j < N; ++j) {
                result.data_[i * N + j] = std::max(0.0f, result.data_[i * N + j] + bias);
            }
        }
        return result;
    #endif
    }

    void print() const {
        for (size_t i = 0; i < shape_.rows; ++i) {
            for (size_t j = 0; j < shape_.cols; ++j) {
                std::cout << data_[i * shape_.cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }

  private:
    Shape shape_;
    std::vector<T> data_;
};