#pragma once

#include <concepts>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>
#include <algorithm>

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