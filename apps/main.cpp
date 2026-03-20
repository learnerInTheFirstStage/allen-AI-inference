#include "MyTenson.hpp"
#include <chrono>
#include <iostream>

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
        const size_t N = 1024;
        std::cout << "Initializing two " << N << "x" << N << " matrices..."
                  << std::endl;

        MyTensor<float> A(N, N);
        A.fill(1.1f);
        MyTensor<float> B(N, N);
        B.fill(2.2f);

        {
            Timer t("Matrix Multiplication with optimization (1024x1024)");
            auto C = MyTensor<float>::matmul(A, B);
        }

        {
            Timer t("Matrix Multiplication (1024x1024)");
            auto C = MyTensor<float>::matmul_no_op(A, B);
        }

        std::cout << "Done." << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
