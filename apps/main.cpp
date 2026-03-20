#include <iostream>
#include "MyTenson.hpp"

int main() {
    try {
        MyTensor<float> A(4, 4);
        A.fill(1.5f);

        MyTensor<float> B(4, 4);
        B.fill(2.0f);

        auto C = MyTensor<float>::matmul(A, B);

        std::cout << "MyInference Engine Test:\n";
        std::cout << "Result Matrix (4x4):\n";
        C.print();

    } catch (const std::exception &e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
