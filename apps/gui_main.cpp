#include "MyTenson.hpp"
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <chrono>

class MyWindow : public QWidget {
  public:
    MyWindow() {
        auto *layout = new QVBoxLayout(this);
        auto *btn = new QPushButton("Run Fused Kernel Benchmark", this);
        label_ = new QLabel("Result will appear here...", this);

        layout->addWidget(btn);
        layout->addWidget(label_);

        connect(btn, &QPushButton::clicked, this, &MyWindow::runBenchmark);

        setWindowTitle("AllenInference Dashboard");
        resize(400, 200);
    }

  private:
    QLabel *label_;

    void runBenchmark() {
        const size_t N = 1024;
        MyTensor<float> A(N, N);
        A.fill(1.1f);
        MyTensor<float> B(N, N);
        B.fill(2.2f);

        auto start = std::chrono::high_resolution_clock::now();

        auto C = MyTensor<float>::matmul_fused_neon(A, B, -0.5f);

        auto end = std::chrono::high_resolution_clock::now();
        auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();

        label_->setText(QString("Inference Success! Time: %1 ms").arg(ms));
    }
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    MyWindow window;
    window.show();
    return app.exec();
}