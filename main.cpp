#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cmath>

struct Tensor {
    std::vector<float> data;
    size_t rows;
    size_t cols;

    Tensor(size_t r, size_t c) {
        rows = r;
        cols = c;
        data.resize(r * c, 0.0f);
    }
};

void matmul(const Tensor&, const Tensor&, Tensor&);
void transpose(const Tensor&, Tensor&);
void softmax(Tensor&);
void attention(const Tensor&, const Tensor&, const Tensor&, Tensor&);

int main() {
    // 1. 搜索词 q: 想找“水果” (1 行 2 列的行向量)
    Tensor q(1, 2);
    q.data = { 1.0f, 0.0f };
    // 2. 钥匙 K (2 个词，每个词 2 维):
    // 词 0：“苹果” [1.0, 0.0]
    // 词 1：“汽车” [0.0, 1.0]
    Tensor k(2, 2);
    k.data = {
        1.0f, 0.0f,  // 苹果
        0.0f, 1.0f   // 汽车
    };
    // 3. 内容 V (2 个词的正文内容):
    // 苹果的内容: [10.0, 20.0]
    // 汽车的内容: [80.0, 90.0]
    Tensor v(2, 2);
    v.data = {
        10.0f, 20.0f,
        80.0f, 90.0f
    };
    // 4. 输出容器 output: (1 行 2 列)
    Tensor output(1, 2);
    // Attention is all you need!
    attention(q, k, v, output);
    // 打印结果：
    std::cout << "--- Final Attention Result ---" << std::endl;
    for (size_t i = 0; i < output.data.size(); ++i) {
        std::cout << "output[" << i << "] = " << output.data[i] << std::endl;
    }
    return 0;
}

void matmul(const Tensor& a, const Tensor& b, Tensor& c) {
    assert(a.cols == b.rows);
    assert(c.cols == b.cols);
    assert(c.rows == a.rows);

    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < b.cols; j++) {
            float sum = 0;
            for (size_t k = 0; k < a.cols; k++) {
                sum += a.data[i * a.cols + k] * b.data[k * b.cols + j];
            }
            c.data[i * c.cols + j] = sum;
        }
    }
}

void transpose(const Tensor& k, Tensor& k_T) {
    assert(k.rows == k_T.cols);
    assert(k.cols == k_T.rows);

    for (size_t i = 0; i < k.rows; i++) {
        for (size_t j = 0; j < k.cols; j++) {
            k_T.data[j * k.rows + i] = k.data[i * k.cols + j];
        }
    }
}

void softmax(Tensor& x) {
    float max_val = std::ranges::max(x.data);
    float sum = 0.0f;
    for (float& f: x.data) {
        f = std::exp(f - max_val);
        sum += f;
    }
    for (float& f: x.data) {
        f /= sum;
    }
}

void attention(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& output) {
    Tensor k_T(k.cols, k.rows);
    Tensor scores(1, k.rows);
    transpose(k, k_T);
    matmul(q, k_T, scores);
    for (float& score: scores.data) score /= std::sqrt(static_cast<float> (q.cols));
    softmax(scores);
    matmul(scores, v, output);
}
