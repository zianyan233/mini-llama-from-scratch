#include "ops.h"
#include <cmath>
#include <algorithm>
#include <cassert>

void matmul(const Tensor& a, const Tensor& b, Tensor& c) {
    assert(a.cols == b.rows);
    assert(c.cols == b.cols);
    assert(c.rows == a.rows);

    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < b.cols; j++) {
            float sum = 0;
            for (size_t k = 0; k < a.cols; k++) {
                sum += a(i, k) * b(k, j);
            }
            c(i, j) = sum;
        }
    }
}

void transpose(const Tensor& k, Tensor& k_T) {
    assert(k.rows == k_T.cols);
    assert(k.cols == k_T.rows);

    for (size_t i = 0; i < k.rows; i++) {
        for (size_t j = 0; j < k.cols; j++) {
            k_T(j, i) = k(i, j);
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

void rmsnorm(const Tensor& x, const Tensor& w, Tensor& output, float eps) {
    assert(x.cols == w.cols);
    assert(x.cols == output.cols);
    assert(x.rows == output.rows);

    for (size_t i = 0; i < x.rows; i++) {
        float sum = 0;
        for (size_t j = 0; j < x.cols; j++) {
            sum += x(i, j) * x(i, j);
        }
        float rms = std::sqrt(sum / static_cast<float> (x.cols) + eps);
        for (size_t j = 0; j < x.cols; j++) {
            output(i, j) = x(i, j) / rms * w.data[j];
        }
    }
}

void silu(Tensor& x) {
    for (float& f : x.data) f = f / (1.0f + std::exp(-f));
}

void swiglu(const Tensor& x, const Tensor& w_gate, const Tensor& w_up, const Tensor& w_down, Tensor& output) {
    Tensor x_up(x.rows, w_up.cols);
    Tensor x_gate(x.rows, w_gate.cols);
    Tensor hidden(x.rows, w_gate.cols);

    matmul(x, w_up, x_up);
    matmul(x, w_gate, x_gate);
    silu(x_gate);
    for (size_t i = 0; i < hidden.data.size(); i++) {
        hidden.data[i] = x_up.data[i] * x_gate.data[i];
    }
    matmul(hidden, w_down, output);
}

void rope(Tensor& x, size_t pos, float theta_base) {
    assert(x.cols % 2 == 0);

    for (size_t i = 0; i < x.rows; i++) {
        size_t curr_pos = pos + i;
        for (size_t j = 0; j < x.cols; j += 2) {
            float freq = 1.0f / std::pow(theta_base, float(j) / float(x.cols));
            float theta = float(curr_pos) * freq;
            float v0 = x(i, j);
            float v1 = x(i, j + 1);
            x(i, j)     = v0 * std::cos(theta) - v1 * std::sin(theta);
            x(i, j + 1) = v0 * std::sin(theta) + v1 * std::cos(theta);
        }
    }
}

void add(Tensor& a, const Tensor& b) {
    assert(a.rows == b.rows);
    assert(a.cols == b.cols);

    for (size_t i = 0; i < a.data.size(); i++) {
        a.data[i] += b.data[i];
    }
}