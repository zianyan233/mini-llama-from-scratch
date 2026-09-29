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

struct TransformerBlockWeights {
    // 1. Attention 分支的权重
    Tensor attn_norm_w; // 归一化缩放系数 [1, d]
    Tensor w_q;         // Query 投影矩阵 [d, d]
    Tensor w_k;         // Key 投影矩阵 [d, d]
    Tensor w_v;         // Value 投影矩阵 [d, d]
    Tensor w_o;         // Output 融合矩阵 [d, d]

    // 2. FFN (SwiGLU) 分支的权重
    Tensor ffn_norm_w;  // 归一化缩放系数 [1, d]
    Tensor w_gate;      // 门控升维矩阵 [d, d_ffn]
    Tensor w_up;        // 内容升维矩阵 [d, d_ffn]
    Tensor w_down;      // 降维压缩矩阵 [d_ffn, d]
};

void matmul(const Tensor&, const Tensor&, Tensor&);
void transpose(const Tensor&, Tensor&);
void softmax(Tensor&);
void attention(const Tensor&, const Tensor&, const Tensor&, Tensor&);
void rmsnorm(const Tensor&, const Tensor&, Tensor&, float eps = 1e-5f);
void silu(Tensor&);
void swiglu(const Tensor&, const Tensor&, const Tensor&, const Tensor&, Tensor&);
void rope(Tensor&, size_t, float theta_base = 10000.0f);
void add(Tensor&, const Tensor&);
void transformer_block(Tensor&, size_t, const TransformerBlockWeights&, size_t n_heads = 2);

int main() {
    size_t d = 4;
    size_t d_ffn = 6;

    // 1. 输入数据 x: 1 行 4 列
    Tensor x(1, d);
    x.data = { 1.0f, 0.0f, 1.0f, 0.0f };

    // 2. 构造装满 9 个权重的结构体 (C++20 风格)
    TransformerBlockWeights w{
        .attn_norm_w = Tensor(1, d),
        .w_q = Tensor(d, d),
        .w_k = Tensor(d, d),
        .w_v = Tensor(d, d),
        .w_o = Tensor(d, d),
        .ffn_norm_w = Tensor(1, d),
        .w_gate = Tensor(d, d_ffn),
        .w_up = Tensor(d, d_ffn),
        .w_down = Tensor(d_ffn, d)
    };

    // 归一化权重初始化为 1.0f (标准比例)
    std::ranges::fill(w.attn_norm_w.data, 1.0f);
    std::ranges::fill(w.ffn_norm_w.data, 1.0f);

    // Q, K, V, O 设为单位矩阵 (主对角线为 1)
    for (size_t i = 0; i < d; i++) {
        w.w_q.data[i * d + i] = 1.0f;
        w.w_k.data[i * d + i] = 1.0f;
        w.w_v.data[i * d + i] = 1.0f;
        w.w_o.data[i * d + i] = 1.0f;
    }

    // FFN 权重填充一组测试小数值
    std::ranges::fill(w.w_gate.data, 0.1f);
    std::ranges::fill(w.w_up.data, 0.1f);
    std::ranges::fill(w.w_down.data, 0.1f);

    // 🔥 点火！执行多头 Transformer Block (pos = 0, n_heads = 2)
    transformer_block(x, 0, w, 2);

    // 打印最终输出
    std::cout << "--- Transformer Block Output ---" << std::endl;
    for (float val : x.data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

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

void rmsnorm(const Tensor& x, const Tensor& w, Tensor& output, float eps) {
    assert(x.cols == w.cols);
    assert(x.cols == output.cols);
    assert(x.rows == output.rows);

    for (size_t i = 0; i < x.rows; i++) {
        float sum = 0;
        for (size_t j = 0; j < x.cols; j++) {
            sum += x.data[i * x.cols + j] * x.data[i * x.cols + j];
        }
        float rms = std::sqrt(sum / static_cast<float> (x.cols) + eps);
        for (size_t j = 0; j < x.cols; j++) {
            output.data[i * x.cols + j] = x.data[i * x.cols + j] / rms * w.data[j];
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
            float v0 = x.data[i * x.cols + j];
            float v1 = x.data[i * x.cols + j + 1];
            x.data[i * x.cols + j] = v0 * std::cos(theta) - v1 * std::sin(theta);
            x.data[i * x.cols + j + 1] = v0 * std::sin(theta) + v1 * std::cos(theta);
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

void transformer_block(Tensor& x, size_t pos, const TransformerBlockWeights& w, size_t n_heads) {
    assert(x.cols % n_heads == 0);
    size_t head_dim = x.cols / n_heads;
    assert(head_dim % 2 == 0);
    
    Tensor x_norm(x.rows, x.cols);
    rmsnorm(x, w.attn_norm_w, x_norm);
    Tensor q(x.rows, w.w_q.cols), k(x.rows, w.w_k.cols), v(x.rows, w.w_v.cols);
    matmul(x_norm, w.w_q, q), matmul(x_norm, w.w_k, k), matmul(x_norm, w.w_v, v);

    Tensor attn_v(x.rows, x.cols);
    for (size_t h = 0; h < n_heads; h++) {
        Tensor q_h(x.rows, head_dim);
        Tensor k_h(x.rows, head_dim);
        Tensor v_h(x.rows, head_dim);
        Tensor out_h(x.rows, head_dim);
        for (size_t i = 0; i < x.rows; i++) {
            for (size_t j = 0; j < head_dim; j++) {
                q_h.data[i * head_dim + j] = q.data[i * q.cols + h * head_dim + j];
                k_h.data[i * head_dim + j] = k.data[i * k.cols + h * head_dim + j];
                v_h.data[i * head_dim + j] = v.data[i * v.cols + h * head_dim + j];
            }
        }
        rope(q_h, pos), rope(k_h, pos);
        attention(q_h, k_h, v_h, out_h);
        for (size_t i = 0; i < x.rows; i++) {
            for (size_t j = 0; j < head_dim; j++) {
                attn_v.data[i * attn_v.cols + h * head_dim + j] = out_h.data[i * head_dim + j];
            }
        }
    }
    Tensor attn_out(x.rows, w.w_o.cols);
    matmul(attn_v, w.w_o, attn_out);
    add(x, attn_out);

    rmsnorm(x, w.ffn_norm_w, x_norm);
    Tensor ffn_out(x.rows, w.w_down.cols);
    swiglu(x_norm, w.w_gate, w.w_up, w.w_down, ffn_out);
    add(x, ffn_out);
}