#include "ops.h"
#include <cmath>
#include <algorithm>
#include <cassert>

/**
 * @brief 经典矩阵乘法 C = A * B
 * 
 * 维度规则：[M, K] * [K, N] -> [M, N]
 * 外层循环遍历 A 的行，中层遍历 B 的列，内层累加点积
 */
void matmul(const Tensor& a, const Tensor& b, Tensor& c) {
    assert(a.cols == b.rows);
    assert(c.cols == b.cols);
    assert(c.rows == a.rows);

    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < b.cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a.cols; k++) {
                sum += a(i, k) * b(k, j);
            }
            c(i, j) = sum;
        }
    }
}

/**
 * @brief 矩阵转置：将 [rows, cols] 转置为 [cols, rows]
 */
void transpose(const Tensor& k, Tensor& k_T) {
    assert(k.rows == k_T.cols);
    assert(k.cols == k_T.rows);

    for (size_t i = 0; i < k.rows; i++) {
        for (size_t j = 0; j < k.cols; j++) {
            k_T(j, i) = k(i, j);
        }
    }
}

/**
 * @brief 数值稳定的按行 Softmax 归一化
 * 
 * 物理细节：
 * - 针对张量的每一行分别归一化（使得每一行概率之和为 1.0）
 * - 先找出该行最大值 max_val，每个元素减去 max_val 再计算 exp(f - max_val)，防止浮点数指数爆炸上溢出
 */
void softmax(Tensor& x) {
    for (size_t i = 0; i < x.rows; i++) {
        // 1. 寻找当前行的最大值
        float max_val = x(i, 0);
        for (size_t j = 1; j < x.cols; j++) {
            if (x(i, j) > max_val) max_val = x(i, j);
        }

        // 2. 指数映射并求该行的总和
        float sum = 0.0f;
        for (size_t j = 0; j < x.cols; j++) {
            x(i, j) = std::exp(x(i, j) - max_val);
            sum += x(i, j);
        }

        // 3. 归一化为百分比概率
        for (size_t j = 0; j < x.cols; j++) {
            x(i, j) /= sum;
        }
    }
}

/**
 * @brief 缩放点积注意力 (Scaled Dot-Product Attention)
 * 
 * 流程：
 * 1. 打分矩阵 S = Q * K^T，尺寸为 [q.rows, k.rows]
 * 2. 缩放：S = S / sqrt(head_dim)
 * 3. 概率化：Softmax(S)
 * 4. 加权提取内容：Output = S * V，尺寸为 [q.rows, v.cols]
 */
void attention(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& output) {
    Tensor k_T(k.cols, k.rows);
    Tensor scores(q.rows, k.rows); // 完美支持 1 行或多行提问
    transpose(k, k_T);
    matmul(q, k_T, scores);

    // 缩放因子：除以 sqrt(head_dim)，防止点积数值过大导致 Softmax 梯度饱和
    float scale = 1.0f / std::sqrt(static_cast<float>(q.cols));
    for (float& score : scores.data) {
        score *= scale;
    }

    softmax(scores);
    matmul(scores, v, output);
}

/**
 * @brief 均方根层归一化 (RMSNorm)
 * 
 * 公式：y = (x / RMS(x)) * w，其中 RMS(x) = sqrt( mean(x^2) + eps )
 * 相比标准 LayerNorm，省略了减去均值均值化的步骤，计算更轻量，大模型首选
 */
void rmsnorm(const Tensor& x, const Tensor& w, Tensor& output, float eps) {
    assert(x.cols == w.cols);
    assert(x.cols == output.cols);
    assert(x.rows == output.rows);

    for (size_t i = 0; i < x.rows; i++) {
        // 计算当前行所有元素的平方和
        float sum = 0.0f;
        for (size_t j = 0; j < x.cols; j++) {
            sum += x(i, j) * x(i, j);
        }
        // 计算均方根 RMS
        float rms = std::sqrt(sum / static_cast<float>(x.cols) + eps);
        // 归一化并乘上可学习的缩放权重 w
        for (size_t j = 0; j < x.cols; j++) {
            output(i, j) = (x(i, j) / rms) * w.data[j];
        }
    }
}

/**
 * @brief SiLU 激活函数 (Sigmoid Linear Unit / Swish-1)
 * 
 * 公式：f(x) = x / (1 + exp(-x))
 */
void silu(Tensor& x) {
    for (float& f : x.data) {
        f = f / (1.0f + std::exp(-f));
    }
}

/**
 * @brief SwiGLU 门控前馈网络 (Feed-Forward Network)
 * 
 * 物理架构：
 * 1. 升维两条分支：x_gate = x * W_gate,  x_up = x * W_up
 * 2. 门控激活：对 x_gate 应用 SiLU
 * 3. 门控点乘：hidden = SiLU(x_gate) ⊙ x_up (元素逐个相乘)
 * 4. 降维投射：output = hidden * W_down 回到原始模型维度
 */
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

/**
 * @brief 旋转位置编码 (RoPE - Rotary Position Embedding)
 * 
 * 物理原理：
 * - 将特征两两一组当作二维复平面坐标 (v0, v1)
 * - 按照时间步位置 curr_pos = pos + i 和对应频率进行旋转：
 *   v0' = v0 * cos(theta) - v1 * sin(theta)
 *   v1' = v0 * sin(theta) + v1 * cos(theta)
 */
void rope(Tensor& x, size_t pos, float theta_base) {
    assert(x.cols % 2 == 0); // 特征维度必须为偶数（两两配对）

    for (size_t i = 0; i < x.rows; i++) {
        size_t curr_pos = pos + i; // 每一行都有属于自己唯一的时间戳坐标！
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

/**
 * @brief 原地向量相加 (残差连接专用)
 */
void add(Tensor& a, const Tensor& b) {
    assert(a.rows == b.rows);
    assert(a.cols == b.cols);

    for (size_t i = 0; i < a.data.size(); i++) {
        a.data[i] += b.data[i];
    }
}