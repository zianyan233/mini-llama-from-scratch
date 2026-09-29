#include "tensor.h"
#include "ops.h"
#include "model.h"
#include <iostream>
#include <algorithm>

int main() {
    size_t d = 4;
    size_t d_ffn = 6;

    // 1. 输入数据 x: 1 行 4 列 (使用 C++20 统一初始化列表构造函数)
    Tensor x(1, d, { 1.0f, 0.0f, 1.0f, 0.0f });

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

    // Q, K, V, O 设为单位矩阵 (使用优雅的 operator() 索引)
    for (size_t i = 0; i < d; i++) {
        w.w_q(i, i) = 1.0f;
        w.w_k(i, i) = 1.0f;
        w.w_v(i, i) = 1.0f;
        w.w_o(i, i) = 1.0f;
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

