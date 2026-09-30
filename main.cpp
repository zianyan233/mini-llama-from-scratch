#include "tensor.h"
#include "ops.h"
#include "model.h"
#include <iostream>
#include <algorithm>

/**
 * @file main.cpp
 * @brief mini-llama 验证与测试主入口
 * 
 * 验证目标：
 * 1. 验证多头 Transformer Block 在 C++20 工程架构下的数值正确性
 * 2. 验证 KV-Cache 在自回归连续多步 (pos = 0, pos = 1) 过程中的记忆累积能力
 */
int main() {
    size_t d = 4;         // 隐藏层维度 dim = 4
    size_t d_ffn = 6;     // FFN 升维维度 d_ffn = 6
    size_t max_seq = 256; // 最大上下文窗口容量 256

    // 1. 初始化属于本层的持久记忆库：KV-Cache (预分配连续内存)
    KVCache cache(max_seq, d);

    // 2. 构造装满 9 个权重的结构体 (使用 C++20 指定初始化器语法)
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

    // 归一化权重初始化为 1.0f (标准比例缩放)
    std::ranges::fill(w.attn_norm_w.data, 1.0f);
    std::ranges::fill(w.ffn_norm_w.data, 1.0f);

    // Q, K, V, O 设为单位矩阵 (使用优雅的 operator() 二维索引访问)
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

    // =============================================================
    // 🔥 第一步：输入第 0 个词 (pos = 0)
    // =============================================================
    Tensor x0(1, d, { 1.0f, 0.0f, 1.0f, 0.0f });
    transformer_block(x0, 0, w, cache, 2);

    std::cout << "--- [Step 0] Token 0 Output ---" << std::endl;
    for (float val : x0.data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // =============================================================
    // 🔥 第二步：输入第 1 个词 (pos = 1)，见证它调用第 0 步的 KV-Cache！
    // =============================================================
    Tensor x1(1, d, { 0.5f, 1.0f, 0.5f, 1.0f });
    transformer_block(x1, 1, w, cache, 2);

    std::cout << "--- [Step 1] Token 1 Output (With KV-Cache) ---" << std::endl;
    for (float val : x1.data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
