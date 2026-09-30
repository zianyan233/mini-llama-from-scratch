#pragma once

#include "tensor.h"
#include <cstddef>

/**
 * @brief 单层 Transformer Block 所需的全部 9 组静态权重集合
 */
struct TransformerBlockWeights {
    // 1. Attention (多头自注意力) 分支的权重
    Tensor attn_norm_w; // Attention 输入归一化缩放系数 [1, d]
    Tensor w_q;         // Query 提问投影矩阵 [d, d]
    Tensor w_k;         // Key 键特征投影矩阵 [d, d]
    Tensor w_v;         // Value 内容投影矩阵 [d, d]
    Tensor w_o;         // Output 多头汇聚融合矩阵 [d, d]

    // 2. FFN (SwiGLU 门控前馈网络) 分支的权重
    Tensor ffn_norm_w;  // FFN 输入归一化缩放系数 [1, d]
    Tensor w_gate;      // 门控升维矩阵 [d, d_ffn]
    Tensor w_up;        // 内容升维矩阵 [d, d_ffn]
    Tensor w_down;      // 降维压缩矩阵 [d_ffn, d]
};

/**
 * @brief 键值缓存区 (KV-Cache)
 * 
 * 物理意义：
 * - 避免在自回归（逐字解码）生成阶段对历史 Token 重复进行矩阵投影运算
 * - 预先在堆内存中开辟最大上下文窗口大小 [max_seq_len, dim] 的固定缓冲区
 * - 每一轮生成只需计算当前 1 个新词的 K 和 V，追加写入对应行即可
 */
struct KVCache {
    Tensor key_cache;   // 历史 Key 缓存矩阵，形状为 [max_seq_len, dim]
    Tensor value_cache; // 历史 Value 缓存矩阵，形状为 [max_seq_len, dim]

    // 构造函数：初始化指定容量的连续内存缓冲区
    KVCache(size_t max_seq_len, size_t dim)
        : key_cache(max_seq_len, dim), value_cache(max_seq_len, dim) {}
};

/**
 * @brief 执行单层完整 Transformer 块计算
 * 
 * @param x         输入/输出张量 [seq_len, dim]，原地计算并更新
 * @param pos       当前输入批次在整个对话序列中的起始全局绝对时间步 (0-indexed)
 * @param w         当前层的只读权重集合
 * @param cache     当前层的可变键值缓存对象 (KVCache&)
 * @param n_heads   多头注意力的头数 (默认为 2)
 */
void transformer_block(Tensor& x, size_t pos, const TransformerBlockWeights& w, KVCache& cache, size_t n_heads = 2);