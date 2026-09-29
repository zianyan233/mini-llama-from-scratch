#include "tensor.h"
#include <cstddef>

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

void transformer_block(Tensor&, size_t, const TransformerBlockWeights&, size_t n_heads = 2);