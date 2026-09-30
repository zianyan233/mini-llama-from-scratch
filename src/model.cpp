#include "model.h"
#include "ops.h"
#include <cassert>
#include <cstddef>

/**
 * @brief 执行单层完整 Transformer 块计算（支持 KV-Cache 与多头机制）
 * 
 * 物理数据流向全景：
 * 
 *     输入 x [seq_len, dim]
 *          │
 *          ├─────────────────────────────────────────┐ (残差直通支路 1)
 *          ▼                                         │
 *       RMSNorm                                      │
 *          │                                         │
 *          ├─> 算 Q [seq_len, dim]                   │
 *          ├─> 算 K [seq_len, dim]                   │
 *          └─> 算 V [seq_len, dim]                   │
 *                 │                                  │
 *          [多头循环 for h = 0..n_heads]              │
 *                 ├─> 切片得 q_h, k_h, v_h           │
 *                 ├─> 旋转 RoPE (注入绝对位置编码)      │
 *                 ├─> 写入 KV-Cache 对应时间槽位       │
 *                 ├─> 从 Cache 提取历史全量 k_all, v_all│
 *                 ├─> Attention 计算得出 out_h        │
 *                 └─> 拼装回 attn_v                  │
 *                        │                           │
 *                        ▼                           │
 *                  Wo 投影 [seq_len, dim]             │
 *                        │                           │
 *                        ▼                           │
 *                Add (残差相加 1) ◄───────────────────┘
 *                        │
 *                        ├───────────────────────────┐ (残差直通支路 2)
 *                        ▼                           │
 *                     RMSNorm                        │
 *                        │                           │
 *                        ▼                           │
 *                  SwiGLU (FFN 分支)                  │
 *                        │                           │
 *                        ▼                           │
 *                Add (残差相加 2) ◄───────────────────┘
 *                        │
 *                        ▼
 *                      最终输出
 */
void transformer_block(Tensor& x, size_t pos, const TransformerBlockWeights& w, KVCache& cache, size_t n_heads) {
    // -------------------------------------------------------------
    // 0. 哨兵断言安检：确保物理维度完全合法，绝不带病运行
    // -------------------------------------------------------------
    assert(x.cols == cache.key_cache.cols);      // 输入特征维度必须与 Cache 宽度完全一致
    assert(x.cols % n_heads == 0);                // 隐藏层维度必须能被多头数量整除
    assert(pos + x.rows <= cache.key_cache.rows); // 绝对防止上下文超限，不可越界写入
    size_t head_dim = x.cols / n_heads;           // 计算单头维度
    assert(head_dim % 2 == 0);                    // RoPE 旋转要求单头维度必须为偶数（两两配对）
    
    // -------------------------------------------------------------
    // 1. Attention 分支前置归一化 (RMSNorm)
    // -------------------------------------------------------------
    Tensor x_norm(x.rows, x.cols);
    rmsnorm(x, w.attn_norm_w, x_norm);

    // -------------------------------------------------------------
    // 2. 线性投影：计算当前输入词的 Q, K, V (行数为 x.rows)
    // -------------------------------------------------------------
    Tensor q(x.rows, w.w_q.cols), k(x.rows, w.w_k.cols), v(x.rows, w.w_v.cols);
    matmul(x_norm, w.w_q, q);
    matmul(x_norm, w.w_k, k);
    matmul(x_norm, w.w_v, v);

    // -------------------------------------------------------------
    // 3. 多头自注意力计算 (Multi-Head Self-Attention with KV-Cache)
    // -------------------------------------------------------------
    Tensor attn_v(x.rows, x.cols); // 准备接收所有头拼装回去的总注意力输出
    
    for (size_t h = 0; h < n_heads; h++) {
        // (1) 切片：切出当前头 h 对应的 q_h, k_h, v_h (尺寸为 [x.rows, head_dim])
        Tensor q_h(x.rows, head_dim);
        Tensor k_h(x.rows, head_dim);
        Tensor v_h(x.rows, head_dim);
        Tensor out_h(x.rows, head_dim);

        for (size_t i = 0; i < x.rows; i++) {
            for (size_t j = 0; j < head_dim; j++) {
                q_h(i, j) = q(i, h * head_dim + j);
                k_h(i, j) = k(i, h * head_dim + j);
                v_h(i, j) = v(i, h * head_dim + j);
            }
        }

        // (2) 旋转位置编码：给新算出的 q_h 和 k_h 打上属于当前时间点 pos 的旋转烙印
        rope(q_h, pos);
        rope(k_h, pos);

        // (3) 存入 Cache：将旋转好的 k_h 和内容 v_h 写入本头对应的 KV-Cache 全局物理槽位
        for (size_t i = 0; i < x.rows; i++) {
            for (size_t j = 0; j < head_dim; j++) {
                cache.key_cache(pos + i, h * head_dim + j)   = k_h(i, j);
                cache.value_cache(pos + i, h * head_dim + j) = v_h(i, j);
            }
        }

        // (4) 历史全集切片：从 KV-Cache 中捞出从第 0 行到第 (pos + x.rows - 1) 行的全部历史
        size_t total_len = pos + x.rows;
        Tensor k_h_all(total_len, head_dim);
        Tensor v_h_all(total_len, head_dim);
        for (size_t i = 0; i < total_len; i++) {
            for (size_t j = 0; j < head_dim; j++) {
                k_h_all(i, j) = cache.key_cache(i, h * head_dim + j);
                v_h_all(i, j) = cache.value_cache(i, h * head_dim + j);
            }
        }

        // (5) 注意力打分：让当前的提问 q_h 去和整个历史 k_h_all、v_h_all 进行加权计算
        attention(q_h, k_h_all, v_h_all, out_h);

        // (6) 拼回大矩阵：把当前头算出的结果按列拼回 attn_v
        for (size_t i = 0; i < x.rows; i++) {
            for (size_t j = 0; j < head_dim; j++) {
                attn_v(i, h * head_dim + j) = out_h(i, j);
            }
        }
    }

    // -------------------------------------------------------------
    // 4. Output 融合投影与第一道残差连接
    // -------------------------------------------------------------
    Tensor attn_out(x.rows, w.w_o.cols);
    matmul(attn_v, w.w_o, attn_out);
    add(x, attn_out); // 原地相加残差：x = x + attn_out

    // -------------------------------------------------------------
    // 5. FFN (SwiGLU 门控前馈网络) 分支与第二道残差连接
    // -------------------------------------------------------------
    rmsnorm(x, w.ffn_norm_w, x_norm);
    Tensor ffn_out(x.rows, w.w_down.cols);
    swiglu(x_norm, w.w_gate, w.w_up, w.w_down, ffn_out);
    add(x, ffn_out);  // 原地相加残差：x = x + ffn_out
}