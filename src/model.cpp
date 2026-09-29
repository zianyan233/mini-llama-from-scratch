#include "model.h"
#include "ops.h"
#include <cassert>
#include <cstddef>

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
                q_h(i, j) = q(i, h * head_dim + j);
                k_h(i, j) = k(i, h * head_dim + j);
                v_h(i, j) = v(i, h * head_dim + j);
            }
        }
        rope(q_h, pos), rope(k_h, pos);
        attention(q_h, k_h, v_h, out_h);
        for (size_t i = 0; i < x.rows; i++) {
            for (size_t j = 0; j < head_dim; j++) {
                attn_v(i, h * head_dim + j) = out_h(i, j);
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