#pragma once

#include "tensor.h"
#include <cstddef>

/**
 * @file ops.h
 * @brief Transformer 底层 9 大核心数学算子声明
 */

// 1. 标准矩阵乘法：C = A * B，要求 A.cols == B.rows
void matmul(const Tensor& a, const Tensor& b, Tensor& c);

// 2. 矩阵转置：k_T(j, i) = k(i, j)，将行与列对调
void transpose(const Tensor& k, Tensor& k_T);

// 3. 行归一化指数函数 (Softmax)：数值稳定版（先减行最大值防溢出，再指数求和归一）
void softmax(Tensor& x);

// 4. 注意力核心算子 (Scaled Dot-Product Attention)：
// 计算 Attention(Q, K, V) = Softmax(Q * K^T / sqrt(d)) * V
void attention(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& output);

// 5. 均方根层归一化 (RMSNorm)：LLaMA 核心加速算子，相比 LayerNorm 去掉了均值平移中心化
void rmsnorm(const Tensor& x, const Tensor& w, Tensor& output, float eps = 1e-5f);

// 6. SiLU (Swish-1) 激活函数：f(x) = x * sigmoid(x) = x / (1 + exp(-x))
void silu(Tensor& x);

// 7. SwiGLU 门控前馈网络算子 (FFN)：
// LLaMA 核心 FFN 架构：Output = (SiLU(x * W_gate) * (x * W_up)) * W_down
void swiglu(const Tensor& x, const Tensor& w_gate, const Tensor& w_up, const Tensor& w_down, Tensor& output);

// 8. 旋转位置编码 (RoPE - Rotary Position Embedding)：
// 将二维复平面旋转矩阵应用在特征维度相邻的两两元素对上，注入绝对与相对位置信息
void rope(Tensor& x, size_t pos, float theta_base = 10000.0f);

// 9. 原地残差连接 (Residual Addition)：a = a + b
void add(Tensor& a, const Tensor& b);