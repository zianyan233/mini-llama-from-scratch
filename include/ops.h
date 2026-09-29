#pragma once
#include "tensor.h"

void matmul(const Tensor&, const Tensor&, Tensor&);
void transpose(const Tensor&, Tensor&);
void softmax(Tensor&);
void attention(const Tensor&, const Tensor&, const Tensor&, Tensor&);
void rmsnorm(const Tensor&, const Tensor&, Tensor&, float eps = 1e-5f);
void silu(Tensor&);
void swiglu(const Tensor&, const Tensor&, const Tensor&, const Tensor&, Tensor&);
void rope(Tensor&, size_t, float theta_base = 10000.0f);
void add(Tensor&, const Tensor&);