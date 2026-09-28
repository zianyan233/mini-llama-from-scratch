# Mini-Llama-From-Scratch 🦙

A lightweight, educational Transformer inference engine built from scratch in **Modern C++20** without third-party deep learning frameworks.

---

## 🌟 Project Highlights

- **Zero Heavy Frameworks**: No PyTorch, no ONNX, no external BLAS. Everything written from raw memory basics.
- **Physical Memory Model**: All tensors are stored in contiguous 1D memory buffers (Row-Major Order) with zero-cost reshape/transposition abstractions.
- **Modern C++20**: Powered by CMake, `<algorithm>`, C++20 Ranges (`std::ranges::max`), and type-safe reference semantics.

---

## 🏗️ Implemented Operators (v0.1-alpha)

- [x] **Contiguous Tensor (`struct Tensor`)**: Dynamic buffer allocation with bounds-checked assertions and shape metadata.
- [x] **General Matrix Multiplication (`matmul` / GEMM)**: General 2D-by-2D matrix multiplication with cache-friendly row-major scanning.
- [x] **Matrix Transposition (`transpose`)**: Safe 2D tensor transpose operator.
- [x] **Numerically Stable Softmax (`softmax`)**: Safe in-place Softmax mitigating floating-point overflow via max-subtraction.
- [x] **Scaled Dot-Product Attention (`attention`)**: Full $Q \cdot K^T / \sqrt{d}$ self-attention mechanism matching PyTorch numerical accuracy.

---

## 🗺️ Roadmap

- [ ] **Phase 1: Core LLaMA Operators**
  - [ ] RMSNorm (Root Mean Square Normalization)
  - [ ] RoPE (Rotary Position Embedding)
  - [ ] SwiGLU / SiLU Activation Function
- [ ] **Phase 2: Architecture & Memory**
  - [ ] Transformer Block (Residual Connections + FFN)
  - [ ] KV-Cache implementation for autoregressive decoding
- [ ] **Phase 3: Real Weights Inference**
  - [ ] Binary model weights loader (`stories15M.bin` / GGUF)
  - [ ] Tokenizer (BPE)
  - [ ] Autoregressive text generation CLI
- [ ] **Phase 4: Hardware Acceleration**
  - [ ] SIMD / AVX2 CPU Vectorization
  - [ ] CUDA C++ Kernels (`.cu`) for GPU inference acceleration

---

## 🚀 Build & Run

### Prerequisites
- CMake >= 3.20
- Modern C++ Compiler supporting C++23 (GCC >= 13, Clang >= 16, or MSVC)

```powershell
# 1. Configure
cmake -B build -G "MinGW Makefiles"

# 2. Build
cmake --build build

# 3. Run Self-Attention Test
.\build\mini_llama.exe
```
