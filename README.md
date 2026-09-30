# Mini-Llama-From-Scratch 🦙

A lightweight, educational Transformer inference engine built from scratch in **C++20** without third-party deep learning frameworks.

---

## 🌟 Project Highlights

- **Zero Heavy Frameworks**: No PyTorch, no ONNX, no external BLAS. Everything written from raw memory basics.
- **Physical Memory Model**: All tensors are stored in contiguous 1D memory buffers (Row-Major Order) with cache-friendly layout.
- **C++20**: Powered by CMake, `<algorithm>`, C++20 Ranges (`std::ranges::fill`, `std::ranges::max`), and type-safe reference semantics.

---

## 🏗️ Implemented Operators (v0.2-alpha)

- [x] **Contiguous Tensor (`struct Tensor`)**: Dynamic buffer allocation with bounds-checked assertions and shape metadata.
- [x] **General Matrix Multiplication (`matmul` / GEMM)**: 2D-by-2D matrix multiplication with cache-friendly row-major scanning.
- [x] **Matrix Transposition (`transpose`)**: Safe 2D tensor transpose operator.
- [x] **Numerically Stable Softmax (`softmax`)**: Safe in-place Softmax mitigating floating-point overflow via max-subtraction.
- [x] **Scaled Dot-Product Attention (`attention`)**: Full $Q \cdot K^T / \sqrt{d}$ self-attention mechanism matching PyTorch numerical accuracy.
- [x] **RMSNorm (`rmsnorm`)**: Root Mean Square Normalization with learnable per-channel scaling weights.
- [x] **SiLU Activation (`silu`)**: In-place Sigmoid Linear Unit non-linear activation.
- [x] **SwiGLU FFN (`swiglu`)**: LLaMA-style gated feed-forward network with up, gate, and down projection matrices.
- [x] **RoPE (`rope`)**: Rotary Position Embedding decomposing vectors into 2D sub-planes for relative positional encoding.
- [x] **Residual Add (`add`)**: Element-wise in-place tensor addition to mitigate gradient vanishing and preserve features.
- [x] **Multi-Head Transformer Block (`transformer_block`)**: Complete LLaMA layer combining Pre-RMSNorm, Multi-Head Attention with RoPE, Output Fusion, and SwiGLU FFN.

---

## 🗺️ Engineering Roadmap

- [x] **Phase 1: Core LLaMA Operators**
  - [x] GEMM & Transpose
  - [x] Numerically stable Softmax & Scaled Dot-Product Attention
  - [x] RMSNorm (Root Mean Square Normalization)
  - [x] RoPE (Rotary Position Embedding)
  - [x] SwiGLU / SiLU Feed-Forward Network
- [ ] **Phase 2: Architecture & Memory**
  - [x] Multi-Head Transformer Block (Residual Connections + FFN)
  - [ ] KV-Cache implementation for autoregressive decoding
- [ ] **Phase 3: Real Weights Inference (End-to-End Chat)**
  - [ ] Binary model weights loader (`stories15M.bin` / GGUF)
  - [ ] Byte-Pair Encoding (BPE) Tokenizer
  - [ ] Autoregressive text generation CLI
- [ ] **Phase 4: CPU High-Performance Optimization (HPC)**
  - [ ] Cache tiling & block-level memory locality
  - [ ] SIMD / AVX2 vectorization intrinsics
  - [ ] Multi-core parallelization with OpenMP
- [ ] **Phase 5: CUDA C++ GPU Acceleration**
  - [ ] Custom CUDA kernels (`.cu`) for GEMM, RMSNorm, and RoPE
  - [ ] GPU Shared Memory & Warp-level primitives
  - [ ] FlashAttention kernel implementation

---

## 🚀 Build & Run

### Prerequisites
- CMake >= 3.20
- Modern C++ Compiler supporting C++20 (GCC >= 11, Clang >= 13, or MSVC >= 2019)

```powershell
# 1. Configure
cmake -B build -G "MinGW Makefiles"

# 2. Build
cmake --build build

# 3. Run Transformer Block Test
.\build\mini_llama.exe
```
