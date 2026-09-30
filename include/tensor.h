#pragma once

#include <vector>
#include <cassert>
#include <initializer_list>
#include <cstddef>

/**
 * @brief 基础二维张量结构体 (Tensor)
 * 
 * 物理内存设计：
 * - 采用 1D 扁平数组 (std::vector<float>) 存储二维矩阵数据（行优先排布 Row-Major）
 * - 内存连续，对 CPU L1/L2 缓存友好，顺序遍历速度极快
 * - 重载了 operator()，支持通过二维坐标 (row, col) 直接读写，彻底告别繁琐易错的 i * cols + j 下标计算
 */
struct Tensor {
    std::vector<float> data; // 实际存放所有浮点数的连续底层内存池
    size_t rows;             // 矩阵行数（通常对应：时间步长 / Token 序列长度）
    size_t cols;             // 矩阵列数（通常对应：特征维度 / 隐藏层大小 dim）

    // 默认与指定尺寸构造函数：分配 rows * cols 大小内存，并自动全部填零 0.0f
    Tensor(size_t r = 0, size_t c = 0)
        : rows(r), cols(c), data(r * c, 0.0f) {}

    // C++11 列表初始化构造函数：支持用花括号直接灌入初始测试数据，如 Tensor x(1, 4, {1.0f, 0.0f, 1.0f, 0.0f})
    Tensor(size_t r, size_t c, std::initializer_list<float> list)
        : rows(r), cols(c), data(list) {
        assert(data.size() == r * c); // 严格安检：确保传入的数据量与行*列完全匹配
    }

    // 可变版本下标访问：返回引用，支持对特定位置写入，如 x(i, j) = 1.0f
    float& operator() (size_t r, size_t c) {
        return data[r * cols + c];
    }

    // 只读 const 版本下标访问：用于 const 引用传参时的只读读取，如 sum += a(i, k)
    const float& operator() (size_t r, size_t c) const {
        return data[r * cols + c];
    }
};