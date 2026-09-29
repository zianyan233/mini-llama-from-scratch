#pragma once
#include <vector>
#include <cassert>
#include <initializer_list>
#include <cstddef>

struct Tensor {
    std::vector<float> data;
    size_t rows;
    size_t cols;

    Tensor(size_t r = 0, size_t c = 0)
        : rows(r), cols(c), data(r * c, 0.0f) {}
    Tensor(size_t r, size_t c, std::initializer_list<float> list)
        : rows(r), cols(c), data(list) {
        assert(data.size() == r * c);
    }
    float& operator() (size_t r, size_t c) {
        return data[r * cols + c];
    }
    const float& operator() (size_t r, size_t c) const {
        return data[r * cols + c];
    }
};