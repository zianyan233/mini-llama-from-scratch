# 指定 CMake 的最低版本要求（保证兼容性）
cmake_minimum_required(VERSION 3.20)

# 定义我们的工程名字，这里叫 mini_llama
project(mini_llama CXX)

# 设置 C++ 标准为 C++20（现代 C++ 的坚实基石，Ranges 特性完美支持）
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 告诉编译器：用 main.cpp 生成一个可执行程序，名字叫 mini_llama
add_executable(mini_llama main.cpp)
