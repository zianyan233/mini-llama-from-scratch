# 指定 CMake 的最低版本要求（保证兼容性）
cmake_minimum_required(VERSION 3.20)

# 定义我们的工程名字，这里叫 mini_llama (CXX 代表 C++)
project(mini_llama CXX)

# 设置 C++ 标准为 C++20（现代 C++ 的坚实基石，Ranges 特性完美支持）
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 4. 告诉编译器：这个可执行文件由哪些 .cpp 组成！
# （编译时，CMake 会分别把它们编译成 main.obj, ops.obj, model.obj，最后粘在一起）
add_executable(mini_llama
    main.cpp
    src/model.cpp
    src/ops.cpp
)

# 5. 告诉编译器：头文件去哪找！
# （遇到 #include "tensor.h" 时，顺便去 include/ 目录里搜寻）
target_include_directories(mini_llama PRIVATE include)