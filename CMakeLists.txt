cmake_minimum_required(VERSION 3.20)
project(miniserving LANGUAGES CXX)

#设置C++标准是C++17，且强制要求
set(CMAKE_CXX_STANDARD 17)

#查找pybind11（scikit_build_core 会把它的路径注入进来）
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#定义一个Python扩展模块 _miniserving
#pybind11_add_module会自动处理 .so/.pyd命名和Python链接
pybind11_add_module(_miniserving src/bindings.cpp)

#让模块能自己找到头文件
target_include_directories(_miniserving PRIVATE src)

#告诉CMake把编译好的 .so 安装到 miniserving/ 目录下
#没有这一行，scikit-build-core 不会把 .so 打进 wheel
install(TARGETS _miniserving LIBRARY DESTINATION miniserving)

