这是一个MiniServing inference项目：试图建立一个AI infra框架，目标是vLLM项目，但是重点在于其feature和技术细节，而不是整体规模。
# 系统要求
- 语言要求： python + c++
- Build: 
    - CMake (>=3.20) 构建底层算子
    - pybind11 用于python绑定c++
    - scikit-build-core （CMake的python扩展，用于构建python扩展）
        - 让 pip install 时自动调用CMake构建的c++代码
        - 把CMake的产物（.so/.pyd） 正确打包进wheel
        - 用PEP 517/ PEP 660标准接口
    - setuptools （最传统的Python打包工具，让经过pybind绑定过的c++扩展能像普通package一样被pip install）
- Compiler
    - Linux: GCC >= 9 / Clang >= 10
    - macOS: Apple Clang >= 12
    - Windows: MSVC 2019+ (v19.20+)
    - 统一要求: C++17 支持(若用 C++20 特性则提至对应版本)
- Test: pytest
- platform: MacOS + Windows + linux
# Module Frontier
在第一个milstone里面，我们集中构建四个module:
- Engine
- Scheduler
    schedule主要负责request的分配
- BackEnd
    Backend是执行层，负责把token喂给模型，跑前向传播，返回输出token
- Request
# Minimum Data Structure