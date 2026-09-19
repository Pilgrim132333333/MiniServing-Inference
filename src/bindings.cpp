#include <pybind11/pybind11.h>

namespace py = pybind11;

// 一个最简单的函数，用来验证编译链路是否畅通
int add(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(_miniserving, m) {
    m.doc() = "miniserving c++ extensions";
    m.def("add", &add, "Add two integers");
}
