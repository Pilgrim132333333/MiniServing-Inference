
//把c++的函数登记到python侧的命名空间里

#include <pybind11/pybind11.h>
#include "cache_ops.h"
#include <torch/extension.h>

namespace py = pybind11;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reshape_and_cache", &reshape_and_cache,
          "Python 调用 reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)",
          py::arg("key"), py::arg("value"),
          py::arg("key_cache"), py::arg("value_cache"),
          py::arg("slot_mapping"));
}
