// 算子声明，唯一对外接口
// 是bindings.cpp中的函数和具体的算子实现之间的桥梁，只提供函数签名
#pragma once
#include <torch/extension.h>

void reshape_and_cache(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping
);



