"""
此算子是第一个CUDA算子，用于替代在Model上write_to_cache里面的for循环：
# 原来的写法（Python 循环逐 slot 写）
k_flat = k_cache.view(-1, self.num_kv_heads, self.head_dim)
k_new = k_states.permute(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_dim)
for i in range(len(slot)):          # ← Python 循环，每个 token 一次
    k_flat[slot[i]] = k_new[i]      # ← 一次 GPU kernel launch
    v_flat[slot[i]] = v_new[i]      # ← 又一次 launch

当换成CUDA算子以后，一次launch,GPU上24576个thread并行完成同样的搬运
"""

template <typename scalar_t>  //模版参数，让同一个kernel同时支持fp16/fp32/fp64 编译器为每一个数据类型生成一个机器码
__global__ void reshape_and_cache_kernel(
    //CUDA关键字，表示CPU发起，在GPU执行。调用时用<<<grid,threads>>>语法

    //源数据裸指针，pybind层传进来的tensor
    //_restrict_表示向编译器承诺，没有别的指针指向这块内存，让它放开最激进的优化
    const scalar_t* _restrict_ key //[num_tokens,num_heads,num_dim]
    const scalar_t* _restrict_ value //[num_tokens,num_heads,num_dim]

    const int64_t* _restrict_ slot_mapping //[num_tokens] 每个新token的slot号
    
    const int64_t num_tokens;
    const int64_t vec_size; // num_heads*num_dim
    const int64_t cache_total_slots; //num_blocks*block_size

    const int64_t token_idx = blockIdx.x //这里对应grid的x，表示每个block正在处理的tokenIdx

    if(token_idx >= num_tokens) return;
    //从显存读出本token的目标slot，注意每个thread都要读取一次
    const int64_t slot = slot_mapping[token_idx];

    if(slot >= cache_total_slots) return;

    const int64_t vec_idx = threadIdx.x + blockIdx.y* blockDim.x;
    if(vec_idx < vec_size){
        const int64_t src = token_idx*vec_size + vec_idx; //源tensor里滴token_idx个token的第vec_idx个元素
        const int64_t tgt = slot*vec_size + vec_idx; //KV pool里第slot个槽位的第vec_idx个元素

        key_cache[tgt] = key[src];
        value_cache[tgt] = value[src];
    }

)

//CPU侧封装函数，python调用后的一切准备工作
void reshape_and_cache(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor slot_mapping){
        TORCH_CHECK(key.is_cuda() && value.is_cuda(), "key/value must be CUDA tensors");
        TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(), "k_cache/v_cache must be CUDA tensors");
        TORCH_CHECK(key.scalar_type() == key_cache.scalar_type(),"dtypy mismatch")
        TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt64, "slot_mapping must be int64");
        
        const int64_t num_tokens = key.size(0);
        const int64_t vec_size = key.size(1) * key.size(2);
        const int64_t cache_total_slots = k_cache.size(0) / vec_size;

        const int64_t threads = 128
        //向上取整： ceil(a/b) = (a+b-1)/b
        const int64_t blocks_y = (vec_size + threads - 1) / threads;

        dim3 grid(num_tokens, blocks_y);
        //kernel 会launch到·这个stream上
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        //展开是一个switch-case
        """
        `<<<grid, threads, 0, stream>>>` ：launch 四件套——grid 布局、block 大小、动态共享内存字节数（0 = 不用）、目标流
        `data_ptr<scalar_t>()` ：从 tensor 取裸指针。泛型版本会校验 dtype 匹配（不匹配抛异常），这是最后一道防线
        只把kernel发射出去，CPU立即返回，不等GPU干完
        """
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            key.scalar_type(), "reshape_and_cache", ([&] {
            reshape_and_cache_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
                k_cache.data_ptr<scalar_t>(), v_cache.data_ptr<scalar_t>(),
                slot_mapping.data_ptr<int64_t>(), num_tokens, vec_size, cache_total_slots);
        }));
}
