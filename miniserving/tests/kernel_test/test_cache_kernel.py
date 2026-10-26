# 验证 reshape_and_cache CUDA 算子与原 Python for 循环写入结果一致
import pytest
import torch

from miniserving.kernels import _C

def ref_write_to_cache(key, value, k_cache, v_cache, slot_mapping):
    # 原实现：逐 token 两次赋值，作为 ground truth
    for i, slot in enumerate(slot_mapping.tolist()):
        k_cache[slot] = key[i].reshape(-1)
        v_cache[slot] = value[i].reshape(-1)

def make_inputs(num_tokens, num_heads, head_dim, num_slots, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    key = torch.randn(num_tokens, num_heads, head_dim, generator=g).to(dtype).cuda()
    value = torch.randn(num_tokens, num_heads, head_dim, generator=g).to(dtype).cuda()
    vec = num_heads * head_dim
    k_pool = torch.zeros(num_slots, vec, dtype=dtype, device="cuda")
    v_pool = torch.zeros(num_slots, vec, dtype=dtype, device="cuda")
    slots = torch.randperm(num_slots, generator=g)[:num_tokens].cuda().to(torch.int64)
    return key, value, k_pool, v_pool, slots

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
@pytest.mark.parametrize("num_tokens,num_heads,head_dim,num_slots", [
    (7, 2, 8, 32),       # 非 2 次幂，触发 grid.y 分块
    (1, 1, 4, 8),        # 极端小规模
    (64, 12, 64, 256),   # 接近 GPT2 规模
])
def test_matches_python_loop(dtype, num_tokens, num_heads, head_dim, num_slots):
    key, value, k_pool, v_pool, slots = make_inputs(num_tokens, num_heads, head_dim, num_slots, dtype)
    ref_k, ref_v = k_pool.clone(), v_pool.clone()

    ref_write_to_cache(key, value, ref_k, ref_v, slots)
    _C.reshape_and_cache(key, value, k_pool, v_pool, slots)
    torch.cuda.synchronize()

    torch.testing.assert_close(k_pool, ref_k)
    torch.testing.assert_close(v_pool, ref_v)
    # 未被选中的 slot 必须保持为 0，证明没有越界写
    untouched = torch.ones(num_slots, dtype=torch.bool)
    untouched[slots.cpu()] = False
    assert (k_pool[untouched.cuda()] == 0).all()

def test_shape2d_input_also_works():
    # decode 阶段 key 已是 [num_tokens, vec_size]，走同一算子
    key = torch.randn(4, 16, device="cuda")
    value = torch.randn(4, 16, device="cuda")
    k_pool = torch.zeros(16, 16, device="cuda")
    v_pool = torch.zeros(16, 16, device="cuda")
    slots = torch.tensor([3, 7, 0, 15], dtype=torch.int64, device="cuda")

    ref_k, ref_v = k_pool.clone(), v_pool.clone()
    ref_write_to_cache(key, value, ref_k, ref_v, slots)
    _C.reshape_and_cache(key, value, k_pool, v_pool, slots)
    torch.cuda.synchronize()
    torch.testing.assert_close(k_pool, ref_k)
    torch.testing.assert_close(v_pool, ref_v)

def test_dtype_mismatch_raises():
    key = torch.randn(2, 4, device="cuda", dtype=torch.float32)
    value = torch.randn(2, 4, device="cuda", dtype=torch.float32)
    k_pool = torch.zeros(8, 4, device="cuda", dtype=torch.float16)
    v_pool = torch.zeros(8, 4, device="cuda", dtype=torch.float16)
    slots = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError):
        _C.reshape_and_cache(key, value, k_pool, v_pool, slots)

def test_interleaved_qkv_slice_view():
    """回归测试：qkv 交织切片视图——本次集成 bug 的真实形态
    c_attn 输出逐 token 交织 [q_t|k_t|v_t]，split 出的 k/v 是非连续视图，
    permute+reshape 后 stride=(3*vec,1)。旧版 kernel 假设 768 连排读错数据；
    B3 版按 stride 原地读，必须与连续输入产出逐位相同的结果
    """
    torch.manual_seed(0)
    T, H, D = 7, 2, 8
    vec = H * D
    qkv = torch.randn(1, T, 3 * vec, device="cuda").half()   # 模拟 c_attn 输出
    _q, k, v = qkv.split(vec, dim=-1)                       # 非连续切片视图（_q本测试不用）
    k_states = k.view(1, T, H, D).transpose(1, 2)            # [B,H,S,D] 视图链
    v_states = v.view(1, T, H, D).transpose(1, 2)
    k_in = k_states.permute(0, 2, 1, 3).reshape(-1, vec)     # [T,vec] stride=(3*vec,1)
    v_in = v_states.permute(0, 2, 1, 3).reshape(-1, vec)
    assert not k_in.is_contiguous(), "测试前提：输入必须是非连续视图"

    k_ref = k_in.contiguous()                                # 逻辑值完全相同的连续版
    v_ref = v_in.contiguous()

    k_pool = torch.zeros(16, vec, dtype=torch.float16, device="cuda")
    v_pool = torch.zeros(16, vec, dtype=torch.float16, device="cuda")
    ref_k, ref_v = k_pool.clone(), v_pool.clone()
    slots = torch.tensor([5, 0, 9, 15, 3, 7, 2], dtype=torch.int64, device="cuda")

    ref_write_to_cache(k_ref, v_ref, ref_k, ref_v, slots)    # ground truth 走连续输入
    _C.reshape_and_cache(k_in, v_in, k_pool, v_pool, slots)  # 被测路径走交织视图
    torch.cuda.synchronize()
    # 纯搬运无计算，必须 bit-exact
    assert torch.equal(k_pool, ref_k), "交织视图写入与连续写入不一致"
    assert torch.equal(v_pool, ref_v), "交织视图写入与连续写入不一致"