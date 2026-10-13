
import torch
from miniserving.Backend.forward import forward
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/root/models/gpt2")

forward = forward(model)

def test_forward_embedding():
    token_ids = [[1,2,3]]                 # [batch=1, seq=3]
    embedding = forward._embedding(token_ids)
    print(embedding)
    assert embedding.shape == (1, 3, 768)

def test_forward_map_qkv():
    token_ids = [[1,2,3]]
    hidden_states = forward._embedding(token_ids)   # [1, 3, 768]
    qkv = forward._map_qkv(hidden_states, 0)         # [1, 3, 2304]
    q = qkv[..., 0:768]      # 等价 qkv[..., :768]
    k = qkv[..., 768:1536]   # 等价 qkv[..., 768:2*768]
    v = qkv[..., 1536:2304]  # 等价 qkv[..., 2*768:]
    assert q.shape == (1, 3, 768)
    assert k.shape == (1, 3, 768)
    assert v.shape == (1, 3, 768)
    assert qkv.shape == (1, 3, 3*768)

def test_write_to_cache():
    token_ids = [[1,2,3]]
    hidden_states = forward._embedding(token_ids)   # [1, 3, 768]
    qkv = forward._map_qkv(hidden_states, 0)
    k = qkv[..., 768:1536]   # [1, 3, 768]
    v = qkv[..., 1536:2304]  # [1, 3, 768]

    k_states = k.view(1, 3, forward.num_kv_heads, forward.head_dim).permute(0, 2, 1, 3)  # [1,12,3,64]
    v_states = v.view(1, 3, forward.num_kv_heads, forward.head_dim).permute(0, 2, 1, 3)
    slot = [0,1,2]
    layer_idx = 0
    forward._write_to_cache(k_states, v_states, slot, layer_idx)
    k_cache = forward.k_tensor[layer_idx]
    v_cache = forward.v_tensor[layer_idx]
    assert k_cache.shape == (4,2,12,64)
    assert v_cache.shape == (4,2,12,64)
    assert k_cache[slot[0]].shape == (2,12,64)
    assert v_cache[slot[0]].shape == (2,12,64)

    k_flat = k_cache.view(-1, forward.num_kv_heads, forward.head_dim)
    v_flat = v_cache.view(-1, forward.num_kv_heads, forward.head_dim)
    print(k_flat)
    print(v_flat)

    # cast dtype 的同时搬到与缓存相同的设备，避免 torch.equal 跨设备报错
    k_exp = k_states.to(device=k_cache.device, dtype=forward.dtype) # [1,12,3,64]
    v_exp = v_states.to(device=k_cache.device, dtype=forward.dtype)

    for i, s in enumerate(slot):
        # s = block_id * block_size + offset
        assert torch.equal(k_flat[s], k_exp[0, :, i, :]), f"K slot {s} (token {i}) 写入不正确"
        assert torch.equal(v_flat[s], v_exp[0, :, i, :]), f"V slot {s} (token {i}) 写入不正确"
        # 交叉验证: block 视角与 slot 视角指向同一块内存
        block_id, offset = s // forward.block_size, s % forward.block_size
        assert torch.equal(k_cache[block_id, offset], k_flat[s])
    # 未写入区域不应被误写: slot 3 之外的位置不应包含 token 数据
    assert not torch.equal(k_flat[3], k_exp[0, :, 0, :])

def test_find_past_key_value():
    """
    测试查找历史缓存
    """
    sequence_id = "0"
    token_first = [[1,2,3]]

    first_hidden_states = forward._embedding(token_first)   # [1,3,768]
    first_qkv = forward._map_qkv(first_hidden_states, 0)    # [1,3,2304]

    k = first_qkv[..., 768:1536]   # [1,3,768]
    v = first_qkv[..., 1536:2304]  # [1,3,768]

    k_state_first = k.view(1, 3, forward.num_kv_heads, forward.head_dim).permute(0, 2, 1, 3)  # [1,12,3,64]
    v_state_first = v.view(1, 3, forward.num_kv_heads, forward.head_dim).permute(0, 2, 1, 3)
    slot = [0,1,2]
    layer_idx = 0
    forward._write_to_cache(k_state_first, v_state_first, slot, layer_idx)

    forward.block_table[sequence_id] = [0,1]
    k_past, v_past = forward._find_past_kv(sequence_id, layer_idx, 3)

    assert k_state_first.shape == k_past.shape
    assert v_state_first.shape == v_past.shape

    k_exp = k_state_first.to(device=k_past.device, dtype=forward.dtype)
    v_exp = v_state_first.to(device=v_past.device, dtype=forward.dtype)
    assert torch.equal(k_past, k_exp), "K 数据与写入不一致"
    assert torch.equal(v_past, v_exp), "V 数据与写入不一致"

def test_single_attention():
    token_ids = [[1,2,3]]
    hidden_states = forward._embedding(token_ids)   # [1,3,768]
    qkv = forward._map_qkv(hidden_states, 0)        # [1,3,2304]

    k = qkv[..., 768:1536]   # [1,3,768]
    v = qkv[..., 1536:2304]  # [1,3,768]

    k = k.view(1, 3, forward.num_kv_heads, forward.head_dim).permute(0, 2, 1, 3)  # [1,12,3,64]
    v = v.view(1, 3, forward.num_kv_heads, forward.head_dim).permute(0, 2, 1, 3)

    input_id = [[4]]
    input_hidden_states = forward._embedding(input_id)   # [1,1,768]
    input_qkv = forward._map_qkv(input_hidden_states, 0) # [1,1,2304]
    q = input_qkv[..., 0:768]                             # [1,1,768]
    q = q.view(1, 1, forward.num_kv_heads, forward.head_dim).permute(0, 2, 1, 3)  # [1,12,1,64]

    attention_output = forward._attention(q, k, v)
    assert attention_output.shape == (1, 12, 1, 64)   # [batch, H, seq_q, D]

def test_residual_connect():
    """
    残差连接测试：[N,768] + [1,768]
    验证相加后的形状、首行等于二者相加、其余行未变。
    """
    N = 5
    H = forward.dim   # 768
    hidden_states = torch.randn(N, H)
    attn_out = torch.randn(1, H)         # 模拟 attention 输出（仅 1 个 token）

    # 残差：将 [1,768] 广播加到 hidden_states 的第一行
    result = hidden_states.clone()
    result[0:1] = result[0:1] + attn_out

    # 1. 形状不变
    assert result.shape == (N, H), f"shape mismatch: {result.shape}"
    # 2. 首行 = 原 hidden_states[0] + attn_out
    assert torch.allclose(result[0], hidden_states[0] + attn_out[0])
    # 3. 其余行未受影响
    assert torch.allclose(result[1:], hidden_states[1:])


def test_process_attn_output():
    token_ids = [[1,2,3]]
    hidden_states = forward._embedding(token_ids)   # [1,3,768]
    qkv = forward._map_qkv(hidden_states, 0)        # [1,3,2304]

    k = qkv[..., 768:1536]   # [1,3,768]
    v = qkv[..., 1536:2304]  # [1,3,768]

    k = k.view(1, 3, forward.num_kv_heads, forward.head_dim).permute(0, 2, 1, 3)  # [1,12,3,64]
    v = v.view(1, 3, forward.num_kv_heads, forward.head_dim).permute(0, 2, 1, 3)

    input_id = [[4]]
    input_hidden_states = forward._embedding(input_id)
    input_qkv = forward._map_qkv(input_hidden_states, 0)
    q = input_qkv[..., 0:768]
    q = q.view(1, 1, forward.num_kv_heads, forward.head_dim).permute(0, 2, 1, 3)  # [1,12,1,64]

    attention_output = forward._attention(q, k, v)   # [1,12,1,64]
    attention_output = forward._process_attn_output(attention_output, 0)
    assert attention_output.shape == (1, 768)


def test_MLP():
    N = 5
    H = forward.dim   # 768
    hidden_states = torch.randn(1, N, H, device=forward.device, dtype=forward.dtype)   # [batch, seq, 768]
    mlp_output = forward._MLP(hidden_states, 0)

    mlp = forward.layers[0].mlp
    mlp_mid = mlp.c_fc(hidden_states)      # [1, N, 3072]
    mlp_mid = mlp.act(mlp_mid)
    mlp_mid = mlp.c_proj(mlp_mid)          # [1, N, 768]

    assert mlp_mid.shape == (1, N, H)
    assert mlp_output.shape == (1, N, H)
    assert torch.allclose(mlp_mid, mlp_output, atol=1e-3), \
        "手动分步调用结果应与 HF 原版 mlp(...) 一致"    
    


def test_forward():
    token_ids = [[1,2,3]]              # [batch=1, seq=3]
    output = forward.forward(token_ids,"0")   # forward.__call__ -> forward.forward
    assert output.shape == (1, 3, 768)
