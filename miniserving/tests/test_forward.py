import pytest
import torch
from miniserving.Backend.forward import forward
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/root/models/gpt2")

forward = forward(model)

def test_forward_forward():
    pass

def test_forward_embedding():
    token_ids = [1,2,3]
    embedding = forward.embedding(token_ids)
    print(embedding)
    assert embedding.shape == (3,768)

def test_forward_map_qkv():
    token_ids = [1,2,3]
    hidden_states = forward.embedding(token_ids)
    qkv = forward.map_qkv(hidden_states)
    q = qkv[..., 0:768]      # 等价 qkv[..., :768]
    k = qkv[..., 768:1536]   # 等价 qkv[..., 768:2*768]
    v = qkv[..., 1536:2304]  # 等价 qkv[..., 2*768:]
    assert q.shape == (3,768)
    assert k.shape == (3,768)
    assert v.shape == (3,768)
    assert qkv.shape == (3,3*768)

def test_write_to_cache():
    token_ids = [1,2,3]
    hidden_states = forward.embedding(token_ids)
    qkv = forward.map_qkv(hidden_states)
    k = qkv[..., 768:1536]
    v = qkv[..., 1536:2304]
    slot = 0
    layer_idx = 0
    forward.write_to_cache(k,v,slot,layer_idx)
    k_cache = forward.k_tensor[layer_idx]
    v_cache = forward.v_tensor[layer_idx]
    assert k_cache.shape == (3,12,1,64)
    assert v_cache.shape == (3,12,1,64)
    assert k_cache[slot].shape == (12,64)
    assert v_cache[slot].shape == (12,64)
