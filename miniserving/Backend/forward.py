
import logging

import torch
import torch.nn as nn


class forward:
    def __init__(self,model):
        self.model = model
        self.wte = self.model.transformer.wte #Embedding vector
        self.spe = self.model.transformer.wpe #position Embedding
        self.layers = self.model.transformer.h #layers
        self.num_layers = len(self.layers)
        self.logger = logging.getLogger(__name__)


        # === 从 model.config 读模型结构相关字段 === 
        cfg = self.model.config 
        self.dim = cfg.n_embd # 768  (hidden size) 
        self.num_heads = cfg.n_head # 12   (attention heads) 
        self.head_dim = self .dim // self .num_heads # 64 
        # GPT2 是 MHA 不是 GQA，num_kv_heads == num_heads 
        self.num_kv_heads = self.num_heads # 12 
        self.vocab_size  = cfg.vocab_size # 50257 
        self.num_layers  = cfg.n_layer # 12   (跟 len(layers) 一致)

        # === KV 缓存配置 ===
        self.num_blocks = 4
        self.block_size = 2
        self.dtype = torch.float16

        self.block_table = []
        self.k_tensor,self.v_tensor = self.init_real_memory()

    def forward(self,inputs):
        """
        前向传播
        """
        pass
    
    def embedding(self,input_id:list[int]):
        """
        获取token的embedding
        return hidden state [len(input_id),hidden_dim]
        """
        input_id = torch.tensor(input_id)

        return self.wte(input_id) 
    
    def map_qkv(self,hidden_states):
        """
        hidden_states: [N,dim]
        return: [N,3*dim]
        """
        return self.layers[0].attn.c_attn(hidden_states)
    
    def write_to_cache(self, k_states, v_states, slot, layer_idx):
        """
        写入缓存
        k_states and v_states : [batch_size, num_kv_heads, num_new_token, head_dim]
        slot : [batch_size * num_new_token] 每个 token 在缓存中的位置 (block_id * block_size + offset)
        layer_idx : layer index
        """
        k_cache = self.k_tensor[layer_idx]
        v_cache = self.v_tensor[layer_idx]
        k_flat = k_cache.view(-1, self.num_kv_heads, self.head_dim)
        v_flat = v_cache.view(-1, self.num_kv_heads, self.head_dim)
        k_new = k_states.permute(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_dim)
        v_new = v_states.permute(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_dim)
        k_flat[slot] = k_new
        v_flat[slot] = v_new

    def init_real_memory(self):
        k_tensors = []
        v_tensors = []
        # 初始化 KV 缓存
        # kv_tensor 代表每个 layer 对应的所有缓存
        for layer in range(self.num_layers):
            try:
                self.logger.info(f"Try to init the layer {layer} CUDA memory...")
                k_tensor = torch.empty((self.num_blocks,self.block_size,self.num_kv_heads,self.head_dim), 
                dtype=self.dtype,
                device = "cuda",
                out = None,
                layout = None,
                requires_grad = False)
                v_tensor = torch.empty((self.num_blocks,self.block_size,self.num_kv_heads,self.head_dim), 
                dtype=self.dtype,
                device = "cuda",
                out = None,
                layout = None,
                requires_grad = False)

            except Exception as e:
                self.logger.error(f"KV 缓存初始化失败：{e}")
                raise

            self.logger.info(f"Layer {layer} K 缓存初始化完成，大小为：{k_tensor.shape}，dtype为：{k_tensor.dtype},device:{k_tensor.device},dim:{k_tensor.dim()},总元素数：{k_tensor.numel()}")
            self.logger.info(f"Layer {layer} 每个 block 元素数为：{k_tensor.numel() / self.block_size}，总共的字节数：{k_tensor.numel() * k_tensor.dtype.itemsize} bytes")
            self.logger.info(f"Layer {layer} V 缓存初始化完成，大小为：{v_tensor.shape}，dtype为：{v_tensor.dtype},device:{v_tensor.device},dim:{v_tensor.dim()},总元素数：{v_tensor.numel()}")          
            self.logger.info(f"Layer {layer} 每个 block 元素数为：{v_tensor.numel() / self.block_size}，总共的字节数：{v_tensor.numel() * v_tensor.dtype.itemsize} bytes") 
            k_tensors.append(k_tensor)
            v_tensors.append(v_tensor)

        return k_tensors, v_tensors
        return k_tensors,v_tensors