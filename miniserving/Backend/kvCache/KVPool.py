import logging

import torch


class KVPool:
    def __init__(self,config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.block_size = config.block_size
        self.layer_num = config.n_layer
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.n_embd
        self.dtype = getattr(torch, config.torch_dtype)
        self.num_blocks = config.num_blocks
        self.byte_per_block = None
        self.k_tensor,self.v_tensor = self.init_real_memory() #shape[num_layer,num_blocks,num_kv_heads,head_dim]


    def init_real_memory(self):
        k_tensors = []
        v_tensors = []
        # 初始化 KV 缓存
        # kv_tensor 代表每个 layer 对应的所有缓存
        for layer in range(self.layer_num):
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
                return

            self.logger.info(f"Layer {layer} K 缓存初始化完成，大小为：{k_tensor.shape}，dtype为：{k_tensor.dtype},device:{k_tensor.device},dim:{k_tensor.dim()},总元素数：{k_tensor.numel()}")
            self.logger.info(f"Layer {layer} 每个 block 元素数为：{k_tensor.numel() / self.block_size}，总共的字节数：{k_tensor.numel() * k_tensor.dtype.itemsize} bytes")
            self.logger.info(f"Layer {layer} V 缓存初始化完成，大小为：{v_tensor.shape}，dtype为：{v_tensor.dtype},device:{v_tensor.device},dim:{v_tensor.dim()},总元素数：{v_tensor.numel()}")          
            self.logger.info(f"Layer {layer} 每个 block 元素数为：{v_tensor.numel() / self.block_size}，总共的字节数：{v_tensor.numel() * v_tensor.dtype.itemsize} bytes") 
            k_tensors.append(k_tensor)
            v_tensors.append(v_tensor)
        
        return k_tensors,v_tensors
    
    
    