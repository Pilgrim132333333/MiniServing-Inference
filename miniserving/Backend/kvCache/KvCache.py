from transformers import Cache,CacheLayerMixin
from miniserving.Backend.kvCache.BlockManager import BlockManager
import torch
from miniserving.utils import logger

class PagedLayer(CacheLayerMixin):
    def __init__(self):
        super().__init__()

    def update(self,keys,values):
        """
        更新当前层的 key-value 缓存，同时我们需要更新page Table，动态管理block
        """
        self.keys = keys
        self.values = values

class KvCache(Cache):
    def __init__(self,model_config,block_size: int):
        """
        self.layers[i] = DynamicLayer(
        keys   = [batch, num_kv_heads, seq_len, head_dim],   ← 第 i 层所有历史 token 的 K
        values = [batch, num_kv_heads, seq_len, head_dim],   ← 第 i 层所有历史 token 的 V
        )
        """
        self.model_config = model_config
        self.layer_num = model_config.n_layer
        self.num_kv_heads = model_config.num_key_value_heads
        self.head_dim = model_config.n_embd
        self.dtype = model_config.torch_dtype
        self.byte_per_block = None
        pagelayers = [PagedLayer() for _ in range(self.layer_num)]
        super().__init__(layers=pagelayers)
    
    def init_layer_KV_Cache(self):
        """
        初始化 KV 缓存
        kv_tensor代表每个layer对应的所有缓存
        """
       try:
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

       except:
            logger.error("KV 缓存初始化失败")
            return

        logger.info(f"K,V 缓存初始化完成，大小为：{k_tensor.shape}，dtype为：{k_tensor.dtype},device:{k_tensor.device},dim:{k_tensor.dim()},总元素数：{k_tensor.numel()}")
        logger.info(f"每个 block 元素数为：{k_tensor.numel() / self.block_size}，总共的字节数：{k_tensor.numel() * k_tensor.dtype.itemsize()} bytes")
        logger.info(f"V 缓存初始化完成，大小为：{v_tensor.shape}，dtype为：{v_tensor.dtype},device:{v_tensor.device},dim:{v_tensor.dim()},总元素数：{v_tensor.numel()}")
        logger.info(f"每个 block 元素数为：{v_tensor.numel() / self.block_size}，总共的字节数：{v_tensor.numel() * v_tensor.dtype.itemsize()} bytes")

        return k_tensor,v_tensor
    
    def init_layer(self,layer:PagedLayer):
        k_tensor,v_tensor = self.init_layer_KV_Cache()
        layer.keys = k_tensor
        layer.values = v_tensor
    
    def init_all_layers(self):
        for layer in self.layers:
            self.init_layer(layer)
        
        return