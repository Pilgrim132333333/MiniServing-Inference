from transformers.cache_utils import Cache,CacheLayerMixin
from miniserving.Backend.kvCache.BlockManager import BlockManager
import torch
from miniserving.utils import logger

class PagedLayer(CacheLayerMixin):
    def __init__(self):
        super().__init__()
        self.key_tensor = None
        self.value_tensor = None

    def update(self,keys,values):
        pass

class KvCache(Cache):
    def __init__(self,kv_config,block_manager: BlockManager,sequenceID: str):
       
        self.config = kv_config
        self.layer_num = kv_config.n_layer
        self.num_kv_heads = kv_config.num_key_value_heads
        self.head_dim = kv_config.n_embd
        self.dtype = kv_config.torch_dtype
        self.byte_per_block = None
        self.block_Manager = block_manager
        self.block_size = kv_config.block_size
        self.sequence_id = sequenceID
        super().__init__(layers=pagelayers)
        
        return
    
    def load_memory(self):
        """
        全局唯一一次申请内存
        """
        self.init_all_layers()
    
    def update(self,key_states, value_states, layer_idx, cache_kwargs):
        pass
