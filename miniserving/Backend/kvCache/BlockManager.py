from miniserving.Backend.kvCache.Block import BlockPool,Block
from miniserving.Backend.kvCache.KvCache import KvCache

from miniserving.Engine.Sequence import Sequence,SequenceGroup
import math

class BlockManager:
    def __init__(self,model_config,block_size: int,cache:KvCache):
        self.model_config = model_config
        self.layer_num = model_config.n_layer
        self.num_kv_heads = model_config.num_key_value_heads
        self.head_dim = model_config.n_embd
        self.dtype = model_config.torch_dtype
        self.byte_per_block = None
        self.block_table = {} # cache-> list[Block]
        self.cache = cache
        self.k_tensor,self.v_tensor = self.init_layer_KV_Cache() #shape[num_blocks,batch_size,num_kv_heads,head_dim]
        self.k_pool = BlockPool()
        self.v_pool = BlockPool()

        
    def calculate_single_layer_single_token(self):
        """
        计算单 token 的 key-value 缓存大小
        """
        return self.num_kv_heads * self.head_dim * self.dtype.itemsize() * 2
    
    def calculate_model_single_token(self):
        """
        计算模型单 token 的 key-value 缓存大小
        """
        return self.layer_num * self.calculate_single_layer_single_token()
    
    def load_byte_per_block(self):
        """
        计算每个 block 的大小
        """   
        self.byte_per_block = self.calculate_model_single_token()*block_size
        return
    
    def caculate_num_blocks(self,seq: Sequence):
        """
       �算当前序列计算需要的 block 数
        """
        input_tokens = seq.get_input_tokens()
        num_blocks = math.ceil(len(input_tokens) / block_size)
        return num_blocks
    
    def check_remaining_blocks(self,seq: Sequence):
        """
        检查当前序列是否有足够的 block
        """
        num_blocks = self.caculate_num_blocks(seq)
        return num_blocks <= len(self.k_pool.free_blocks) and num_blocks <= len(self.v_pool.free_blocks)
    
    def allocate_sequence(self,seq: Sequence):
        """
        为当前序列分配 block
        """

        if not self.check_remaining_blocks(seq):
            raise ValueError("Free block pool is not enough to allocate block for current sequence")
        
        else:
            allocated = []
            num_blocks = self.caculate_num_blocks(seq)
            for i in range(num_blocks):
                allocated.append(self.blockPool.allocate())
            self.update_allocated_table(allocated,seq)
            return allocated
        
    def allocate_sequence_group(self,seqGroup: SequenceGroup):
        """
        为当前序列组分配 block, 这个fuc只适用于prefill阶段
        """
        example_seq = seqGroup.sequences[0]
        allocated = self.allocate_sequence(example_seq)
        for seq in seqGroup.sequences:
            self.update_allocated_table(allocated,seq)
        return

    def update_allocated_table(self,allocated: list[Block],cache: Cache):
        """
        更新已分配 block 表
        """
        self.block_table[cache] = allocated
        return
    
    def get_allocated_blocks(self,cache: Cache):
        """
        获取当前序列已分配的 block id
        """
        if cache not in self.block_table:
            return []
        return self.block_table[cache]
    
    def calculate_slot(self,cache: Cache):
        """
        计算当前序列的 slot 数
        """
        allocated_blocks = self.get_allocated_blocks(cache)
        num_tokens = cache.num_tokens
        num_blocks = len(allocated_blocks)
        slot = num_blocks % ((num_blocks-1 )* self.block_size)
        return slot
    
    def check_if_allocate_block(self,cache: Cache):
        """
        判断这个sequence是否需要新分配block
        """
        
        num_token = cache.num_tokens
        slot = num_token % self.block_size
        last_block = cache.last_block
        
        if slot==self.block_size:
            return True
        elif slot>last_block.slot_size:
            raise ValueError(f"Slot Error: {slot}")
        else:
            if last_block.ref_count>0:
                return True
            else:
                return False
        

    def init_real_memory(self):
        k_tensors = []
        v_tensors = []
        for layer in range(self.layer_num):
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

            logger.info(f"Layer {layer} K 缓存初始化完成，大小为：{k_tensor.shape}，dtype为：{k_tensor.dtype},device:{k_tensor.device},dim:{k_tensor.dim()},总元素数：{k_tensor.numel()}")
            logger.info(f"Layer {layer} 每个 block 元素数为：{k_tensor.numel() / self.block_size}，总共的字节数：{k_tensor.numel() * k_tensor.dtype.itemsize()} bytes")
            logger.info(f"Layer {layer} V 缓存初始化完成，大小为：{v_tensor.shape}，dtype为：{v_tensor.dtype},device:{v_tensor.device},dim:{v_tensor.dim()},总元素数：{v_tensor.numel()}")          
            logger.info(f"Layer {layer} 每个 block 元素数为：{v_tensor.numel() / self.block_size}，总共的字节数：{v_tensor.numel() * v_tensor.dtype.itemsize()} bytes") 
            k_tensors.append(k_tensor)
            v_tensors.append(v_tensor)
        
        return k_tensors,v_tensors
    
    def next_token_block(self,cache: Cache):
        """
        获取下一个token对应的block ID
        """
        if self.check_if_allocate_block(cache):
            return [self.k_pool.allocate(),self.v_pool.allocate()]
        else:
            return last_block.block_id
        
    def update_layer(self,cache: Cache,key_states,value_states,layer_index: int):

        """
        更新当前layer的缓存, 
        key_states and value_states : [batch_size,num_kv_heads,num_new_token,head_dim]
        在判断Cow后，我们仍然需要区分是进入prefill阶段还是decode阶段
        """
        next_block = self.next_token_block(cache)
        last_block = cache.block_table[-1]
        num_token = cache.num_tokens
        if next_block == last_block.block_id:
            """
            不需要Cow,直接在当前block续写
            """
        else:
            """
            需要Cow,创建新block
            """
            filled = num_token % self.block_size
            
            new_view = self.k_pool[layer_index][next_block]
            new_view[:filled].copy_(self.k_pool[layer_index][last_block.block_id][:filled])
            new_view = self.v_pool[layer_index][next_block]
            new_view[:filled].copy_(self.v_pool[layer_index][last_block.block_id][:filled])
        
        """
        判断是否需要进入prefill阶段还是decode阶段
        """
        if sequence.output_tokens is None or len(sequence.output_tokens)==0:
            """
            进入prefill阶段
            """
            self.prefill_key_value_states(next_block,filled,key_states,value_states,layer_index)

        else:
            """
            进入decode阶段
            """
            self.decode_key_value_states(next_block,filled,key_states,value_states,layer_index)
    
    def prefill_key_value_states(self,next_block: int,filled: int,key_states,value_states,layer_index: int):
        """
        填充新block的缓存
        这里的key_states 和 value_states 都是 [num_kv_heads,num_token,head_dim]
        我们需要将它们复制到新block的缓存中
        """
        remaining = num_token - filled
        num_tokens = key_states.shape[1]
        if num_tokens > remaining:
            num_write_token = 0
            num_write_token += remaining
            for i in range(remaining):
                new_view = self.k_pool[layer_index][next_block]
                new_view[:filled+i].copy_(key_states[:,i,:])
                new_view[:filled+i].copy_(value_states[:,i,:])
            
            #allocate new block
            while num_write_token < num_tokens:
                new_block = self.blockPool.allocate()
                cur_block_filled = 0
                while cur_block_filled < self.block_size and num_write_token < num_tokens:
                    new_view = self.k_pool[layer_index][new_block]
                    new_view[cur_block_filled].copy_(key_states[:,num_write_token,:])
                    new_view[cur_block_filled].copy_(value_states[:,num_write_token,:])
                    cur_block_filled += 1
                    num_write_token += 1
        else:
            for i in range(num_tokens):
                new_view = self.k_pool[layer_index][next_block]
                new_view[:filled+i].copy_(key_states[:,i,:])
                new_view[:filled+i].copy_(value_states[:,i,:])
    
    def decode_key_value_states(self,next_block: int,filled: int,key_states,value_states,layer_index: int):
        """
        解码新block的缓存
        这里的key_states 和 value_states 都是 [num_kv_heads,1,head_dim]
        我们需要将它们复制到新block的缓存中
        """
        new_view = self.k_pool[layer_index][next_block]
        new_view[:filled].copy_(key_states[:,0,:])
        new_view = self.v_pool[layer_index][next_block]
        new_view[:filled].copy_(value_states[:,0,:])

