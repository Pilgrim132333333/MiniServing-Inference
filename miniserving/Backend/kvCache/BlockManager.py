from miniserving.Backend.kvCache.Block import BlockPool,Block
from miniserving.Backend.kvCache.KvCache import KvCache

from miniserving.Engine.Sequence import Sequence,SequenceGroup
import math

class BlockManager:
    def __init__(self,model_config,block_size: int,cache:KvCache):
        self.blockPool = BlockPool(block_size)
        self.model_config = model_config
        self.layer_num = model_config.n_layer
        self.num_kv_heads = model_config.num_key_value_heads
        self.head_dim = model_config.n_embd
        self.dtype = model_config.torch_dtype
        self.byte_per_block = None
        self.block_table = {} # sequence (str)-> block id(str)
        self.cache = cache
        
        k_pool = [] #[layer_num,num_blocks,block_size,num_kv_heads,head_dim]
        for layer in self.cache.layers:
            k_pool.append(layer.key_tensor)
        self.k_pool = BlockPool(k_pool)
        
        v_pool = [] #[layer_num,num_blocks,block_size,num_kv_heads,head_dim]
        for layer in self.cache.layers:
            v_pool.append(layer.value_tensor)
        self.v_pool = BlockPool(v_pool)

        
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
        return num_blocks <= len(free_blocks)
    
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


    def update_allocated_table(self,allocated: list[Block],seq: Sequence):
        """
        更新已分配 block 表
        """
        allocated_ids = []
        for block in allocated:
            allocated_ids.append(block.block_id)
        self.block_table[seq.sequence_id] = allocated_ids
        return
    
    def get_allocated_blocks(self,seqID: str):
        """
        获取当前序列已分配的 block id
        """
        if seqID not in self.block_table:
            return []
        return self.block_table[seqID]
    
    def calculate_slot(self,sequence: Sequence):
        """
        计算当前序列的 slot 数
        """
        sequenceID = sequence.sequence_id
        allocated_blocks = self.get_allocated_blocks(sequenceID)
        num_tokens = len(sequence.get_input_tokens())+len(sequence.get_output_tokens())
        num_blocks = len(allocated_blocks)
        slot = num_blocks % ((num_blocks-1 )* self.block_size)
        return slot
    
    def check_if_allocate_block(self,sequence: Sequence):
        """
        检查当前序列是否有足够的 block
        """
        allocated = self.block_table[sequence.sequence_id]
        slot = self.calculate_slot(sequence)
        last_block = allocated[-1]
        
        if slot==block_size:
            return True
        elif slot>last_block.slot_size:
            raise ValueError(f"Slot Error: {slot}")
        else:
            return last_block[slot+1].ref_count>0
        

