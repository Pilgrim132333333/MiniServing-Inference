from miniserving.Backend.kvCache.Block import BlockPool,Block
from miniserving.Engine.Sequence import Sequence,SequenceGroup
import math

class BlockManager:
    def __init__(self,model_config,block_size: int):
        self.blockPool = BlockPool(block_size)
        self.model_config = model_config
        self.layer_num = model_config.n_layer
        self.num_kv_heads = model_config.num_key_value_heads
        self.head_dim = model_config.n_embd
        self.dtype = model_config.torch_dtype
        self.byte_per_block = None
        self.block_table = {} # sequence (str)-> block id(str)
    
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
    
    def allocate_sequence(self,seq: Sequence):
        """
        为当前序列分配 block
        """
        num_blocks = self.caculate_num_blocks(seq)
        free_blocks = self.blockPool.get_free_blocks()

        if len(free_blocks) < num_blocks:
            raise ValueError("Free block pool is not enough to allocate block for current sequence")
        
        else:
            allocated = []
            for i in range(num_blocks):
                allocated.append(free_blocks[i])
                free_blocks.remove(free_blocks[i])
            self.update_allocated_table(allocated,seq)
            return allocated
        
    def allocate_sequence_group(self,seqGroup: SequenceGroup):
        """
        为当前序列组分配 block
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
