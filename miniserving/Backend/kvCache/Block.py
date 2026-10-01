from dataclasses import dataclass
from miniserving.Backend.kvCache.KVCache import KvCache

@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    prev_free_block: Optional["Block"] = None
    next_free_block: Optional["Block"] = None


class BlockPool:
    def __init__(self,pool):
        self.pool = pool
        self.block_table = {}   #
        self.free_blocks = []
        self.block_size = pool.block_size
    
    def allocate(self):
        """
        申请一个 block
        """
        if self.free_blocks:
            blockID = self.free_blocks.pop(0)
            self.block_table[blockID].ref_count += 1
            return self.block_table[blockID]
        else:
            raise ValueError("Free block pool is empty")
       