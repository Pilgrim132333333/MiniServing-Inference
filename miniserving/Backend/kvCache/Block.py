from dataclasses import dataclass


@dataclass
class Block:
    block_id: int
    prev_free_block: Optional["Block"] = None
    next_free_block: Optional["Block"] = None


class BlockPool:
    def __init__(self,block_size: int,num_blocks: int):
        self.block_table = {}   #
        self.free_blocks = []
        self.block_size = block_size
    
    def 