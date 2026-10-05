from dataclasses import dataclass


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    # prev_free_block: Optional["Block"] = None
    # next_free_block: Optional["Block"] = None


class BlockPool:
    def __init__(self,Pool_spec:dict):
        self.Pool_spec = Pool_spec
        self.free_blocks = []
        self.block_size = Pool_spec["block_size"]
        self.num_blocks = Pool_spec["num_blocks"]
        self.pool = []
        for i in range(self.num_blocks):
            self.pool.append(Block(block_id=i))
            self.free_blocks.append(self.pool[i])

    def getBlock(self,block_id: int) -> Block:
        return self.pool[block_id]
    
    def allocate(self):
        """
        申请一个 block,返回block ID
        """
        if self.free_blocks:
            block = self.free_blocks.pop(0)
            block.ref_count += 1
            
            return block.block_id
        else:
            raise ValueError("Free block pool is empty")
    
    def free_block(self,block: Block):
        """
        释放一个 block
        """
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)
        return
