from dataclasses import dataclass


@dataclass
class BlockManagerConfig:
    block_size: int = 1024
    num_blocks: int = 256
    n_layer: int = 12
    num_key_value_heads: int = 8
    n_embd: int = 768
    torch_dtype: str = "float16"
    device: str = "cuda"