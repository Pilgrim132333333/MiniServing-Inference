from miniserving.configs.BlockManagerConfig import BlockManagerConfig
from dataclasses import dataclass, field
@dataclass
class WorkerConfig:
    model: str = "GPT2"
    worker_type: str = "torch"
    blockmanager_config: BlockManagerConfig = field(default_factory=BlockManagerConfig)
    kvcachemanager_config: BlockManagerConfig = field(default_factory=BlockManagerConfig)
