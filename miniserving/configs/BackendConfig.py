from miniserving.configs.BlockManagerConfig import BlockManagerConfig
from dataclasses import dataclass, field
@dataclass
class BackendConfig:
    model: str = "GPT2"
    backend_type: str = "torch"
    blockmanager_config: BlockManagerConfig = field(default_factory=BlockManagerConfig)
