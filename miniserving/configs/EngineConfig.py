from miniserving.configs.BackendConfig import BackendConfig
from miniserving.configs.SchedulerConfig import SchedulerConfig
from dataclasses import dataclass, field

@dataclass
class EngineConfig:
    model: str = "/root/models/gpt2"
    backend_type: str = "torch"

    backend_config: BackendConfig = field(default_factory=BackendConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)


engineConfig = EngineConfig()