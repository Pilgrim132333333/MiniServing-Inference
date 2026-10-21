from miniserving.configs.WorkerConfig import WorkerConfig
from miniserving.configs.SchedulerConfig import SchedulerConfig
from dataclasses import dataclass, field

@dataclass
class EngineConfig:
    model: str = "/root/models/gpt2"
    worker_type: str = "torch"

    worker_config: WorkerConfig = field(default_factory=WorkerConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)


engineConfig = EngineConfig()