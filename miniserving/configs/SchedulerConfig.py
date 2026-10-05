from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    batch_size: int = 16
    chunk_block: int = 16
    