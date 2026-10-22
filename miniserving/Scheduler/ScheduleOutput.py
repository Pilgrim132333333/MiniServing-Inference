from dataclasses import dataclass, field


@dataclass
class ScheduleOutput:
    """
    Scheduler 每一步的输出:账本向 Worker 传递的增量元数据。
    只含索引编号,不含任何 tensor 数据。
    权威账本持久保存在 BlockManager 中,本对象只是单步快照,用完即弃。
    """
    seqs: list = field(default_factory=list)          # 本步参与执行的 Sequence 列表
    token_ids: list = field(default_factory=list)     # 本步逐 token 的输入 id(按 batch 顺序展开)
    block_table: dict = field(default_factory=dict)   # sequence_id -> list[block_id]
    slot_mapping: list = field(default_factory=list)  # 逐 token 写入槽位 block_id * block_size + offset
    start_positions: list = field(default_factory=list) # 每个sequence这step开始的位置（已经计算的token数量）
