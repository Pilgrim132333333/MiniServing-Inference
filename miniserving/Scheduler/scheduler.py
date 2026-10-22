import copy
import time
from collections import deque

from transformers import DynamicCache

from miniserving.worker.ExecuteOutput import ExecuteOutput
from miniserving.configs.EngineConfig import EngineConfig
from miniserving.configs.SchedulerConfig import SchedulerConfig
from miniserving.Backend.Block.BlockManager import BlockManager
from miniserving.Scheduler.ScheduleOutput import ScheduleOutput
from miniserving.Engine.SequenceGroup import SeqGroupStatus, Sequence, SequenceGroup


class Scheduler:
    def __init__(self,config):
        self.WAITING_QUEUE = deque()
        self.RUNNING_QUEUE = deque()
        self.FINISHED_QUEUE = deque()
        self.FAILED_QUEUE = deque()

        self.config = config.scheduler_config
        self.batch_size = self.config.batch_size
        self.chunk_block = self.config.chunk_block
        #持久账本:Scheduler 侧唯一权威,跨 step 累积,序列结束才销账
        self.block_manager = BlockManager(config.worker_config.blockmanager_config)
    
    def schedule(self):
        if not self.check_remaining():
            return None
        output = ScheduleOutput()
        batch = output.seqs
        if len(self.RUNNING_QUEUE) > 0:
            #Decode Period:每个序列推进 1 个 token
            #最后一个输出 token 的位置 = len(input)+len(output)-1,其 KV 尚未计算
            sequence_group = self.RUNNING_QUEUE[0]
            for seq in sequence_group.get_sequences():
                seq.set_first_schedule_time(time.time())
                start_pos = len(seq.get_input_tokens()) + len(seq.get_output_tokens()) - 1
                self._fill_schedule_output(output, seq, start_pos, 1, [seq.get_output_tokens()[-1]])
            return output

        #Enter Prefill Period
        sequence_group = self.WAITING_QUEUE[0]
        self.WAITING_QUEUE.popleft()
        self.RUNNING_QUEUE.append(sequence_group)

        #chunked prefill
        chunk = None
        if self.check_if_chunk(sequence_group):
            chunk = self.chunk(sequence_group)
            sequence_group.get_sequences()[0].set_chunk(chunk)

        for seq in sequence_group.get_sequences():
            seq.set_first_schedule_time(time.time())
            if chunk is not None:
                start_pos = seq.get_chunk_index() * self.chunk_block
                token_ids = chunk[seq.get_chunk_index()]
            else:
                start_pos = 0
                token_ids = seq.get_input_tokens()
            self._fill_schedule_output(output, seq, start_pos, len(token_ids), token_ids)
        return output

    def _fill_schedule_output(self, output: ScheduleOutput, seq: Sequence, start_pos: int, num_new: int, token_ids: list):
        """
        向 BlockManager 确保账本充足,并填充 block_table / slot_mapping / token_ids
        """
        sequence_id = seq.get_sequence_id()
        self.block_manager.ensure_blocks(sequence_id, start_pos + num_new)
        output.block_table[sequence_id] = self.block_manager.get_allocated_blocks(sequence_id)
        output.slot_mapping.extend(self.block_manager.get_slot_mapping(sequence_id, start_pos, num_new))
        output.token_ids.extend(token_ids)
        output.seqs.append(seq)

        
    def add_sequence_group(self,sequence_group:SequenceGroup):
        #check status
        if sequence_group.get_status() == SeqGroupStatus.WAITING:
            self.WAITING_QUEUE.append(sequence_group)
        else:
            raise ValueError("SequenceGroup status is not WAITING")
        
    def check_remaining(self):
        if len(self.WAITING_QUEUE) > 0 or len(self.RUNNING_QUEUE) > 0:
            return True
        else:
            return False
    
    #Update the status of the request based on the result of executor
    def update(self,execute_output:ExecuteOutput = None):

        seqs = execute_output.get_seqs()
        output_tokens = execute_output.get_output_tokens()
        seq = seqs[0]

        #prefill update
        if seq.is_prefilling:
            if seq.get_chunk() is None:
                for seq in seqs:
                    seq.update_computed_tokens(len(seq.get_input_tokens()))
                    seq.set_output_tokens([output_tokens[0]])


            else:
                chunk_size = self.chunk_block
                chunk_index = seq.get_chunk_index()
                if chunk_index == len(seq.input_tokenIDs) // chunk_size: #此时是已经产生新的token
                    for seq in seqs:
                        seq.update_computed_tokens(len(seq.get_input_tokens())-chunk_size*chunk_index)
                        seq.set_output_tokens([output_tokens[0]])
                else: #此时仍然处于prefill阶段
                    for seq in seqs:
                        seq.update_computed_tokens((chunk_index+1)*chunk_size)
        
        #decode update
        else:
            for seq in seqs:
                seq.add_output_token(output_tokens[seqs.index(seq)])

                if self._is_finished(seq) == True:
                    sequence_group = seq.get_sequence_group()
                    check = True
                    for member in sequence_group.get_sequences():
                        if not self._is_finished(member):
                            check = False
                    if check and sequence_group.get_status() != SeqGroupStatus.FINISHED:
                        sequence_group.set_status(SeqGroupStatus.FINISHED)
                        self.RUNNING_QUEUE.remove(sequence_group)
                        self.FINISHED_QUEUE.append(sequence_group)
                        for member in sequence_group.get_sequences():
                            self.block_manager.free_sequence(member.get_sequence_id())
        return

        

    def _is_finished(self,sequence:Sequence):
        if sequence.check_finished() == True:
            return True
        max_tokens = sequence.get_max_tokens()
        output_tokenIDs = sequence.get_output_tokens()
        curr_tokens = len(output_tokenIDs)
        if curr_tokens >= max_tokens or output_tokenIDs[-1] == sequence.get_eos_token_id():
            sequence.set_finished()
            return sequence.check_finished()
        else:
            return False
    
    def remove_sequence_group(self,sequence_group:SequenceGroup):
        self.FINISHED_QUEUE.remove(sequence_group)

    def check_if_chunk(self,sequence_group:SequenceGroup):
        example_seq = sequence_group.get_sequences()[0]
        seq_size = len(example_seq.get_input_tokens())
        return seq_size > self.chunk_block
    
    def chunk(self,sequence_group:SequenceGroup):
        example_seq = sequence_group.get_sequences()[0]
        seq_size = len(example_seq.get_input_tokens())
        input_tokens = example_seq.get_input_tokens()
        chunk = []
        cnt = 0
        while cnt < seq_size:
            chunk.append(input_tokens[cnt:cnt+self.chunk_block])
            cnt += self.chunk_block
        return chunk
