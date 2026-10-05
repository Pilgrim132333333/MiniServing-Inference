from miniserving.Engine.SequenceGroup import SequenceGroup,Sequence,SeqGroupStatus
from miniserving.Backend.BackendFactory import BackendFactory
from miniserving.Backend.ExecuteOutput import ExecuteOutput
from collections import deque
from transformers import DynamicCache
import copy
import time
from miniserving.configs.SchedulerConfig import SchedulerConfig
from miniserving.configs.EngineConfig import EngineConfig



class Scheduler:
    def __init__(self,config):
        self.WAITING_QUEUE = deque()
        self.RUNNING_QUEUE = deque()
        self.FINISHED_QUEUE = deque()
        self.FAILED_QUEUE = deque()

        self.config = config.scheduler_config
        self.batch_size = self.config.batch_size
        self.chunk_block = self.config.chunk_block
    
    def schedule(self):
        if not self.check_remaining():
            return None
        batch = []
        while len(self.RUNNING_QUEUE) > 0 and len(batch) < self.batch_size:
            sequence_group = self.RUNNING_QUEUE[0]
            for seq in sequence_group.get_sequences():
                batch.append(seq)
                seq.set_first_schedule_time(time.time())
            return batch
            
        #Enter Prefill Period
        sequence_group = self.WAITING_QUEUE[0]
        self.WAITING_QUEUE.popleft()
        self.RUNNING_QUEUE.append(sequence_group)

        #chunked prefill
        if self.check_if_chunk(sequence_group):
            chunk = self.chunk(sequence_group)
            sequence_group[0].set_chunk(chunk)

        for seq in sequence_group.get_sequences():
            batch.append(seq)
            seq.set_first_schedule_time(time.time())
        return batch

        
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
        past_key_values = execute_output.get_output_key_values()
        output_tokens = execute_output.get_output_tokens()
        seq = seqs[0]

        #prefill update
        if seq.is_prefilling:
            if execute_output.chunk_index is None:
                for seq in seqs:
                    seq.update_computed_tokens(len(seq.get_input_tokens()))
                    seq.set_output_tokens([output_tokens[0]])
                    new_key_values = self._extract_seq_KV(0,past_key_values[0])
                    seq.set_past_key_values(new_key_values)
            else:
                chunk_size = self.chunk_block
                chunk_index = seq.get_chunk_index()
                if chunk_index == len(seq.input_tokenIDs) // chunk_size: #此时是已经产生新的token
                    for seq in seqs:
                        seq.update_computed_tokens(len(seq.get_input_tokens())-chunk_size*chunk_index)
                        seq.set_output_tokens([output_tokens[0]])
                        new_key_values = self._extract_seq_KV(chunk_index,past_key_values[chunk_index])
                        seq.set_past_key_values(new_key_values)
                else: #此时仍然处于prefill阶段
                    for seq in seqs:
                        seq.update_computed_tokens((chunk_index+1)*chunk_size)
                        new_key_values = self._extract_seq_KV(chunk_index,past_key_values[chunk_index])
                        seq.set_past_key_values(new_key_values)
        
        #decode update
        else:
            for seq in seqs:
                seq.add_output_token(output_tokens[seqs.index(seq)])
                new_key_values = past_key_values[seqs.index(seq)]
                seq.set_past_key_values(new_key_values)

                if self._is_finished(seq) == True:
                    sequence_group = seq.get_sequence_group()
                    check = True
                    for seq in sequence_group.get_sequences():
                        if not self._is_finished(seq):
                            check = False
                    if check:
                        sequence_group.set_status(SeqGroupStatus.FINISHED)
                        self.RUNNING_QUEUE.remove(sequence_group)
                        self.FINISHED_QUEUE.append(sequence_group)
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

    def _extract_seq_KV(self,batchIndex:int,past_key_values):
        new_cache = DynamicCache()
        # transformers>=5.x: DynamicCache 内部是 layers 列表, 每层是 DynamicLayer
        # past_key_values.layers[layer_idx].keys/.values shape: [B, heads, seq, dim]
        for layer_idx, layer in enumerate(past_key_values.layers):
            K = layer.keys
            V = layer.values
            K_b = K[batchIndex:batchIndex+1]
            V_b = V[batchIndex:batchIndex+1]
            new_cache.update(K_b, V_b, layer_idx)
        
        return new_cache
    
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
