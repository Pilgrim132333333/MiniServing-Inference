import logging

import torch
from miniserving.worker.baseworker import BaseWorker
from miniserving.worker.ExecuteOutput import ExecuteOutput
from miniserving.Backend.kvCache.KVCacheManager import KVCacheManager
from miniserving.Backend.model.model_register import model_register
from miniserving.configs.WorkerConfig import WorkerConfig
from miniserving.Engine.SequenceGroup import Sequence
from miniserving.ModelRunner.ModelRunner import ModelRunner
from miniserving.Scheduler.ScheduleOutput import ScheduleOutput


class GPUWorker(BaseWorker):
    def __init__(self,config:WorkerConfig):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.KVCacheManager = KVCacheManager(config.kvcachemanager_config)
        self.model = None
        self.modelRunner = None

    def load_model(self):
        model_name = self.model_name
        self.model = model_register[model_name]()
        self.modelRunner = ModelRunner(self.model,self.config.blockmanager_config)
    
    def execute(self,schedule_output: ScheduleOutput):
        if self.model is None:
            self.load_model()
        batch = schedule_output.seqs
        block_table = schedule_output.block_table
        slot_mapping = schedule_output.slot_mapping
        start_positions = schedule_output.start_positions
        output_tokens = []
        for idx,seq in enumerate(batch):
            output_tokens.extend(seq.get_output_tokens())
            seq_blocks = block_table[seq.get_sequence_id()]
            start_pos = start_positions[idx]
            if output_tokens  == []:
                execute_output = self.execute_prefill(batch,seq_blocks,slot_mapping,start_pos)   
            else:
                execute_output = self.execute_decode(batch,seq_blocks,slot_mapping,start_pos)
        return execute_output

    def execute_decode(self,batch: list[Sequence],seq_blocks: list[int],slot_mapping: list[int],start_pos: int):   
        next_token_ids = []
        for seq in batch:
            input_tokens = [[seq.get_output_tokens()[-1]]]
            input_tensors = torch.tensor(input_tokens)

            #一次decode
            # output:[batch, seq_len, vocab_size]
            output = self.modelRunner.forward(input_tensors,seq_blocks,slot_mapping,start_pos)
            
            #这里先试用greedy decode
            for logit in output:
                next_token_ids.append(torch.argmax(logit[-1,:]).item())
        execute_output = ExecuteOutput(batch,next_token_ids)

        return execute_output
        
    def execute_prefill(self,batch: list[Sequence],seq_blocks: list[int],slot_mapping: list[int],start_pos: int):
        """
        执行prefill阶段
        """
        seq = batch[0]
    
        #check if chunked
        if seq.get_chunk() is not None:
            input_tensors = torch.tensor([seq.get_chunk()[seq.get_chunk_index()]])
        
        else:
            input_tokens = seq.get_input_tokens()
            input_tensors = torch.tensor([input_tokens])
            
        #一次prefill
        #output = [batch, seq_len, vocab_size]
        output = self.modelRunner.forward(input_tensors,seq_blocks,slot_mapping,start_pos)
        
        """
        past_key_values 的形状： [batch_size, num_kv_heads, seq_len, head_dim]
        """

        #这里先试用greedy decode
        next_token_ids = torch.argmax(output[-1,-1:]).item()
            

        execute_output = ExecuteOutput(batch,[next_token_ids])
        return execute_output