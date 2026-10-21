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
        self.modelRunner = ModelRunner(self.model)
    
    def execute(self,schedule_output: ScheduleOutput):
        if self.model is None:
            self.load_model()
        batch = schedule_output.seqs
        output_tokens = []
        for seq in batch:
            output_tokens.extend(seq.get_output_tokens())
            if output_tokens  == []:
                execute_output = self.execute_prefill(batch)
            else:
                execute_output = self.execute_decode(batch)
        return execute_output

    def execute_decode(self,batch: list[Sequence]):   
        next_token_ids = []
        output_key_values = []
        for seq in batch:
            input_tokens = [[seq.get_output_tokens()[-1]]]
            past_key_values = seq.get_past_key_values()
            input_tensors = torch.tensor(input_tokens)

            #一次decode
            output = self.model.model(input_tensors,use_cache=True,past_key_values=past_key_values)
            out_put_logits = output.logits
            out_put_pasts = output.past_key_values
            output_key_values.append(out_put_pasts)

            #这里先试用greedy decode
            next_token_ids = []
            for logit in out_put_logits:
                next_token_ids.append(torch.argmax(logit[-1,:]).item())
        execute_output = ExecuteOutput(batch,next_token_ids,output_key_values)

        return execute_output
        
    def execute_prefill(self,batch: list[Sequence]):
        seq = batch[0]
    
        #check if chunked
        if seq.get_chunk() is not None:
            input_tensors = torch.tensor([seq.get_chunk()[seq.get_chunk_index()]])
        
        else:
            input_tokens = seq.get_input_tokens()
            input_tensors = torch.tensor([input_tokens])
            
        #一次prefill
        output = self.model.model(input_tensors,use_cache=True)
        out_put_logits = output.logits
        out_put_pasts = output.past_key_values
        """
        past_key_values 的形状： [batch_size, num_kv_heads, seq_len, head_dim]
        """

        #这里先试用greedy decode
        next_token_ids = torch.argmax(out_put_logits[-1,:]).item()
            

        execute_output = ExecuteOutput(batch,[next_token_ids],[out_put_pasts])
        return execute_output