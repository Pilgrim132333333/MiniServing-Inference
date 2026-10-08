from miniserving.Backend.model.model_register import model_register
from miniserving.Backend.Basebackend import BaseBackend
from miniserving.Engine.SequenceGroup import SequenceGroup,Sequence
from miniserving.Backend.ExecuteOutput import ExecuteOutput
from miniserving.Backend.kvCache.KvCache import KvCache
from miniserving.Backend.kvCache.BlockManager import BlockManager
from miniserving.configs.BackendConfig import BackendConfig
from miniserving.configs.BlockManagerConfig import BlockManagerConfig
import logging

import torch
class TorchBackend(BaseBackend):
    def __init__(self,config:BackendConfig):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.blockManager = BlockManager(config.blockmanager_config)
        self.model = None
        
    def load_model(self):
        model_name = self.model_name
        self.model = model_register[model_name]()
        return
    
    def execute(self,batch: list[Sequence]):
        if self.model is None:
            self.load_model()
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
        self._init_key_values(batch)
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

        #这里先试用greedy decode
        next_token_ids = torch.argmax(out_put_logits[-1,:]).item()
            

        execute_output = ExecuteOutput(batch,[next_token_ids],[out_put_pasts])
        return execute_output
    
    def _init_key_values(self,batch):
        for seq in batch:
            cache = KvCache(self.model_config)
            seq.set_past_key_values(cache)
        return