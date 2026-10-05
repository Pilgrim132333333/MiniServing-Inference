from transformers import AutoTokenizer
from miniserving.Engine.SequenceGroup import SequenceGroup,Sequence
from miniserving.Backend.BackendFactory import BackendFactory
from miniserving.Scheduler.scheduler import Scheduler
from miniserving.Observability.metrics import REQUEST_SERVER_TTFT,REQUEST_CLIENT_TTFT,REQUEST_TPOT,STEP_TIME,BATCH_SIZE,RUNNING_REQUESTS,WAITING_REQUESTS,KV_CACHE_USED,TOTAL_TOKENS,TOTAL_REQUESTS
from miniserving.utils.logger import logger
import time
import uuid
import logging
from miniserving.configs.EngineConfig import EngineConfig



class EngineCore:
    def __init__(self,config:EngineConfig):
        self.model_name = config.model
        self.backend = BackendFactory(self.model_name).instance(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.outputs = []
        self.scheduler = Scheduler(config)
        self.output = []
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"EngineCore inited. model_name={self.model_name}, backend_type={config.backend_type},time = {time.time()}")

        

    def add_sequence_group(self,prompt:str,sampling_params:dict = None) -> SequenceGroup:
        request_id = int(uuid.uuid4().hex,16)
        tokenID = self._encode_prompt(prompt)
        eos_token_id = self.tokenizer.eos_token_id
        if sampling_params is None:
            n = 1
        else:
            n = sampling_params.get_n()
        tokenIDs = [tokenID]*n
        seqs = [Sequence(tokenIDs[i]) for i in range(n)]
        sequence_group = SequenceGroup(request_id,prompt,[],sampling_params,eos_token_id)
        for sequence in seqs:
            sequence_group.add_sequence(sequence)
    
        self.scheduler.add_sequence_group(sequence_group)
        self.logger.info(f"Request:%d's sequence group inited.",request_id)

    def step(self):
        #从request_queue中取出一个request
        requests = self.scheduler.schedule()

        if requests is None:
            return None
        execute_output = self.backend.execute(requests)
    
        self.scheduler.update(execute_output)
        
        if self.scheduler.check_remaining():
            finished_sequences = self.scheduler.FINISHED_QUEUE
            for sequence_group in finished_sequences:
                print(sequence_group+"ada")
                for sequence in sequence_group.get_sequences():
                    output_tokens = sequence.get_output_tokens()
                    output_prompt = self.tokenizer.decode(output_tokens)
                    sequence.set_output_prompt(output_prompt)
                    self.output.append(output_prompt)
    
    def is_running(self) -> bool:
        remaining = self.scheduler.check_remaining()
        return remaining

    def _encode_prompt(self,prompt:str) -> list:
        return AutoTokenizer.from_pretrained(self.model_name).encode(prompt)
    
    #-----------------------------------#
    #setter and getter
    def get_output(self) -> list:
        return self.output
