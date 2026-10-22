import logging
import time
import uuid

from miniserving.worker.workerFactory import WorkerFactory
from miniserving.configs.EngineConfig import EngineConfig
from miniserving.Engine.SequenceGroup import Sequence, SequenceGroup
from miniserving.Observability.metrics import (
    BATCH_SIZE,
    KV_CACHE_USED,
    REQUEST_CLIENT_TTFT,
    REQUEST_SERVER_TTFT,
    REQUEST_TPOT,
    RUNNING_REQUESTS,
    STEP_TIME,
    TOTAL_REQUESTS,
    TOTAL_TOKENS,
    WAITING_REQUESTS,
)
from miniserving.Scheduler.scheduler import Scheduler
from transformers import AutoTokenizer


class EngineCore:
    def __init__(self,config:EngineConfig):
        self.model_name = config.model
        self.worker = WorkerFactory(self.model_name).instance(config)
        self.outputs = []
        self.scheduler = Scheduler(config)
        self.output = []
        self.logger = logging.getLogger(__name__)

        try:
            self.logger.info(f"尝试初始化Tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        except:
            self.logger.error(f"The tokenizer init fails. model_name={self.model_name}, worker_type={config.worker_type},time = {time.time()}")



        self.logger.info(f"EngineCore inited. model_name={self.model_name}, worker_type={config.worker_type},time = {time.time()}")

        

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
            sequence.set_eos_token_id(eos_token_id)
            sequence_group.add_sequence(sequence)
    
        self.scheduler.add_sequence_group(sequence_group)
        self.logger.info(f"Request:%d's sequence group inited.",request_id)

    def step(self):
        #从 Scheduler 拿到本步调度输出(含 block_table/slot_mapping 账本增量)
        schedule_output = self.scheduler.schedule()

        if schedule_output is None:
            return None
        execute_output = self.worker.execute(schedule_output)
    
        self.scheduler.update(execute_output)
        
        finished_sequences = self.scheduler.FINISHED_QUEUE
        while finished_sequences:
            sequence_group = finished_sequences.popleft()
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
