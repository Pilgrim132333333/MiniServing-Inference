from miniserving.Engine.SequenceGroup import SequenceGroup,Sequence
from miniserving.Backend.Basebackend import BaseBackend
from miniserving.Backend.BackendFactory import BackendFactory
from miniserving.EntryPoints.Request import Request
from miniserving.Scheduler.Scheduler import Scheduler
from transformers import AutoTokenizer
from collections import deque
import asyncio
class AsyncEngineCore:
    def __init__(self,model_name:str):
        self.model_name = model_name
        self.backend = BackendFactory(model_name).instance()
        self.scheduler = Scheduler()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.message_queue = {} #request_id -> Future
            
    def start(self):
        self.task = asyncio.create_task(self._run())
    
    def stop(self):
        self.task.cancel()
        self.backend.close_model()
    
    def load_model(self):
        self.backend.load_model()
    
    async def add_request(self,request:Request):
        self.message_queue[request.request_id] = asyncio.Future()
        prompt = request.prompt
        sampling_params = request.sampling_params
        request_id = request.request_id
        self.add_sequenceGroup(prompt,sampling_params,request_id)
        return self.message_queue[request_id]

    def add_sequenceGroup(self,prompt:str,sampling_params:dict,request_id:str):
        eos_token_id = self.tokenizer.eos_token_id
        if sampling_params is None:
            n = 1
        else:
            n = sampling_params["n"]
        sequenceGroup = SequenceGroup(prompt,sampling_params,request_id,eos_token_id)
        for i in range(n):
            sequenceGroup.add_sequence(Sequence(self.tokenizer.encode(prompt)))
        
        self.scheduler.add_sequence_group(sequenceGroup)
        return sequenceGroup
    
    def step():
        #从request_queue中取出一个request
        batch = self.scheduler.schedule()

        if batch is None:
            return None

        execute_output = self.backend.execute(batch)
    
        self.scheduler.update(execute_output)
        
        return

    def _drain(self):
        finished_sequences = self.scheduler.FINISHED_QUEUE
        for sequence_group in finished_sequences:
            request_id = sequence_group.get_request_id()
            output_prompts = []
            print(sequence_group+"ada")
            for sequence in sequence_group.get_sequences():
                output_tokens = sequence.get_output_tokens()
                output_prompt = self.tokenizer.decode(output_tokens)
                sequence.set_output_prompt(output_prompt)
                output_prompts.append(output_prompt)
            self.message_queue[request_id].set_result(output_prompts)
            self.scheduler.remove_sequence_group(sequence_group)
    
    async def _run(self):
        while True:
            asyncio.to_thread(self.step)
            _drain()
            await asyncio.sleep(0.1)
            
