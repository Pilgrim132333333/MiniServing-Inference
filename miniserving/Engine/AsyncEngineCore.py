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
    
    def load_model(self):
        self.backend.load_model()
    
    async def add_request(self,request:Request):
        self.message_queue[request.request_id] = asyncio.Future()
        prompt = request.prompt
        sampling_params = request.sampling_params
        request_id = request.request_id
        self.add_sequenceGroup(prompt,sampling_params,request_id)
        return self.message_queue[request_id]

    def add_sequenceGroup(self, prompt: str, sampling_params: dict, request_id: str):
        eos_token_id = self.tokenizer.eos_token_id
        if sampling_params is None:
            sampling_params = {}
        n = sampling_params.get("n", 1)
        sequences = [Sequence(self.tokenizer.encode(prompt)) for _ in range(n)]
        sequenceGroup = SequenceGroup(
            request_id=request_id,
            prompt=prompt,
            sequences=sequences,
            sampling_params=sampling_params,
            eos_token_id=eos_token_id,
        )
        self.scheduler.add_sequence_group(sequenceGroup)
        return sequenceGroup
    
    def step(self):
        batch = self.scheduler.schedule()
        if batch is None:
            return None
        execute_output = self.backend.execute(batch)
        self.scheduler.update(execute_output)
        return

    def _drain(self):
        finished = list(self.scheduler.FINISHED_QUEUE)
        for sg in finished:
            request_id = sg.request_id
            output_prompts = []
            for seq in sg.get_sequences():
                output_tokens = seq.get_output_tokens()
                output_prompt = self.tokenizer.decode(output_tokens)
                seq.set_output_prompt(output_prompt)
                output_prompts.append(output_prompt)
            if request_id in self.message_queue:
                self.message_queue[request_id].set_result(output_prompts)
            self.scheduler.remove_sequence_group(sg)

    async def _run(self):
        while True:
            try:
                await asyncio.to_thread(self.step)
                self._drain()
            except Exception as e:
                print(f"[_run ERROR] {type(e).__name__}: {e}")
            await asyncio.sleep(0.1)
            
