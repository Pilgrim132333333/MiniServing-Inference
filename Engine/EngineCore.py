from transformers import AutoTokenizer
from Engine.Request import Request
from Backend.BackendFactory import BackendFactory
from Scheduler.scheduler import Scheduler
from collections import deque
import uuid


class EngineCore:
    def __init__(self,model_name:str, backend_type:str = "torch"):
        self.model_name = model_name
        self.backend = BackendFactory(model_name).instance(backend_type)
        self.request_queue = deque()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.outputs = []
        self.scheduler = Scheduler()


    def add_request(self,prompt:str,sampling_params:dict = None) -> Request:
        request_id = int(uuid.uuid4().hex,16)
        tokenIDs = self._encode_prompt(prompt)
        request = Request(request_id,prompt,tokenIDs,sampling_params)
        
        self.scheduler.add_request(request)

    def step(self):
        #从request_queue中取出一个request
        request = self.scheduler.schedule()

        if request is None:
            return None
        self.backend.generate(request)

        self.scheduler.update()
        
        
        #在Request被backend处理完以后：
        output_tokens = request.get_output_token()
        output_prompt = self.tokenizer.decode(output_tokens)
        self.outputs.append(output_prompt)
    
    def is_running(self) -> bool:
        remaining = self.scheduler.check_remaining()
        return remaining

    
    
    def _encode_prompt(self,prompt:str) -> list:
        return AutoTokenizer.from_pretrained(self.model_name).encode(prompt)
    
    #-----------------------------------#
    #setter and getter
    
    def get_output(self) -> list:
        return self.outputs