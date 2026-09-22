from transformers import AutoTokenizer
from Engine.Request import Request
from Backend.BackendFactory import BackendFactory
from collections import deque
import uuid


class EngineCore:
    def __init__(self,model_name:str, backend_type:str = "torch"):
        self.model_name = model_name
        self.backend = BackendFactory(model_name).instance(backend_type)
        self.request_queue = deque()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.outputs = []


    def add_request(self,prompt:str,sampling_params:dict = None) -> Request:
        request_id = int(uuid.uuid4().hex,16)
        tokenIDs = self._encode_prompt(prompt)
        request = Request(request_id,prompt,tokenIDs,sampling_params)
        
        self.request_queue.append(request)
    
    def step(self):
        #从request_queue中取出一个request
        requests = []
        requests.append(self.request_queue[0])
        for request in requests:
            self.backend.generate(request)
        
        #在Request被backend处理完以后：
        for request in requests:
            output_tokens = request.get_output_token()
            output_prompt = self.tokenizer.decode(output_tokens)
            self.outputs.append(output_prompt)
            self.request_queue.popleft()
    
    def is_running(self) -> bool:
        return len(self.request_queue) > 0
    
    
    def _encode_prompt(self,prompt:str) -> list:
        return AutoTokenizer.from_pretrained(self.model_name).encode(prompt)
    
    #-----------------------------------#
    #setter and getter
    def get_queue_top(self) -> Request:
        return self.request_queue[0]
    def get_queue_size(self) -> int:
        return len(self.request_queue)
    
    def get_output(self) -> list:
        return self.outputs