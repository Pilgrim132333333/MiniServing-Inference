
from transformers import AutoTokenizer
from Backend.backend import Backend
from Model.init_request import init_request
class Request:
    def __init__(self,request_id:int,prompt:str,token:list,model_name:str):
        self.request_id = request_id
        self.prompt = prompt
        self.token = token
        self.output_token = None
        self.model_name = model_name
        self.backend = Backend(model_name)
    
    def get_input_token(self):
        return self.token
    def set_input_token(self,token:list):
        self.token = token
        return
    def get_output_token(self):
        return self.output_token
    def set_out_put_token(self,output_token:list):
        self.output_token = output_token
        return
    def set_output_prompt(self,output_prompt:str):
        self.output_prompt = output_prompt
        return
    def get_output_prompt(self):
        return self.output_prompt

class EngineCore:
    def __init__(self,model_name:str):
        self.model_name = model_name
        self.backend = Backend(model_name)

    
    def addRequest(self,init_request:init_request) -> Request:
        request_id = init_request.get_request_id()
        prompt = init_request.get_prompt()
        request = Request(request_id,prompt,AutoTokenizer.from_pretrained(self.model_name).encode(prompt),self.model_name)
        return request
    def generate(self,init_request:init_request) -> Request:
        request = self.addRequest(init_request)
        out_put_tensor = self.backend.generate(request)
        out_put_token = out_put_tensor.tolist()
        request.set_output_prompt(AutoTokenizer.from_pretrained(self.model_name).decode(out_put_token))
        return request