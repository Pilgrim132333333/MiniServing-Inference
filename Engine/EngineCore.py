
from transformers import AutoTokenizer
from Model.init_request import init_request
from Engine.Request import Request

from Backend.BackendFactory import BackendFactory



class EngineCore:
    def __init__(self,model_name:str, backend_type:str = "torch"):
        self.model_name = model_name
        self.backend = BackendFactory(model_name).instance(backend_type)

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