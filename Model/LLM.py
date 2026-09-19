from Model.init_request import init_request
from Engine.EngineCore import EngineCore
class LLM:
    def __init__(self,LLM_name:str,):
        self.model_name = LLM_name
        self.request_queue = {}
        self.engineCore = EngineCore(self.model_name)
    def generate(self,prompt:str):
        request = self.encapsulate_request(prompt)
        output = self.generate_output(request)
        #check output logic
        return output
    def encapsulate_request(self,prompt:str):
        request = init_request(prompt,self.model_name)
        self.add_request(request)
        return request
    def add_request(self,request):
        self.request_queue[request.get_request_id()] = request
    def generate_output(self,request:init_request):
        output = self.engineCore.generate(request).get_output_prompt()
        return output
    ##setter and getter