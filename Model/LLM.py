from Engine.EngineCore import EngineCore
class LLM:
    def __init__(self,LLM_name:str,):
        self.model_name = LLM_name
        self.engineCore = EngineCore(self.model_name)
    def generate(self,prompts:list,sampling_params:dict = None):
        for prompt in prompts:
            self._add_request(prompt,sampling_params)
        
        while self.engineCore.is_running():
            self.engineCore.step()
        
        output = self.engineCore.get_output()
        #check output logic
        return output
    
    
    def _add_request(self,prompt:str,sampling_params:dict = None):
        self.engineCore.add_request(prompt,sampling_params)
    ##setter and getter