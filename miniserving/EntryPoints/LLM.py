from miniserving.Engine.EngineCore import EngineCore
from miniserving.configs.EngineConfig import engineConfig
class LLM:
    def __init__(self,LLM_name:str,):
        self.model_name = LLM_name
        self.engineCore = EngineCore(engineConfig)
    
    #现在仅支持single prompt/ single request
    def generate(self,prompt:str,sampling_params:dict = None):
        if prompt is None:
            raise ValueError("prompt is empty")

        self._add_sequence_group(prompt,sampling_params)
        
        while self.engineCore.is_running():
            self.engineCore.step()
        
        output = self.engineCore.get_output()
        #check output logic
        return output
    
    def _add_sequence_group(self,prompt:str,sampling_params:dict = None):
        self.engineCore.add_sequence_group(prompt,sampling_params)
    ##setter and getter