from miniserving.Engine.EngineCore import EngineCore
class LLM:
    def __init__(self,LLM_name:str,):
        self.model_name = LLM_name
        self.engineCore = EngineCore(self.model_name)
    
    #现在仅支持single prompt/ single request
    def generate(self,request,sampling_params:dict = None):
        if isinstance(request,Request):
            prompt = request.get_prompt()
            sampling_params = request.get_sampling_params()
        else:
            prompt = request

        self._add_sequence_group(prompt,sampling_params)
        
        while self.engineCore.is_running():
            self.engineCore.step()
        
        output = self.engineCore.get_output()
        #check output logic
        return output
    
    def _add_sequence_group(self,prompt:str,sampling_params:dict = None):
        self.engineCore.add_sequence_group(prompt,sampling_params)
    ##setter and getter