from miniserving.ModelRunner.GPT2ForCausalLM import GPT2ForCausalLM
from miniserving.Engine.SequenceGroup import SequenceGroup,Sequence
from miniserving.configs.EngineConfig import EngineConfig
from miniserving.configs.BackendConfig import BackendConfig
class ModelRunner:
    def __init__(self,config,model):
        self.config = config

        if model is None:
            raise ValueError("Model is None")
        else:
            self.model = GPT2ForCausalLM(model)
        
    
    def forward(self,request:Sequence):
        return self.model(request)
        