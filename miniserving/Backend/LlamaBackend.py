from miniserving.Backend.Basebackend import BaseBackend
from miniserving.Engine.SequenceGroup import SequenceGroup, Sequence


class LlamaBackend(BaseBackend):
    def __init__(self,model_name:str):
        super().__init__(model_name)