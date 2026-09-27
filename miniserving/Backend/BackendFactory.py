import torch
from miniserving.Backend.TorchBackend import TorchBackend
from miniserving.Backend.LlamaBackend import LlamaBackend
from miniserving.Backend.Basebackend import BaseBackend


class BackendFactory:
    def __init__(self,model_name:str):
        self.impl = None
        self.model_name = model_name
    
    def instance(self,backend_type:str = "torch"):
        if backend_type == "torch":
            return TorchBackend(self.model_name)
        elif backend_type == "llama":
            return LlamaBackend(self.model_name)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

