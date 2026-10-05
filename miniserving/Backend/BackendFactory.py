import torch
from miniserving.Backend.TorchBackend import TorchBackend
from miniserving.Backend.LlamaBackend import LlamaBackend
from miniserving.Backend.Basebackend import BaseBackend
from miniserving.configs.BackendConfig import BackendConfig
from miniserving.configs.EngineConfig import EngineConfig



class BackendFactory:
    def __init__(self,model_name:str):
        self.impl = None
        self.model_name = model_name
    
    def instance(self,config):
        backend_config = config.backend_config
        backend_type = backend_config.backend_type

        if backend_type == "torch":
            return TorchBackend(backend_config)
        elif backend_type == "llama":
            return LlamaBackend(backend_config)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

