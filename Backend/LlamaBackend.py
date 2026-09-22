from Backend.Basebackend import BaseBackend
from Engine.Request import Request


class LlamaBackend(BaseBackend):
    def __init__(self,model_name:str):
        super().__init__(model_name)