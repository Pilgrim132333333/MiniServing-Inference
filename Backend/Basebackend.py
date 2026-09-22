from abc import ABC, abstractmethod
from Engine.Request import Request
from Backend.model.model_register import model_register
class BaseBackend(ABC):
    def __init__(self,model_name:str):
        self.model_name = model_name
        self.model = None
    @abstractmethod
    def generate(self,request:Request) -> list:
        pass# 返回token list
    
    @abstractmethod
    def load_model(self):
        pass
