from abc import ABC, abstractmethod
from miniserving.Engine.SequenceGroup import SequenceGroup,Sequence
from miniserving.Backend.model.model_register import model_register
class BaseBackend(ABC):
    def __init__(self,model_name:str):
        self.model_name = model_name
        self.model = None
    
    @abstractmethod
    def load_model(self):
        self.model = model_register.get_model(self.model_name)
        return
    
    @abstractmethod
    def execute(self,request:Sequence):
        pass