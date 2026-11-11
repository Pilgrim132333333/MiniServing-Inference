from abc import ABC, abstractmethod
from miniserving.EntryPoints.Request import Request
from miniserving.Gateway.InstanceTable import Instance
"""
根据Request和backendlist里面选择backend
"""
class RouterPolicy(ABC):
    def __init__(self, config):
        self.config = config
    @abstractmethod
    def route(self, request:Request, backends: list[Instance]):
        pass

class RoundRobinRouterPolicy(RouterPolicy):
    def __init__(self, config):
        super().__init__(config)
        self._cursor = 0
    
    def route(self, request:Request, backends: list[Instance]) -> Instance:
        if not backends:
            raise ValueError("No instance available")
        
        candidates = [b for b in backends if b.healthy and b.model == request.model]
        if not candidates:
            raise ValueError("No healthy backend available for model")
        
        instance = candidates[self._cursor]
        self._cursor = (self._cursor + 1) % len(candidates)
        return instance
        