"""
一个维护instance的table，没有实际影响instance的业务逻辑
"""
from dataclasses import dataclass

@dataclass
class Instance:
    id: str
    url:str
    model:str
    healthy:bool = True
    last_heartbeat:float = 0.0

class InstanceTable:
    def __init__(self):
        self._instances = {}
    
    def list(self) -> list[Instance]:
        return list(self._instances.values())
    
    def get(self, instance_id: str) -> Instance:
        return self._instances[instance_id]
    
    def set_healthy(self, instance_id: str, healthy: bool):
        self._instances[instance_id].healthy = healthy
    
    def upsert(self, instance: Instance):
        self._instances[instance.id] = instance
    
    def remove(self, instance_id: str):
        del self._instances[instance_id]
