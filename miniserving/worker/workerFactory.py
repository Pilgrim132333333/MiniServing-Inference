from miniserving.worker.gpuworker import GPUWorker
from miniserving.worker.baseworker import BaseWorker


class WorkerFactory:
    def __init__(self,model_name:str):
        self.impl = None
        self.model_name = model_name
    
    def instance(self,config):
        worker_config = config.worker_config
        worker_type = worker_config.worker_type

        if worker_type == "torch":
            return GPUWorker(worker_config)
        else:
            raise ValueError(f"Unknown worker type: {worker_type}")

