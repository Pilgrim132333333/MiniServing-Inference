import uuid



class Request:
    def __init__(self,prompt:str,sampling_params:dict = None):
        self.prompt = prompt
        self.sampling_params = sampling_params
        self.request_id = uuid.uuid4().hex
    
    def get_prompt(self):
        return self.prompt
        
    def get_sampling_params(self):
        return self.sampling_params