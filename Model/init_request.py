import uuid
class init_request:
    def __init__(self,prompt:str,model_name:str):
        self.request_id = str(uuid.uuid4())
        self.prompt = prompt
        self.model_name = model_name
    def get_request_id(self):
        return self.request_id
    def get_prompt(self):
        return self.prompt
    def get_model_name(self):
        return self.model_name