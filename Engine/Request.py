from enum import Enum
class RequestStatus(Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2
    FAILED = 3
class Request:
    def __init__(self,request_id:int,prompt:str,tokenIDs:list,sampling_params:dict = None,max_tokens:int = 20,eos_token_id:int = None):
        self.request_id = request_id
        self.prompt = prompt
        self.tokenIDs = tokenIDs
        self.output_tokenIDs= []
        self.sampling_params = sampling_params
        self.status = RequestStatus.WAITING
        self.eos_token_id = eos_token_id
        self.past_key_values = None
        self.max_tokens = max_tokens
    
    def get_input_token(self):
        return self.tokenIDs
    def set_input_token(self,tokenIDs:list):
        self.tokenIDs = tokenIDs
        return
    def get_output_token(self):
        return self.output_tokenIDs
    def set_output_token(self,output_tokenIDs:list):
        self.output_tokenIDs = output_tokenIDs
        return
    def add_output_token(self,tokenIDs:list):
        self.output_tokenIDs.extend(tokenIDs)
        return

    def set_output_prompt(self,output_prompt:str):
        self.output_prompt = output_prompt
        return
    def get_output_prompt(self):
        return self.output_prompt

    def get_status(self):
        return self.status
    def set_status(self,status:RequestStatus):
        self.status = status
        return
    def get_eos_token_id(self):
        return self.eos_token_id
    def set_eos_token_id(self,eos_token_id:int):
        self.eos_token_id = eos_token_id
        return

    def add_output_token(self,tokenIDs:list):
        self.output_tokenIDs.extend(tokenIDs)
        return
    
    def get_past_key_values(self):
        return self.past_key_values
    def set_past_key_values(self,past_key_values:tuple):
        self.past_key_values = past_key_values
        return
    
    def get_max_tokens(self):
        return self.max_tokens
    def set_max_tokens(self,max_tokens:int):
        self.max_tokens = max_tokens
        return
    
    