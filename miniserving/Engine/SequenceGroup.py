from enum import Enum

class Sequence:
    def __init__(self,tokenIDs:list):
        self.tokenIDs = tokenIDs
        self.output_tokenIDs= []
        self.is_finished = False
        self.eos_token_id = None
        self.past_key_values = None
        self.max_tokens = 20

    def check_finished(self):
        return self.is_finished
    def get_output_tokens(self):
        return self.output_tokenIDs
    def set_output_tokens(self,tokenIDs:list):
        self.output_tokenIDs = tokenIDs
        return
    def get_input_tokens(self):
        return self.tokenIDs
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
    def get_eos_token_id(self):
        return self.eos_token_id
    def set_eos_token_id(self,eos_token_id:int):
        self.eos_token_id = eos_token_id
        return
    
class SeqGroupStatus(Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2
    FAILED = 3
class SequenceGroup:
    def __init__(self,request_id:int,prompt:str,sequence:list[Sequence],sampling_params:dict = None,max_tokens:int = 20,eos_token_id:int = None):
        self.request_id = request_id
        self.prompt = prompt
        self.sequence = sequence
        self.output_tokenIDs= []
        self.sampling_params = sampling_params
        self.status = SeqGroupStatus.WAITING
        self.eos_token_id = eos_token_id
        self.past_key_values = None
        self.max_tokens = max_tokens
    
    def get_sequences(self):
        return self.sequence

    def set_output_prompt(self,output_prompt:str):
        self.output_prompt = output_prompt
        return
    def get_output_prompt(self):
        return self.output_prompt

    def get_status(self):
        return self.status
    def set_status(self,status:SeqGroupStatus):
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
    
    