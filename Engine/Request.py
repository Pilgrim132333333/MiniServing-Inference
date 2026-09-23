from enum import Enum
class RequestStatus(Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2
    FAILED = 3
class Request:
    def __init__(self,request_id:int,prompt:str,tokenIDs:list,sampling_params:dict = None):
        self.request_id = request_id
        self.prompt = prompt
        self.tokenIDs = tokenIDs
        self.output_tokenIDs= None
        self.sampling_params = sampling_params
        self.status = RequestStatus.WAITING
    
    def get_input_token(self):
        return self.tokenIDs
    def set_input_token(self,tokenIDs:list):
        self.tokenIDs = tokenIDs
        return
    def get_output_token(self):
        return self.output_tokenIDs
    def set_out_put_token(self,output_tokenIDs:list):
        self.output_tokenIDs = output_tokenIDs
        return
    def set_output_prompt(self,output_prompt:str):
        self.output_promptIDs = AutoTokenizer.from_pretrained(self.model_name).decode(output_token_promptIDs)
        return
    def get_output_prompt(self):
        return self.output_promptIDs
    def get_status(self):
        return self.status
    def set_status(self,status:RequestStatus):
        self.status = status
        return