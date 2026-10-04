from enum import Enum
import uuid
from miniserving.Backend.kvCache.kvCache import kvCache

class Sequence:
    def __init__(self,tokenIDs:list):
        self.input_tokenIDs = tokenIDs
        self.sequence_id = str(uuid.uuid4())
        self.output_tokenIDs= []
        self.is_finished = False
        self.eos_token_id = None
        self.computed_tokens = 0
        self.past_key_values = None
        self.sequence_group = None
        self.max_tokens = self.sequence_group.max_tokens if self.sequence_group else 20
        self.is_finished = False
        self.sequence_id = str(uuid.uuid4())
        self.chunk = None
        self.chunk_index = None

    def check_finished(self):
        return self.is_finished

    def get_output_tokens(self):
        return self.output_tokenIDs
    def set_output_tokens(self,tokenIDs:list):
        self.output_tokenIDs = tokenIDs
        return
    def add_output_token(self,tokenID:int):
        self.output_tokenIDs.append(tokenID)
        return
    
    def get_input_tokens(self):
        return self.input_tokenIDs
    def set_input_tokens(self,tokenIDs:list):
        self.input_tokenIDs = tokenIDs
        return

    def get_past_key_values(self):
        return self.past_key_values
    def set_past_key_values(self,past_key_values:kvCache):
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
    
    def check_finished(self):
        return self.is_finished
    def set_finished(self):
        self.is_finished = True
        return
    
    def get_sequence_group(self):
        return self.sequence_group
    
    def set_output_prompt(self,prompt:str):
        self.output_prompt = prompt
        return
    def get_output_prompt(self):
        return self.output_prompt

    @property
    def is_prefilling(self):
        return self.computed_tokens < self.input_tokenIDs
    
    def update_computed_tokens(self,num_tokens:int):
        self.computed_tokens += num_tokens
        self.chunk_index += 1
        return
    
    def set_chunk(self,chunk:[]):
        self.chunk = chunk
        self.chunk_index = 0
        return
    def get_chunk(self):
        return self.chunk
        
    def get_chunk_index(self):
        return self.chunk_index
    
class SeqGroupStatus(Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2
    FAILED = 3
class SequenceGroup:
    def __init__(self,request_id:int,prompt:str,sequences:list[Sequence],sampling_params:dict = None,eos_token_id:int = None):
        self.request_id = request_id
        self.prompt = prompt
        self.sequences = sequences
        self.sampling_params = sampling_params
        self.status = SeqGroupStatus.WAITING
        self.eos_token_id = eos_token_id
        if sampling_params:
            self.max_tokens = sampling_params["max_tokens"]
        else:
            self.max_tokens = 20

        for sequence in sequences:
            sequence.sequence_group = self
    
    def get_sequences(self):
        return self.sequences
    def add_sequence(self,sequence:Sequence):
        sequence.sequence_group = self
        self.sequences.append(sequence)

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
    
    def get_max_tokens(self):
        return self.max_tokens
    def set_max_tokens(self,max_tokens:int):
        self.max_tokens = max_tokens
        return
    
    