from miniserving.Engine.SequenceGroup import SequenceGroup, Sequence
class ExecuteOutput:
    def __init__(self,seqs:list[Sequence],output_tokens:list = [],past_key_values = None):
        self.seqs = seqs
        self.output_tokens = output_tokens
        self.output_key_values = past_key_values

    def get_seqs(self):
        return self.seqs
    def get_output_tokens(self):
        return self.output_tokens
    def get_output_key_values(self):
        return self.output_key_values
