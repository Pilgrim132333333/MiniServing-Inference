from Engine.Request import Request
class ExecuteOutput:
    def __init__(self,requests:list[Request],output_tokens:list = [],past_key_values:tuple = None):
        self.requests = requests
        self.output_tokens = output_tokens
        self.past_key_values = past_key_values

    def get_requests(self):
        return self.requests
    def get_output_tokens(self):
        return self.output_tokens
    def get_past_key_values(self):
        return self.past_key_values