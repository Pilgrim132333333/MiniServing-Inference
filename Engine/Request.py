class Request:
    def __init__(self,request_id:int,prompt:str,token:list,model_name:str):
        self.request_id = request_id
        self.prompt = prompt
        self.token = token
        self.output_token = None
        self.model_name = model_name
    
    def get_input_token(self):
        return self.token
    def set_input_token(self,token:list):
        self.token = token
        return
    def get_output_token(self):
        return self.output_token
    def set_out_put_token(self,output_token:list):
        self.output_token = output_token
        return
    def set_output_prompt(self,output_prompt:str):
        self.output_prompt = output_prompt
        return
    def get_output_prompt(self):
        return self.output_prompt