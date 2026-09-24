from Backend.model.model_register import model_register
from Backend.Basebackend import BaseBackend
from Engine.Request import Request
from Backend.ExecuteOutput import ExecuteOutput
import torch
class TorchBackend(BaseBackend):
    def __init__(self,model_name:str):
        super().__init__(model_name)
    def generate(self,request:Request):
        #传递给model
        if not self.model:
            self.load_model()
        max_tokens = request.get_max_tokens()
        input_token = self.get_request_input_token(request)
        output_token = self.model.generate(input_token,max_new_tokens=max_tokens)
        #这里要包装output
        request.set_out_put_token(output_token)

    def load_model(self):
        model_name = self.model_name
        self.model = model_register[model_name]()
        return
    
    def execute(self,requests:list[Request]):
        if not self.model:
            self.load_model()
        for req in requests:
            output_token = req.get_output_token()
            if output_token  == []:
                execute_output = self.execute_prefill(req)
            else:
                execute_output = self.execute_decode(req)
        return execute_output

    def execute_decode(self,request:Request):   
        input_tokens = []
        input_token = [request.get_output_token()[-1]]
        input_tokens.append(input_token)
        input_tensors = torch.tensor(input_tokens)
        past_key_values = request.get_past_key_values()

        #一次decode
        output = self.model.model(input_tensors,use_cache=True,past_key_values=past_key_values)
        out_put_logits = output.logits
        out_put_pasts = output.past_key_values

        #这里先试用greedy decode
        next_token_ids = []
        for logit in out_put_logits:
            next_token_ids.append(torch.argmax(logit[-1,:]).item())
        execute_output = ExecuteOutput([request],next_token_ids,out_put_pasts)
        return execute_output
        
    def execute_prefill(self,request:Request):
        input_tokens = []
        input_token = request.get_input_token()
        input_tokens.append(input_token)
        input_tensors = torch.tensor(input_tokens)

        #一次prefill
        output = self.model.model(input_tensors,use_cache=True)
        out_put_logits = output.logits
        out_put_pasts = output.past_key_values

        #这里先试用greedy decode
        next_token_ids = []
        for logit in out_put_logits:
            next_token_ids.append(torch.argmax(logit[-1,:]).item())

        execute_output = ExecuteOutput([request],next_token_ids,out_put_pasts)
        return execute_output