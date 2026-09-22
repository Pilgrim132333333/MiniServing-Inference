import torch
from transformers import AutoModelForCausalLM
from typing import TYPE_CHECKING
from Engine.Request import Request
model_register = {}

def register_model(model_name:str):
    def decorator(cls):
        model_register[model_name] = cls
        return cls
    return decorator


@register_model("deepseek-ai/DeepSeek-V4.1-Flash")
class DeepSeekV41Flash:
    def __init__(self):
        self.load_model()
    
    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V4.1-Flash")
        self.model = model
        return  
    
    #返回token id
    def default_generate(self,input_token:list) :
        output = self.model.generate(input_token)
        return output
@register_model("GPT2")
class GPT2:
    def __init__(self):
        self.load_model()
        self.max_token = 20
        

    
    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.model = model
        return
    
    #默认连续调用prefill和decode循环
    #返回tensor
    def default_generate(self,input_token:list) :
        input_ids = torch.tensor([input_token])
        output = self.model.generate(input_ids)
        return output

    def generate(self,input_token:list) :
        input_tensors = torch.tensor([input_token])
        
        with torch.no_grad():
        #prefill 
            outputs = self.model(input_tensors,use_cache=True)
            past_key_values = outputs.past_key_values  
            logits = outputs.logits
            print(logits)
            next_token_id = torch.argmax(logits[0,-1,:]).item()

            #decode
            generate_token = [next_token_id]
            next_token = torch.tensor([[next_token_id]])
            count = 0
            eos_token_id = self.model.config.eos_token_id

            while(next_token_id != eos_token_id and count < self.max_token):
                new_outputs =  self.model(next_token,past_key_values=past_key_values,use_cache=True)
                past_key_values = new_outputs.past_key_values
                next_logits = new_outputs.logits[0,-1,:]
                next_token_id = torch.argmax(next_logits).item()
                generate_token.append(next_token_id)
                next_token = torch.tensor([[next_token_id]])
                count += 1
                if count % 5 == 0:
                    print(f"  decode step {count}/{self.max_token}, current token_id={next_token_id}")
        
        #将[token_id]转换为[tensor,token_id]
        output = torch.tensor(generate_token)
        return output