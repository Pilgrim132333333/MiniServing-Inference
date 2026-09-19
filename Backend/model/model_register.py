import torch
from transformers import AutoModelForCausalLM
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
    
    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.model = model
        return
    
    def default_generate(self,input_token:list) :
        input_ids = torch.tensor([input_token])
        output = self.model.generate(input_ids)
        return output