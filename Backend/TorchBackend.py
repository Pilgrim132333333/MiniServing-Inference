from Backend.model.model_register import model_register
from Backend.Basebackend import BaseBackend
from Engine.Request import Request
class TorchBackend(BaseBackend):
    def __init__(self,model_name:str):
        super().__init__(model_name)
    def generate(self,request:Request):
        #传递给model
        if not self.model:
            self.load_model()
        input_token = self.get_request_input_token(request)
        output_token = self.model.generate(input_token)
        #这里要包装output
        request.set_out_put_token(output_token)

    def load_model(self):
        model_name = self.model_name
        self.model = model_register[model_name]()
        return
    
    def get_request_input_token(self,request):
        return request.get_input_token()
        