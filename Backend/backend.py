from Backend.model.model_register import model_register


class Backend():
    def __init__(self,model_name:str):
        self.model_name = model_name
        self.model = None
    def generate(self,request):
        #传递给model
        if not self.model:
            self.load_model()
        input_token = self.get_request_input_token(request)
        output_token = self.model.default_generate(input_token)
        #这里要包装output
        request.set_out_put_token(output_token)
        return output_token

        return output
    def load_model(self):
        model_name = self.model_name
        self.model = model_register[model_name]()
        return
    
    def get_request_input_token(self,request):
        return request.get_input_token()
        