from miniserving.ModelRunner.GPT2ForCausalLM import GPT2ForCausalLM


class ModelRunner:
    def __init__(self,model,config):
        if model is None:
            raise ValueError("Model is None")
        else:
            self.model = GPT2ForCausalLM(model,config)

    def forward(self,inputs,block_table: dict[str,list[int]],slot: list[int]):
        return self.model.forward(inputs, block_table, slot)
