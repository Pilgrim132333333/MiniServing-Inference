
from Model.LLM import LLM

prompt = "你好"
llm = LLM("GPT2")
output = llm.generate(prompt)
print(output)
