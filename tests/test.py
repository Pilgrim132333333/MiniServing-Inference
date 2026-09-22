
from Model.LLM import LLM
import time

prompt = "Hello"
llm = LLM("GPT2")

start = time.time()
print(f"Start generate at {start}")

output = llm.generate(prompt)

end = time.time()
print(f"End generate at {end}")
print(f"Cost: {end - start:.2f}s")
print(output)