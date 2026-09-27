from fastapi import FastAPI
from pydantic import BaseModel
from Model.LLM import LLM
from server.Request import Request

app = FastAPI()
llm = LLM("gpt2")


@app.post("/generate")
def generate(request: Request):
    output = llm.generate(request)
    return output