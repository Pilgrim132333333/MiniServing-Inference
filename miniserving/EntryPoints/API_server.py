from fastapi import FastAPI
from pydantic import BaseModel
from miniserving.EntryPoints.LLM import LLM
import requests

app = FastAPI()
llm = LLM("gpt2")


@app.post("/generate", response_model=None)
def generate(request: dict):
    print("Begin")
    output = llm.generate(request)
    return output

@app.post("/v1/completion")
def completion(request: dict):
    output = llm.generate(request)
    return output

@app.post("/v1/chat/completion")
def chat_completion(request: dict):
    output = llm.generate(request)
    return output

@app.post("/v1/models")
def models():
    return None
#返回本服务能提供的所有模型：{"object": "list", "data": [{"id": "gpt2", ...}]}
