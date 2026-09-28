from fastapi import FastAPI
from pydantic import BaseModel
from Model.LLM import LLM
from EntryPoints.Request import Request

app = FastAPI()
llm = LLM("gpt2")


@app.post("/generate")
def generate(request: Request):
    output = llm.generate(request)
    return output

@app.post("/v1/completion")

@app.post("/v1/chat/completion")

@app.post("/v1/models")
#返回本服务能提供的所有模型：{"object": "list", "data": [{"id": "gpt2", ...}]}
