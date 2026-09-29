from fastapi import FastAPI
import uuid
from miniserving.Engine.AsyncEngineCore import AsyncEngineCore
from miniserving.EntryPoints.Request import Request
from miniserving.EntryPoints.Response import Response
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

async def lifespan(app: FastAPI):
    llm.start()
    yield
    llm.stop()
app = FastAPI(lifespan=lifespan)
llm = AsyncEngineCore("GPT2")


@app.post("/generate", response_model=Response)
async def generate(request: Request):
    future = await llm.add_request(request)
    output = await future
    return Response(output=output)

@app.post("/v1/completion")
def completion(request: Request):
    output = llm.generate(request)
    return Response(output=output)

@app.post("/v1/chat/completion")
def chat_completion(request: Request):
    output = llm.generate(request)
    return Response(output=output)

@app.post("/v1/models")
def models():
    return None
#返回本服务能提供的所有模型：{"object": "list", "data": [{"id": "gpt2", ...}]}
