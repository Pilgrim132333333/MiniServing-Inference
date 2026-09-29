
from miniserving.Engine.AsyncEngineCore import AsyncEngineCore
from miniserving.EntryPoints.Request import Request
from miniserving.EntryPoints.API_server import app
import time
import pytest
from fastapi.testclient import TestClient
import requests

@pytest.fixture
def llm():
    engine = AsyncEngineCore("GPT2")
    yield engine
    if hasattr(engine, 'task'):
        engine.task.cancel()

@pytest.fixture
def api_client():
    with TestClient(app) as c:
        yield c
@pytest.mark.asyncio
async def test_API_generate(api_client):
    prompt = "Hello"
    response = api_client.post("/generate", json={
        "model": "GPT2",
        "prompt": prompt,
        "sampling_params": {"n": 1, "max_tokens": 20}
    })
    print(response.json())
    assert response.status_code == 200, response.json()
    assert response.json()["output"] is not None

@pytest.mark.asyncio
async def test_generate(llm):
    llm.start()
    prompt = "Hello"
    start = time.time()
    print(f"Start generate at {start}")
    request = Request(model="GPT2", prompt=prompt, sampling_params={"n": 1, "max_tokens": 20})
    future = await llm.add_request(request)
    output = await future
    
    end = time.time()
    print(f"End generate at {end}")
    print(f"Cost: {end - start:.2f}s")
    print(output)
    assert output is not None

