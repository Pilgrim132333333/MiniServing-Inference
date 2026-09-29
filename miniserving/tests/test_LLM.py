
from miniserving.EntryPoints.AsyncEngineCore import AsyncEngineCore
from miniserving.EntryPoints.API_server import app
import time
import pytest
from fastapi.testclient import TestClient
import requests

@pytest.fixture
def llm():
    return AsyncEngineCore("GPT2")

client = TestClient(app)

def test_API_generate():
    prompt = "Hello"
    response = client.post("/generate", json={"prompt": prompt})
    print(response.json())
    assert response.status_code == 200
    assert response.json()["output"] is not None

def test_generate(llm):
    prompt = "Hello"

    start = time.time()
    print(f"Start generate at {start}")

    output = llm.generate(prompt)

    end = time.time()
    print(f"End generate at {end}")
    print(f"Cost: {end - start:.2f}s")
    print(output)
    assert output is not None

