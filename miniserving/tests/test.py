
from miniserving.EntryPoints.LLM import LLM
from miniserving.EntryPoints.API_server import app
import time
import pytest

def test_generate():
    prompt = "Hello"
    llm = LLM("GPT2")

    start = time.time()
    print(f"Start generate at {start}")

    output = llm.generate(prompt)

    end = time.time()
    print(f"End generate at {end}")
    print(f"Cost: {end - start:.2f}s")
    print(output)

def test_API_completion():
    prompt = "Hello"
    response = requests.post("/v1/generate", json={"prompt": prompt})
    print(response.json())
    assert response.status_code == 200