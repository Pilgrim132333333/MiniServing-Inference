
from miniserving.EntryPoints.LLM import LLM
from miniserving.configs.EngineConfig import EngineConfig
import time
import pytest


def test_generate():
    llm = LLM(EngineConfig(model="GPT2"))
    prompt = "Hello"

    start = time.time()
    print(f"Start generate at {start}")

    output = llm.generate(prompt)

    end = time.time()
    print(f"End generate at {end}")
    print(f"Cost: {end - start:.2f}s")
    print(output)
    assert output is not None

