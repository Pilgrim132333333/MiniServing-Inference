
from miniserving.Engine.EngineCore import EngineCore
import time
import pytest


def test_generate():
    llm = EngineCore("GPT2")
    prompt = "Hello"

    start = time.time()
    print(f"Start generate at {start}")

    output = llm.generate(prompt)

    end = time.time()
    print(f"End generate at {end}")
    print(f"Cost: {end - start:.2f}s")
    print(output)
    assert output is not None

