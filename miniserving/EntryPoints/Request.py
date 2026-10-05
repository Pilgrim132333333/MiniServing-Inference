import uuid
from pydantic import BaseModel, Field
import time


class Request(BaseModel):
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    model: str = Field(default="gpt2", description="The model to use")
    prompt: str = Field(..., description="The prompt to generate text from")
    sampling_params: dict = Field(default_factory=dict, description="The sampling parameters to use")

    #observability
    arrival_time: float = Field(default_factory=lambda: time.time(), description="The arrival time of the request")
