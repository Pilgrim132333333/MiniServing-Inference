import uuid
from pydantic import BaseModel, Field


class Request(BaseModel):
    prompt:str = Field(...,description="The prompt to generate text from")
    sampling_params:dict = Field(description="The sampling parameters to use")
    request_id:str = Field(default_factory=lambda: uuid.uuid4().hex)