"""
1.将client发送过来的字节流，解析成我定义的Request对象
"""
from miniserving.Engine.Sequence import Request
from miniserving.Gateway.RouterPolicy import RouterPolicy
from fastapi import Header, APIRouter
import time
class Gateway:
    def __init__(self, router_policy: RouterPolicy):
        self.router_policy = router_policy

    @app.post("/generate",response_model=Response)
    async def generate(self, request: Request, x_request_id: str | None = 
    Header(default = None, alias="X-Request-Id")):
        request_id = x_request_id or str4().hex
        arrival_time = time.time()
        