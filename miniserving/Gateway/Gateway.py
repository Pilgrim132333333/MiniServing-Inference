"""
Gateway 组装层：
- 解析client发送过来的请求，仅仅用于路由决策，转发时使用原始字节
- 注入 X-Request-Id/ X-Arrival-Time， 保证夸进程观测一致
- 错误语义映射 422（client） 503 （no instance)
"""
import time
import uuid
import math
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request as HTTPRequest, Response
from pydantic import ValidationError

from miniserving.EntryPoints.Request import Request
from miniserving.Gateway.InstanceTable import Instance, InstanceTable
from miniserving.Gateway.ProxyClient import ProxyClient
from miniserving.Gateway.RouterPolicy import RouterPolicy, RoundRobinRouterPolicy
from miniserving.Gateway.RateLimiter import RateLimiter

class Gateway:
    def __init__(self, 
    router_policy: RouterPolicy,
    instance_table: InstanceTable, 
    proxy_client: ProxyClient,
    rate_limiter: RateLimiter):
        self.router_policy = router_policy
        self.instance_table = instance_table
        self.proxy_client = proxy_client
        self.rate_limiter = rate_limiter
        
    async def handle(self, path:str, http_request: HTTPRequest) -> Response:
        raw_body = await http_request.body()

        try:
            request = Request.model_validate_json(raw_body)
        except ValidationError:
            return self._error(422,"invalid request body")
        
        wait =  self.rate_limiter.acquire()
        if wait > 0:
            return Response(status_code=429, content="Rate limit exceeded", media_type = "application/json",
                            headers={"Retry-After": str(math.ceil(wait))})
        
        request_id = http_request.headers.get("X-Request-Id", str(uuid.uuid4().hex))
        arrival_time = time.time()
        
        try:
            instance = self.router_policy.route(request,self.instance_table.list())
        except ValueError:
            return self._error(503, "No instance instance")
        
        headers = {
            "X-Request-Id": request_id,
            "X-Arrival-Time": str(arrival_time),
            "X-Instance-Id": instance.id,
            "content-type": "application/json"
        }

        try:
            upstream = await self.proxy_client.forward(instance, path, raw_body, headers)
        except httpx.HTTPStatusError as e:
            # 后端活着、只是拒绝了这个请求：原样透传它的状态码和响应体
            return Response(content=e.response.content,
            status_code=e.response.status_code,
            media_type="application/json")
        except httpx.HTTPError:
            return self._error(502, f"upstream {instance.id} unreachable")
        
        return Response(content=upstream.content,
        media_type = "application/json",
        status_code = upstream.status_code)



    @staticmethod
    def _error(status_code: int, detail: str):
        return Response(status_code=status_code, content=detail, media_type = "application/json")

STATIC_INSTANCES = [
    Instance(id="inst-1", url="http://127.0.0.1:8001", model="gpt2"),
    Instance(id="inst-2", url="http://127.0.0.1:8002", model="gpt2"),
]       

"""
life span 钩子函数。 yield之前在创建时调用。yield之后在销毁时调用
@asynccontextmanager 是表示用 async生成器手写上下文管理器的语法糖。等价于写一个类：
class Lifespan:
    async def __aenter__(self):   # = yield 之前的代码
        ...
    async def __aexit__(self, *exc):   # = yield 之后的代码
        ...
"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    proxy_client = ProxyClient()
    instance_table = InstanceTable()
    for inst in STATIC_INSTANCES:
        instance_table.upsert(inst)
    app.state.gateway = Gateway(
        router_policy=RoundRobinRouterPolicy(config = None),
        instance_table=instance_table,
        proxy_client=proxy_client,
        rate_limiter=RateLimiter(max_rate=100, burst=100),
    )

    yield

    await proxy_client.aclose()

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate(http_request: HTTPRequest) -> Response:
    return await app.state.gateway.handle("/generate", http_request)
