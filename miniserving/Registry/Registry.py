import time
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class Lease:
    """
    一条租约 = instance infor + 何时过期
    """
    id: str
    url: str
    model: str
    ttl: float = 10.0
    expires: float = 0.0

class RegistryState:
    """
    被动记账员：只收注册和续约，TTL到期就当他死了
    """
    def __init__(self):
        self._leases: dict[str,Lease] = {}
    
    def register(self, id:str, url:str, model:str, ttl: float) -> Lease:
        """
        注册一个 instance
        """
        lease = Lease(id=id, url=url, model=model, ttl=ttl, expires_at = time.monotonic()+ ttl)
        self._leases[id] = lease
        return lease
    
    def heartbeat(self, id: str) -> Lease:
        lease = self._leases.get(id)

        if lease is None:
            raise KeyError(id)
        
        #lease语义： 过期即死，续约无效，必须重新register
        if lease.is_expired(time.monotonic()):
            del self._leases[id]
            raise KeyError(id)
        
        lease.expires_at = time.monotonic() + lease.ttl
        return lease
    
    def deregister(self, id: str) -> None:
        self._leases.pop(id, None)
    
    def list(self) -> list[Lease]:
        return list(self._leases.values())

class RegisterRequest(BaseModel):
    id: str
    url: str
    model: str
    ttl: float = 10.0

class HeartbeatRequest(BaseModel):
    id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry = RegistryState()
    yield


app = FastAPI(lifespan=lifespan)

@app.post("/register")
async def register(req: RegisterRequest):
    app.state.registry.register(req.id, req.url, req.model, req.ttl)
    return {"status": "registered", "ttl": req.ttl}

@app.post("/heartbeat")
async def heartbeat(req: HeartbeatRequest):
    app.state.registry.heartbeat(req.id)
    return {"status": "heartbeat"}

@app.post("/deregister")
async def deregister(req: HeartbeatRequest):
    app.state.registry.deregister(req.id)
    return {"status": "deregister"}

@app.get("/list")
async def list():
    return {"leases": app.state.registry.list()}