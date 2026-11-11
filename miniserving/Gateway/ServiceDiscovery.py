"""
在Gateway侧
负责同步Gateway侧的instanceTable和Registry侧实际注册的instance
"""

import asyncio
import time
from contextlib import suppress

import httpx

from miniserving.Gateway.InstanceTable import Instance, InstanceTable

class ServiceDiscovery:
    def __init__(
        self,
        registry_url: str,
        instance_table: InstanceTable,
        poll_interval: float = 2.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.registry_url = registry_url
        self.instance_table = instance_table
        self.poll_interval = poll_interval
        self.client = client or httpx.AsyncClient(
            timeout = httpx.Timeout(connect=2.0, read=5.0,write=5.0,pool=5.0),
        )
        self._task: asyncio.Task | None = None
    
    async def _poll_once(self):
        resp = await self.client.get(self.registry_url + "/list")
        resp.raise_for_status()

        remote = {info["id"]: info for info in resp.json()["instances"]}

        local_ids = {inst.id for inst in self.instance_table.list()}    

        for id_, info in remote.items():
            if id_ in local_ids:
                #已存在，原地刷新，绝对不覆盖healthy
                inst = self.instance_table.get(id_)
                inst.url = info["url"]
                inst.model = info["model"]
                inst.last_heartbeat = time.time()
            else:
                #新instance， 默认健康入场，具体死活交给HealthChecker探测
                self.instance_table.upsert(Instance(
                    id=id_,
                    url=info["url"],
                    model=info["model"],
                    last_heartbeat=time.time(),
                    healthy=True,
                ))
        
        #Registry里已消失的（TTL 过期/ deregister）
        for gone_id in local_ids - set(remote):
            self.instance_table.delete(gone_id)
    
    async def _run(self):
        while True:
            try:
                await self._poll_once()
            except httpx.HTTPError:
                pass
            await asyncio.sleep(self.poll_interval)
    
    def start(self):
        #作为一个后台asyncio任务，挂在Gateway的事件循环里
        self._task = asyncio.create_task(self._run())
    
    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        
        await self._client.aclose()
