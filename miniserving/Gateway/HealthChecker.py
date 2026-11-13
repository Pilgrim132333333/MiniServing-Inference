"""
Gateway侧的主动健康检查

ServiceDiscovery只能解决是否存在instance，并没有监管健康行
- 每check_interval秒并发探测所有instance的GET/health
- 连续failure——threshold 次失败 -> 标记unhealthy
- 一次成功即恢复 healthy

两种摘除：
- TTL过期： 进程直接死， ServiceDiscovery Remove
- 连续失败三次： 进程活着但是病了
"""

import asyncio
from contextlib import suppress

import httpx

from miniserving.Gateway.InstanceTable import InstanceTable

class HealthChecker:
    def __init__(
        self,
        instance_table: InstanceTable,
        check_interval: float = 2.0,
        failure_threshold: int = 3,
        probe_timeout: float = 2.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.instance_table = instance_table
        self.check_interval = check_interval
        self.failure_threshold = failure_threshold
        self.probe_timeout = probe_timeout
        self.client = client or httpx.AsyncClient()

        self._task: asyncio.Task | None = None

        self.failures: dict[str, int] = {}
    
    async def _probe_one(self,instance) -> bool:
        try:
            resp = await self.client.get(
                f"{instance.url}/health",
                timeout=self.probe_timeout,
            )
            return resp.status_code == 200 and resp.json()["healthy"]
        except httpx.HTTPError:
            return False
    
    async def _check_once(self):
        instances = self.instance_table.list()
        if not instances:
            return
        
        #并发探测所有instances
        results = await asyncio.gather(*[self._probe_one(inst) for inst in instances])

        for instance, ok in zip(instances, results):
            if ok:
                self.failures[instance.id] = 0
                if not instance.healthy:
                    self.instance_table.set_healthy(instance.id, True)
            else:
                count = self.failures.get(instance.id, 0) + 1
                self.failures[instance.id] = count
                if count >= self.failure_threshold:
                    self.instance_table.set_healthy(instance.id, False)
        
        alive_ids = {inst.id for inst in instances if inst.healthy}
        for gone_id in self.failures.keys() - alive_ids:
            self.failures.pop(gone_id)

    async def _run(self):
        with suppress(asyncio.CancelledError):
            while True:
                await self._check_once()
                await asyncio.sleep(self.check_interval)

    def start(self):
        self._task = asyncio.create_task(self._run())
    
    async def stop(self):
        if self._task:
            self._task.cancel()
            await self._task
            self._task = None