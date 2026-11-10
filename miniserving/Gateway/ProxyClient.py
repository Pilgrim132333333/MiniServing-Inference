"""
这个类使用了AsyncClient类，其维护一个已经建立连接的instance的socket的池子
可以用它来进行post
AsyncClient                          ← 你接触的 API 层
  ├─ 配置：base_url, headers, timeout, cookies, auth
  ├─ Request 对象的构造与状态码语义（raise_for_status 等）
  │
  └─ AsyncHTTPTransport              ← 传输层（可替换，比如换成 mock 测试）
       │
       └─ AsyncConnectionPool        ← 核心：连接池（httpcore 实现）
            ├─ 按 origin 分组的连接字典
            │    key = (scheme, host, port)
            │    例如 ("http", "127.0.0.1", 8001)
            ├─ 每组内连接有两种状态：
            │    ACTIVE（已借出，正在传请求）
            │    IDLE  （空闲，躺在池里等复用）
            └─ 三个池参数：
                 max_connections          总量上限
                 max_keepalive_connections 空闲上限
                 keepalive_expiry         空闲存活时间

"""
import httpx
from miniserving.Gateway.InstanceTable import Instance
class ProxyClient:
    def __init__(self, config):
        self.config = config
        self._client = httpx.AsyncClient(
            timeout = httpx.Timeout(connect = 2.0, read = 120.0, write=5.0, pool = 5.0),
        )
    
    async def forward(
        self,
        instance: Instance, #要转发的instance的地址
        path: str, #具体的要执行操作的API，和instance的url拼在一起
        raw_body: bytes,
        headers: dict[str,str],
    ) -> httpx.Response:
        url = f"{instance.url}{path}"
        response = await self._client.post(
            url,
            headers=headers,
            content=raw_body,
        )
        response.raise_for_status()
        return response
    
    async def aclose(self):
        await self._client.aclose()