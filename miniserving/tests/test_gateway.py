"""
Gateway 集成测试（不启动真实引擎进程）。

核心手段：httpx.MockTransport —— 在进程内拦截 ProxyClient 发出的所有 HTTP 请求，
由 handler 直接返回构造的响应。这样可以：
- 断言网关到底把请求转发到了哪个端口（路由结果）
- 断言转发时携带了哪些 header（X-Request-Id / X-Arrival-Time 注入）
- 任意注入故障：连接失败（dead_ports）、上游错误状态码（next_status）
"""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from miniserving.Gateway import Gateway as GatewayModule
from miniserving.Gateway.ProxyClient import ProxyClient


class MockBackendState:
    """mock 引擎的可编程状态：记录收到的请求，支持注入故障。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.received = []       # 每条转发请求的元数据
        self.dead_ports = set()  # 命中即模拟连接失败（进程不存在）
        self.next_status = None  # 非 None 时上游返回该状态码（模拟引擎拒绝请求）

    def handler(self, request):
        port = request.url.port
        if port in self.dead_ports:
            raise httpx.ConnectError(f"simulated connection failure to :{port}")

        self.received.append({
            "port": port,
            "path": request.url.path,
            "request_id": request.headers.get("X-Request-Id"),
            "arrival_time": request.headers.get("X-Arrival-Time"),
        })

        if self.next_status is not None:
            return httpx.Response(
                self.next_status,
                json={"detail": f"upstream rejected with {self.next_status}"},
            )

        body = json.loads(request.content)
        return httpx.Response(200, json={
            "output": f"[mock:{port}] {body['prompt']}",
            "request_id": request.headers.get("X-Request-Id"),
        })


STATE = MockBackendState()


class MockProxyClient(ProxyClient):
    """替换真实网络栈：继承 ProxyClient 的接口，内部换成 MockTransport。"""

    def __init__(self):
        self.config = None
        self._client = httpx.AsyncClient(
            transport=httpx.MockTransport(STATE.handler),
            timeout=httpx.Timeout(connect=2.0, read=120.0, write=5.0, pool=5.0),
        )


@pytest.fixture()
def client(monkeypatch):
    STATE.reset()
    # Gateway 的 lifespan 里调用的是模块命名空间中的 ProxyClient，替换之
    monkeypatch.setattr(GatewayModule, "ProxyClient", MockProxyClient)
    with TestClient(GatewayModule.app) as c:
        # lifespan upsert 的是模块级 STATIC_INSTANCES 里的同一批 Instance 对象，
        # 上一个测试改过的 healthy=False 会泄漏到本测试 —— 这里统一复位
        for inst in GatewayModule.app.state.gateway.instance_table.list():
            inst.healthy = True
        yield c


def post_generate(client, prompt="hi", headers=None):
    return client.post("/generate", json={"prompt": prompt}, headers=headers or {})


class TestForwarding:
    def test_returns_upstream_response(self, client):
        resp = post_generate(client, prompt="Hello")
        assert resp.status_code == 200
        data = resp.json()
        assert data["output"].startswith("[mock:")
        assert "Hello" in data["output"]

    def test_round_robin_distributes_across_instances(self, client):
        ports = [post_generate(client).json()["output"].split(":")[1].split("]")[0]
                 for _ in range(4)]
        assert ports == ["8001", "8002", "8001", "8002"]


class TestHeaderInjection:
    def test_generates_request_id_when_client_omits_it(self, client):
        resp = post_generate(client)
        sent = STATE.received[-1]
        # uuid4().hex 是 32 个十六进制字符
        assert sent["request_id"] and len(sent["request_id"]) == 32
        # 响应回带的 id 与转发到上游的 id 一致（全链路同一个 id）
        assert resp.json()["request_id"] == sent["request_id"]

    def test_client_supplied_request_id_is_preserved(self, client):
        resp = post_generate(client, headers={"X-Request-Id": "trace-001"})
        assert STATE.received[-1]["request_id"] == "trace-001"
        assert resp.json()["request_id"] == "trace-001"

    def test_arrival_time_header_is_forwarded(self, client):
        post_generate(client)
        raw = STATE.received[-1]["arrival_time"]
        assert raw is not None
        assert float(raw) > 0  # 是合法的 epoch 浮点数


class TestErrorSemantics:
    def test_invalid_json_returns_422_without_touching_upstream(self, client):
        resp = client.post(
            "/generate",
            content=b"not json at all",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422
        assert STATE.received == []

    def test_missing_prompt_returns_422_without_touching_upstream(self, client):
        resp = client.post("/generate", json={"model": "gpt2"})
        assert resp.status_code == 422
        assert STATE.received == []

    def test_no_instances_returns_503(self, client):
        table = GatewayModule.app.state.gateway.instance_table
        table.remove("inst-1")
        table.remove("inst-2")
        resp = post_generate(client)
        assert resp.status_code == 503

    def test_unhealthy_instance_is_not_routed(self, client):
        GatewayModule.app.state.gateway.instance_table.set_healthy("inst-2", False)
        for _ in range(3):
            resp = post_generate(client)
            assert resp.status_code == 200
            assert ":8001]" in resp.json()["output"]

    def test_dead_upstream_returns_502(self, client):
        STATE.dead_ports = {8002}
        results = [post_generate(client).status_code for _ in range(4)]
        # 轮转命中 8002 的请求 → 连接失败 → 502；命中 8001 的 → 200
        assert results.count(502) == 2
        assert results.count(200) == 2

    def test_upstream_error_status_is_passed_through(self, client):
        STATE.next_status = 400
        resp = post_generate(client)
        # 后端活着只是拒绝请求：网关应原样透传 400 和 body，而不是转成 503
        assert resp.status_code == 400
        assert "upstream rejected with 400" in resp.json()["detail"]
