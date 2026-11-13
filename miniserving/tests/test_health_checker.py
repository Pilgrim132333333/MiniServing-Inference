"""
HealthChecker 最小测试（不启动真实引擎进程）。

与 test_gateway.py 同思路：用 httpx.MockTransport 拦截健康探测请求，
FakeEngine 按 mode 可编程模拟三种 /health 行为：
- healthy: 200 + {"healthy": true}   进程活着且健康
- sick:    200 + {"healthy": false}  进程活着但病了
- dead:    ConnectError                进程直接没了

覆盖的行为契约（对应 HealthChecker.py 头部的设计注释）：
- 探测成功 → 计数清零，保持 healthy
- 连续失败但未到阈值 → 仍 healthy
- 连续失败到阈值（默认 3）→ 标记 unhealthy
- 一次成功即恢复 healthy
- instance 被摘除（TTL 过期）→ 残留计数被清理
- start/stop 后台循环按 check_interval 周期探测
"""
import asyncio

import httpx

from miniserving.Gateway.HealthChecker import HealthChecker
from miniserving.Gateway.InstanceTable import Instance, InstanceTable


class FakeEngine:
    """一个可编程的引擎实例（按 port 路由）。"""

    def __init__(self, mode="healthy"):
        self.mode = mode
        self.probe_count = 0

    def handler(self, request):
        assert request.url.path == "/health", "探测必须打到 GET /health"
        self.probe_count += 1
        if self.mode == "dead":
            raise httpx.ConnectError("simulated dead process")
        if self.mode == "sick":
            return httpx.Response(200, json={"healthy": False})
        return httpx.Response(200, json={"healthy": True})


def make_instance(inst_id: str, port: int) -> Instance:
    return Instance(id=inst_id, url=f"http://127.0.0.1:{port}", model="test")


def build(engines: dict[int, FakeEngine], instances: list[Instance], **kwargs):
    """组装被测环境：InstanceTable + 注入了 MockTransport 的 HealthChecker。"""
    table = InstanceTable()
    for inst in instances:
        table.upsert(inst)

    def route(request):
        return engines[request.url.port].handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(route))
    checker = HealthChecker(instance_table=table, client=client, **kwargs)
    return table, checker


class TestFailureAccumulation:
    def test_healthy_probe_resets_counter(self):
        engine = FakeEngine()
        inst = make_instance("inst-1", 9001)
        _, checker = build({9001: engine}, [inst])

        asyncio.run(checker._check_once())

        assert inst.healthy is True
        assert checker.failures["inst-1"] == 0
        assert engine.probe_count == 1

    def test_single_failure_below_threshold_keeps_healthy(self):
        engine = FakeEngine(mode="sick")
        inst = make_instance("inst-1", 9001)
        _, checker = build({9001: engine}, [inst], failure_threshold=3)

        asyncio.run(checker._check_once())

        assert inst.healthy is True
        assert checker.failures["inst-1"] == 1

    def test_dead_process_counts_as_failure(self):
        engine = FakeEngine(mode="dead")
        inst = make_instance("inst-1", 9001)
        _, checker = build({9001: engine}, [inst])

        asyncio.run(checker._check_once())

        assert checker.failures["inst-1"] == 1

    def test_reaching_threshold_marks_unhealthy(self):
        engine = FakeEngine(mode="sick")
        inst = make_instance("inst-1", 9001)
        _, checker = build({9001: engine}, [inst], failure_threshold=3)

        for _ in range(3):
            asyncio.run(checker._check_once())

        assert inst.healthy is False

    def test_two_instances_probed_in_same_round(self):
        engines = {9001: FakeEngine(), 9002: FakeEngine()}
        instances = [make_instance("a", 9001), make_instance("b", 9002)]
        _, checker = build(engines, instances)

        asyncio.run(checker._check_once())

        assert engines[9001].probe_count == 1
        assert engines[9002].probe_count == 1


class TestRecovery:
    def test_one_success_recovers_unhealthy_instance(self):
        engine = FakeEngine()
        inst = make_instance("inst-1", 9001)
        inst.healthy = False  # 已被标记为不健康
        _, checker = build({9001: engine}, [inst])

        asyncio.run(checker._check_once())

        assert inst.healthy is True
        assert checker.failures["inst-1"] == 0


class TestCleanup:
    def test_removed_instance_counter_is_cleaned(self):
        engines = {9001: FakeEngine(), 9002: FakeEngine(mode="sick")}
        instances = [make_instance("a", 9001), make_instance("b", 9002)]
        table, checker = build(engines, instances)

        asyncio.run(checker._check_once())  # b 失败一次，计数残留
        table.remove("b")
        asyncio.run(checker._check_once())  # b 已不在表内 → 计数应被清理

        assert "b" not in checker.failures


class TestBackgroundLoop:
    def test_start_stop_probes_periodically(self):
        engine = FakeEngine()
        inst = make_instance("inst-1", 9001)
        _, checker = build({9001: engine}, [inst], check_interval=0.05)

        async def scenario():
            checker.start()
            await asyncio.sleep(0.18)
            await checker.stop()

        asyncio.run(scenario())

        assert engine.probe_count >= 3  # 0.18s / 0.05s 至少探测 3 次
