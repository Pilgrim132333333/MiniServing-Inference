"""
RouterPolicy / InstanceTable 单元测试
纯内存逻辑，无网络、无事件循环，直接构造对象验证行为。
"""
import pytest

from miniserving.EntryPoints.Request import Request
from miniserving.Gateway.InstanceTable import Instance, InstanceTable
from miniserving.Gateway.RouterPolicy import RoundRobinRouterPolicy


def make_instances():
    return [
        Instance(id="inst-1", url="http://127.0.0.1:8001", model="gpt2"),
        Instance(id="inst-2", url="http://127.0.0.1:8002", model="gpt2"),
    ]


class TestRoundRobinRouterPolicy:
    def test_first_request_goes_to_first_instance(self):
        policy = RoundRobinRouterPolicy(config=None)
        pick = policy.route(Request(prompt="hi"), make_instances())
        assert pick.id == "inst-1"

    def test_rotation_alternates(self):
        policy = RoundRobinRouterPolicy(config=None)
        instances = make_instances()
        picks = [policy.route(Request(prompt="hi"), instances).id for _ in range(4)]
        assert picks == ["inst-1", "inst-2", "inst-1", "inst-2"]

    def test_unhealthy_instance_is_skipped(self):
        policy = RoundRobinRouterPolicy(config=None)
        instances = make_instances()
        instances[1].healthy = False
        picks = [policy.route(Request(prompt="hi"), instances).id for _ in range(3)]
        assert picks == ["inst-1", "inst-1", "inst-1"]

    def test_model_mismatch_raises(self):
        policy = RoundRobinRouterPolicy(config=None)
        llama_only = [Instance(id="llama-1", url="http://127.0.0.1:8003", model="llama")]
        with pytest.raises(ValueError):
            policy.route(Request(prompt="hi"), llama_only)

    def test_empty_table_raises(self):
        policy = RoundRobinRouterPolicy(config=None)
        with pytest.raises(ValueError):
            policy.route(Request(prompt="hi"), [])


class TestInstanceTable:
    def test_upsert_and_list(self):
        table = InstanceTable()
        table.upsert(Instance(id="a", url="http://x:1", model="gpt2"))
        assert [inst.id for inst in table.list()] == ["a"]

    def test_set_healthy_toggles(self):
        table = InstanceTable()
        table.upsert(Instance(id="a", url="http://x:1", model="gpt2"))
        table.set_healthy("a", False)
        assert table.get("a").healthy is False

    def test_remove(self):
        table = InstanceTable()
        table.upsert(Instance(id="a", url="http://x:1", model="gpt2"))
        table.remove("a")
        assert table.list() == []
