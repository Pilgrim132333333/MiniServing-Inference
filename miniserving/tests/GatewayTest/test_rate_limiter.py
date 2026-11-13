"""
RateLimiter 最小测试（纯逻辑，无 IO、无引擎进程）。

令牌桶行为契约（对应 RateLimiter.py 头部的设计注释）：
- 构造校验：max_rate <= 0 或 burst < 1 → ValueError
- 初始桶满：新实例可立刻放行 burst 个请求
- 超出突发：第 burst+1 个被拒，acquire() 返回等待秒数 > 0
- 等待秒数语义：桶恰好空时 ≈ 1/max_rate（补满下一个令牌所需时间）
- 时间恢复：桶空后经过足够时间，可再次放行
- 补充封顶：无论空闲多久，连续放行数不超过 burst
- 窗口不变量：T 秒内放行数 <= burst + max_rate * T
"""
import time

import pytest

from miniserving.Gateway.RateLimiter import RateLimiter


class TestConstruction:
    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            RateLimiter(max_rate=0, burst=10)
        with pytest.raises(ValueError):
            RateLimiter(max_rate=-1, burst=10)
        with pytest.raises(ValueError):
            RateLimiter(max_rate=5, burst=0)

    def test_starts_full(self):
        rl = RateLimiter(max_rate=5, burst=3)
        assert rl.tokens == 3.0


class TestBurst:
    def test_admits_burst_requests_immediately(self):
        rl = RateLimiter(max_rate=5, burst=3)
        for _ in range(3):
            assert rl.acquire() == 0.0

    def test_rejects_after_burst_exhausted(self):
        rl = RateLimiter(max_rate=5, burst=3)
        for _ in range(3):
            rl.acquire()

        wait = rl.acquire()

        assert wait > 0

    def test_wait_equals_time_to_next_token_when_bucket_empty(self):
        rl = RateLimiter(max_rate=10, burst=1)
        rl.acquire()  # 桶恰好空

        wait = rl.acquire()

        # 下一个令牌需要 1/max_rate 秒才补满
        assert wait == pytest.approx(1 / 10, rel=0.2)


class TestRecovery:
    def test_token_recovers_after_wait(self):
        rl = RateLimiter(max_rate=50, burst=1)
        rl.acquire()  # 桶空

        time.sleep(0.05)  # 50/s × 0.05s = 2.5 个令牌，足够补 1 个

        assert rl.acquire() == 0.0

    def test_refill_caps_at_burst(self):
        rl = RateLimiter(max_rate=100, burst=2)
        time.sleep(0.1)  # 理论上可补 10 个，但桶容量只有 2

        assert rl.acquire() == 0.0
        assert rl.acquire() == 0.0
        assert rl.acquire() > 0  # 第 3 个必须被拒


class TestThroughputBound:
    def test_window_admissions_bounded_by_burst_plus_rate_times_T(self):
        # 不变量：任意 T 秒窗口内放行数 <= burst + max_rate * T
        max_rate, burst = 100, 5
        rl = RateLimiter(max_rate=max_rate, burst=burst)

        start = time.monotonic()
        admitted = 0
        while time.monotonic() - start < 0.2:
            if rl.acquire() == 0.0:
                admitted += 1
            else:
                time.sleep(0.005)
        elapsed = time.monotonic() - start

        assert admitted >= burst  # 突发额度至少全部放行
        # 上界（+1 容忍最后一次 acquire 与取 elapsed 之间的微小误差）
        assert admitted <= burst + max_rate * elapsed + 1
