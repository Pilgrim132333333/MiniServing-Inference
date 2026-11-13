"""
限制request的一段时间内的平均接受数量
burst = 桶的容量： 定义最多能攒多少个token。同时也意味着瞬时突发request上限
tokens = 当前剩余token数量： 表示此刻还能立刻放行多少个请求
"""
import time

class RateLimiter:
    def __init__(self, max_rate: float, burst: int):
        
        if max_rate <= 0 or burst < 1:
            raise ValueError("max_rate must be positive, burst must be >= 1")
        self.max_rate = max_rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_check = time.monotonic()
    
    def acquire(self) -> float:
        now = time.monotonic()
        delta = now - self.last_check
        self.last_check = now
        self.tokens = min(self.tokens + delta * self.max_rate, self.burst)
        if self.tokens >=1:
            self.tokens -= 1
            return 0.0
        return (1.0 - self.tokens) / self.max_rate

