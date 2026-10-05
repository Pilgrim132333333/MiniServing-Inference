from prometheus_client import Gauge, Histogram, Counter, Summary

REQUEST_E2E = Histogram(
    "request_e2e_seconds", 
    "Request end-to-end latency in seconds",
    ["model"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])

REQUEST_SERVER_TTFT = Histogram(
    "request_ttft_seconds", 
    "Request total-to-first-token latency in seconds",
    ["model"]
) # Server side First Token latency - arrival time

REQUEST_CLIENT_TTFT = Histogram(
    "request_ttft_client_seconds", 
    "Request total-to-first-token latency in seconds",
    ["model"]
) # Client side First Token latency - arrival time

REQUEST_TPOT = Histogram(
    "request_tpot_seconds", 
    "Request total-to-output-token latency in seconds",
    ["model"]
)

STEP_TIME = Histogram(
    "step_time_seconds", 
    "Step time in seconds",
    ["model"]
)

BATCH_SIZE = Gauge(
    "batch_size", 
    "Batch size",
    ["model"]
)

# Gauge：瞬时值，可增可减
RUNNING_REQUESTS = Gauge("running_requests", "正在处理的请求数")
WAITING_REQUESTS = Gauge("waiting_requests", "WAITING 队列中的请求数")
KV_CACHE_USED = Gauge("kv_cache_used_tokens", "已用 KV cache token 数")

# Counter：只增不减的累计值
TOTAL_TOKENS = Counter("generated_tokens_total", "累计生成 token 数")
TOTAL_REQUESTS = Counter("requests_total", "累计请求数", ["model", "status"])

# Summary：另一种延迟指标，自带分位数（但不如 Histogram 灵活）
SCHEDULE_TIME = Summary("schedule_seconds", "调度耗时")
