# ============================================================
# app.py — 带 Prometheus 埋点的 FastAPI 服务
# ============================================================

import time
import random
import asyncio
from fastapi import FastAPI, Request
from prometheus_client import (
    Counter,        # 计数器
    Gauge,          # 仪表盘
    Histogram,      # 直方图
    generate_latest,# 生成 /metrics 文本
    CONTENT_TYPE_LATEST
)
from fastapi.responses import Response

# ── 1. 创建 FastAPI 应用 ──────────────────────────────────────
app = FastAPI(title="我的第一个带监控的服务")


# ── 2. 定义 Prometheus 指标 ───────────────────────────────────

# Counter：统计 HTTP 请求总数
# 参数说明：
#   第1个参数 = 指标名称（Prometheus 里的唯一标识）
#   第2个参数 = 描述（给人看的）
#   labelnames = 标签（用于区分不同维度，比如不同接口、不同状态码）
REQUEST_COUNT = Counter(
    'http_requests_total',          # 指标名
    'HTTP请求总数',                  # 描述
    ['method', 'endpoint', 'status_code']  # 标签维度
)

# Counter：统计错误总数
ERROR_COUNT = Counter(
    'http_errors_total',
    'HTTP错误总数',
    ['endpoint']
)

# Gauge：当前正在处理的请求数（并发数）
REQUESTS_IN_PROGRESS = Gauge(
    'http_requests_in_progress',
    '当前正在处理的请求数'
)

# Histogram：请求响应时间分布
# buckets 定义分桶边界（单位：秒）
# 意思是：统计 <0.01s, <0.05s, <0.1s, <0.5s, <1s, <2s 的请求各有多少
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP请求响应时间（秒）',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

# Gauge：模拟业务指标——当前"任务队列"长度
TASK_QUEUE_SIZE = Gauge(
    'task_queue_size',
    '当前待处理任务队列长度'
)


# ── 3. 中间件：自动给每个请求埋点 ────────────────────────────
# 这是关键设计：不需要在每个接口里手动埋点
# 中间件会自动拦截所有请求，统一处理

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    endpoint = request.url.path
    method = request.method

    # Gauge +1：进来一个请求，并发数+1
    REQUESTS_IN_PROGRESS.inc()

    # 记录开始时间
    start_time = time.time()

    try:
        response = await call_next(request)
        status_code = str(response.status_code)

        # Counter +1：请求完成，总数+1（带标签）
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()

        return response

    except Exception as e:
        # 错误计数
        ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise e

    finally:
        # 无论成功失败，都要：
        # 1. Gauge -1：请求结束，并发数-1
        REQUESTS_IN_PROGRESS.dec()
        # 2. Histogram 记录耗时
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)


# ── 4. 业务接口（模拟真实场景）────────────────────────────────

@app.get("/")
async def root():
    """模拟一个普通接口"""
    # 随机延迟，模拟真实业务耗时
    await asyncio.sleep(random.uniform(0.01, 0.3))
    return {"message": "Hello, Prometheus!"}


@app.get("/slow")
async def slow_endpoint():
    """模拟一个慢接口（P99 会很高）"""
    await asyncio.sleep(random.uniform(0.5, 1.5))
    return {"message": "我是慢接口"}


@app.get("/error")
async def error_endpoint():
    """模拟一个偶发错误的接口"""
    if random.random() < 0.5:  # 50% 概率报错
        raise Exception("模拟错误！")
    return {"message": "这次没报错"}


@app.get("/tasks")
async def update_tasks():
    """模拟任务队列变化"""
    # 随机设置队列长度（模拟业务波动）
    queue_size = random.randint(0, 100)
    TASK_QUEUE_SIZE.set(queue_size)
    return {"queue_size": queue_size}


# ── 5. /metrics 接口：Prometheus 来这里抓数据 ─────────────────
# 这是整个系统的核心接口！
# Prometheus Server 会定期（默认15秒）来 GET 这个接口

@app.get("/metrics")
async def metrics():
    """暴露 Prometheus 指标"""
    return Response(
        content=generate_latest(),           # 生成标准格式的指标文本
        media_type=CONTENT_TYPE_LATEST       # text/plain; version=0.0.4
    )
