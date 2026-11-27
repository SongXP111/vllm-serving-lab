# ADR-0001: 独立异步流式网关与有界准入队列（分离 `429`/`503` 与 vLLM 引擎队列）

- **Status**: Accepted
- **Date**: 2026-11-26
- **Related Code & Evidence**: [`gateway/app.py`](../../gateway/app.py), [`gateway/admission.py`](../../gateway/admission.py), [`tests/test_gateway_step1.py`](../../tests/test_gateway_step1.py), [`tests/test_gateway_step2.py`](../../tests/test_gateway_step2.py), [`docs/slo_benchmark_regression_report.md`](../slo_benchmark_regression_report.md)

## Context

vLLM 内置了 OpenAI 兼容的 HTTP API Server 以及引擎内部的 `running` / `waiting` 连续批处理调度队列。但在多租户与突发洪峰场景下，直接把流量裸露给单机或多个 vLLM 实例会引发四大问题：
1. **无界拥塞与尾延迟雪崩**：无上限放行高并发请求进入 vLLM 会拉大 Batch、加剧 Chunked Prefill 干扰与 KV Cache 抢占，导致原始吞吐看似升高但 **SLO 有效吞吐（Goodput）跌至 0%**；
2. **缺乏多租户隔离与优先级公平**：单一突发租户会撑爆所有引擎等待队列（Noisy Neighbor）；
3. **客户端断连浪费 GPU 算力**：客户端取消或超时后，若不能迅速切断上游 HTTP 流，后端会继续自回归 Decode 浪费显存与算力；
4. **跨实例路由前提缺失**：只有网关层先收敛“接受/排队/拒绝”决策，下游路由器才能拥有准确、可信的节点并发视图。

## Decision

构建独立的 Python FastAPI + `httpx.AsyncClient` 异步流式网关，并在路由层之前实现独立的 **`AdmissionController`**：
1. **三态流式状态机（`NOT_STARTED → STREAMING → FINISHED`）**：首字前上游失败返回明确 JSON 错误；首字后发出 SSE 错误事件；客户端断开时在 `finally:` 立即执行 `upstream_resp.aclose()` 传播取消并幂等释放槽位。
2. **严格区分拒绝语义**：
   - `429 Too Many Requests`（带 `Retry-After`）：Per-Tenant Token Bucket 速率超限或并发配额超限；
   - `503 Service Unavailable`：系统过载导致有界队列打满（`queue_full`）或排队超时（`queue_timeout`）；
   - `504 Gateway Timeout`：端到端请求 `deadline` 过期或后端超时。
3. **多级公平与防饥饿**：`HIGH > NORMAL > LOW` 优先级队列 + Consecutive Cap + Aging 老化 + 同优先级内 Per-Tenant Round-Robin 轮转。
4. **与 vLLM 内部队列解耦**：Gateway 用 `max_inflight`（及后续动态 `effective_max_inflight`）锁死进入引擎的并发水位，将超额请求截留在轻量的入口有界队列中，保护 vLLM Event Loop 与 GPU 批处理稳定性。

## Alternatives Considered

- **方案 A：直接暴露 vLLM API Server + Nginx 反向代理**  
  *放弃原因*：Nginx 无法感知 LLM 前缀亲和、租户并发配额、优先级防饥饿与单副本 KV 显存水位，且难以区分排队超时与流式首字超时。
- **方案 B：不限制 `max_inflight`，完全交由 vLLM 内部调度器排队**  
  *放弃原因*：在 RTX 5080 Step 4 实测中，无界并发（`max_inflight=20`）虽然将原始吞吐推至 `225.3 tok/s`，但 `TPOT P50=24.47ms`、`TTFT P99=5.26s`，导致 **Goodput 归零（`0/20`）**。

## Consequences & Evidence

- **正向收益**：
  - RTX 5080 Step 4 实测证明，开启有界反压（`max_inflight=3, queue_capacity=3`）后，在同洪峰下将 TTFT P99 骤降 **`-88.8%`**（`5,260ms → 588.6ms`），把 Goodput 从 `0.00 rps` 救回 **`1.13 rps`**；
  - 12 项 M1 Step 1/Step 2 自动化测试验证了断连取消零泄漏、无低优饥饿及多租户公平隔离。
- **代价与约束**：
  - 引入了约 1~2ms 的网关转发开销与单网关进程的状态维护（未来扩展多副本 Gateway 时需配合一致性哈希或外部协调状态）。
