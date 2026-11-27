# vLLM Serving Lab — 系统全景架构文档（M0 → M3）

> 本文档汇总本仓库从单机基线（M0）、异步流式网关与准入控制（M1）、多副本前缀/负载感知路由（M2）、可观测性与自动回归门禁（Step 4）、Kubernetes 生命周期治理（Step 5）到 KEDA 队列驱动自动扩缩容（Step 6 / M3）的**完整端到端系统架构**。
>
> 核心架构决策与权衡取舍记录于 **[ADR 架构决策记录索引](adr/README.md)**。

---

## 1. 全栈全景架构图（数据面 / 控制面 / 观测面）

```mermaid
flowchart TB
    subgraph ClientLayer["客户端与压测层 (Clients & Benchmarks)"]
        C1["OpenAI SDK / SSE 流式客户端"]
        C2["压测与演练套件 (Step 4 / Step 5 / Step 6)"]
    end

    subgraph DataPlane["数据面 — 异步流式网关 (llm-gateway :8080 / NodePort :30080)"]
        GW_IN["FastAPI Async Entry + X-Request-ID / W3C traceparent"]
        ADM["AdmissionController (M1 Step 2)\nPer-Tenant Token Bucket & Quota (429)\nBounded Priority+Fair Queue (503/504)\nDynamic effective_max_inflight"]
        RTR["InstanceRouter (M2: Filter → Score → Pick)\nrandom | least-load | prefix-aware | cache+load-aware\nHotspot Damping + EstimatedPrefixCacheIndex"]
        REG["ReplicaRegistry\nHealth + Circuit Breaker + Live Inflight + vLLM Ground-Truth Metrics"]
        STM["3-State Streaming Proxy (M1 Step 1)\nNOT_STARTED → STREAMING → FINISHED\nDisconnect → upstream_resp.aclose()"]
    end

    subgraph BackendLayer["推理后端层 (Kubernetes Deployment / Real GPU)"]
        DNS["Headless Service (clusterIP: None)\nmock-vllm-headless.llm-serving.svc.cluster.local"]
        P1["Replica Pod 1 (Mock vLLM / Real vLLM)\n/startupz /readyz /healthz + 2-Phase preStop"]
        P2["Replica Pod 2 (Mock vLLM / Real vLLM)\n/startupz /readyz /healthz + 2-Phase preStop"]
        PN["Replica Pod N (1 ↔ 4 Autoscale)\n/v1/chat/completions + /metrics"]
    end

    subgraph ControlObsPlane["观测面与自动控制面 (Step 4 / Step 5 / Step 6)"]
        DISC["HeadlessDNSReplicaDiscoverer (1s Sync)\nAdd Ready IP / Drain Unready / Reap inflight=0"]
        PROM["Prometheus (2s Scrape)\ngateway_queue_depth + Histograms (_seconds) + vllm:*"]
        KEDA["KEDA v2.21.1 ScaledObject + Metrics Server\nmax_over_time(gateway_queue_depth[30s]) + Fallback (replicas=2)"]
        HPA["Kubernetes HPA\nscaleUp (+2/15s) / scaleDown (45s stabilization, -1/15s)"]
        REG_GATE["Regression Gatekeeper (scripts/check_regression.py)\nTTFT / TPOT / E2E P99 / Error Rate Gate"]
    end

    C1 --> GW_IN
    C2 --> GW_IN
    GW_IN --> ADM
    ADM --> RTR
    RTR <--> REG
    RTR --> STM
    STM --> P1
    STM --> P2
    STM --> PN

    P1 -. "Ready IP" .- DNS
    P2 -. "Ready IP" .- DNS
    PN -. "Ready IP" .- DNS
    DNS --> DISC
    DISC -- "更新健康/清理过期前缀/联动 effective_max_inflight" --> REG
    DISC -- "动态扩容槽位并唤醒队列" --> ADM

    GW_IN -. "/metrics" .-> PROM
    P1 -. "/metrics" .-> PROM
    PROM --> KEDA
    KEDA -- "external.metrics.k8s.io (AverageValue)" --> HPA
    HPA -- "修改 spec.replicas" --> BackendLayer
    PROM -. "Benchmark JSON" .-> REG_GATE
```

---

## 2. 端到端请求生命周期时序（含取消、故障与扩容联动）

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant GW as Gateway (Admission + Router)
    participant Pod as Backend Replica (vLLM / Mock)
    participant Prom as Prometheus + KEDA + HPA

    Client->>GW: POST /v1/chat/completions (stream=true, X-Request-ID)
    GW->>GW: 1) Tenant Token Bucket / Quota 校验 (超额 → 429)
    GW->>GW: 2) Fast Path 或进入 Bounded Priority/Fair Queue (满/超时 → 503/504)
    alt 客户端在排队期断开
        Client--xGW: Disconnect
        GW->>GW: 立刻踢出队列并释放 tenant_active (不派发给 GPU)
    else 获得准入槽位
        GW->>GW: 3) Filter (健康+熔断+KV) → Score (cache+load-aware + Hotspot Damping)
        GW->>Pod: 转发 POST (透传 X-Gateway-Request-ID + traceparent)
        alt 首字前上游失败/超时 (NOT_STARTED)
            Pod--xGW: Connect/Read Error
            GW-->>Client: 明确 JSON 502/504 + 记熔断失败 + 释放槽位
        else 正常进入流式 (STREAMING)
            Pod-->>GW: SSE Token Chunks (含 backend_request_id)
            GW-->>Client: 零缓冲透传 SSE Chunks
            opt 流式期间客户端取消
                Client--xGW: Disconnect
                GW->>Pod: upstream_resp.aclose() (触发后端释放 KV Blocks)
                GW->>GW: 幂等释放 replica.inflight & admission 槽位
            end
        end
    end
    GW-->>Prom: /metrics 暴露 gateway_queue_depth / Histograms / Rejections
    Prom->>Prom: 队列持续超阈 → KEDA 查询 max_over_time[30s] → HPA 扩容 Pod
    Prom-->>GW: 新 Pod Ready 进 Headless DNS → Gateway 放大 effective_max_inflight 并唤醒排队
```

---

## 3. 分层验证与证据矩阵（严格区分 Mock 控制面 vs 真实 GPU）

| 层次 | 验证目标 | 运行载体 | 关键代码与配置 | 实测证据报告与原始 JSON |
| :--- | :--- | :--- | :--- | :--- |
| **M0 单机基线** | 真实推理 16 组负载基线、Prefix Cache / Chunked Prefill / 量化实测 | **真实单卡 RTX 5080 + vLLM** | `scripts/benchmark.sh`, `scripts/run_exp*.sh` | [optimization_benchmarks_report.md](optimization_benchmarks_report.md), `results/baseline/summary.json` |
| **M1 异步网关 & 准入** | HTTP/SSE 转发、三态机、断连零泄漏、有界队列、`429/503/504`、公平与防饥饿 | 单元测试 + **真实单卡 RTX 5080 vLLM** | `gateway/app.py`, `gateway/admission.py`, `gateway/models.py` | [gateway.md](gateway.md), `tests/test_gateway_step{1,2}.py`, `tests/test_gateway_real_vllm.py` |
| **M2 多副本路由** | 4 种可切换策略、集群有效 KV 放大 $N$ 倍、热点倾斜 Cache vs Load 权衡 | Mock 多副本算法验证 + **真实单卡 RTX 5080 vLLM 物理校验** | `gateway/router.py`, `gateway/registry.py` | [gateway.md](gateway.md), `results/gateway/m2_*.json` |
| **Step 4 证据链与门禁** | Prometheus Histogram (`_seconds`)、全链路 Trace 关联、吞吐-尾延迟权衡、Goodput、回归门禁 | **真实单卡 RTX 5080 + vLLM** | `gateway/metrics.py`, `scripts/run_gateway_slo_benchmark.py`, `scripts/check_regression.py` | [slo_benchmark_regression_report.md](slo_benchmark_regression_report.md), `results/step4/*.json` |
| **Step 5 K8s 生命周期** | Headless DNS 发现、三探针隔离、两阶段 `preStop` 零丢包滚动发布、3 类实时故障演练 | 本地 **`kind` K8s 集群 + 3 Mock Replicas**（控制面验证） | `k8s/{mock-vllm,gateway,configmap}.yaml`, `gateway/discovery.py`, `mock_backend/server.py` | [k8s_lifecycle_report.md](k8s_lifecycle_report.md), `results/k8s/*.json` |
| **Step 6 队列自动扩缩容** | `gateway_queue_depth → Prometheus → KEDA → HPA → Deployment` 闭环、防抖、冷启动滞后、`fallback` | 本地 **`kind` + Prometheus + KEDA v2.21.1 + Mock Replicas**（控制面验证） | `k8s/{prometheus,keda-scaledobject}.yaml`, `scripts/run_keda_autoscaling_experiments.py` | [keda_autoscaling_report.md](keda_autoscaling_report.md), `results/k8s_autoscaling/*.json` |

---

## 4. 架构决策记录（ADR）索引

- **[ADR-0001](adr/0001-gateway-bounded-admission-queue.md)**：为什么采用独立异步网关与有界准入队列（分离 `429` / `503` 及独立于 vLLM 引擎队列）
- **[ADR-0002](adr/0002-cache-load-aware-routing-and-metric-separation.md)**：为什么采用带热点衰减的 `cache+load-aware` 路由，并严格隔离网关估计亲和度与真实 vLLM 命中率
- **[ADR-0003](adr/0003-k8s-headless-discovery-probes-and-graceful-drain.md)**：为什么在 Kubernetes 上采用 Headless Service DNS 发现、三探针职责分离与两阶段 `preStop` 流式优雅下线
- **[ADR-0004](adr/0004-keda-queue-driven-autoscaling-and-dynamic-capacity.md)**：为什么采用基于 `gateway_queue_depth` 的 KEDA 扩缩容、Gateway 动态并发联动、非对称防抖与 `fallback` 容错
