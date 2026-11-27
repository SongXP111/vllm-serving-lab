# ADR-0004: 基于 `gateway_queue_depth` 的 KEDA 扩缩容、动态并发上限联动与 Fail-Safe Fallback

- **Status**: Accepted
- **Date**: 2026-11-26
- **Related Code & Evidence**: [`k8s/keda-scaledobject.yaml`](../../k8s/keda-scaledobject.yaml), [`k8s/prometheus.yaml`](../../k8s/prometheus.yaml), [`gateway/admission.py`](../../gateway/admission.py), [`docs/keda_autoscaling_report.md`](../keda_autoscaling_report.md), [`results/k8s_autoscaling/summary.json`](../../results/k8s_autoscaling/summary.json)

## Context

在 Kubernetes 上对 LLM 推理副本实施自动扩缩容存在 4 个核心挑战：
1. CPU/Memory 与 GPU/KV 显存瓶颈脱节，无法作为有效前导信号；
2. 如果 Gateway 的 `max_inflight` 是写死的静态常量，即便 KEDA 把后端从 1 扩到 4 副本，请求仍会被挡在 Gateway 静态并发上限之外，扩容无法缓解排队；
3. 大模型冷启动慢且反馈闭环存在显著控制滞后（Control Lag），激进扩缩容极易引发冷启动过冲（Overshoot）与反复缩容抖动（Flapping）；
4. 监控链路（Prometheus）短暂不可用时，若无保护策略，外部指标缺失可能导致误缩容事故。

## Decision

1. **采用 KEDA Prometheus Scaler 消费 Gateway 队列前导指标**：
   - 使用 `metricType: AverageValue` 查询 **`max_over_time(gateway_queue_depth[30s])`**（`threshold: "2"`, `minReplicaCount: 1`, `maxReplicaCount: 4`），明确区分该队列属于 **Gateway 入口准入队列**而非引擎内部调度队列；
   - 由 KEDA Operator 自动生成托管 HPA，通过 `external.metrics.k8s.io` 驱动 `Deployment/mock-vllm`。
2. **Gateway 动态并发上限联动（`GATEWAY_DYNAMIC_MAX_INFLIGHT=true`）**：
   - `HeadlessDNSReplicaDiscoverer` 每次检测到 Ready 副本数变化时，实时更新 `effective_max_inflight = min(max_inflight, max(per_replica_max_inflight, ready_replicas × per_replica_max_inflight))`；
   - 扩容时**立即唤醒有界队列中的等待请求**，缩容时让在途请求自然排空；同步暴露 `gateway_effective_max_inflight` 指标供观测。
3. **非对称 HPA `behavior` 防过冲与防抖**：
   - **快扩限步**：`scaleUp.policies` 限制每次最多 `+2 Pods / 15s`，缓和冷启动期连续过冲；
   - **慢缩长稳定窗**：`scaleDown.stabilizationWindowSeconds: 45` + 每次最多 `-1 Pod / 15s`（明确 `cooldownPeriod` 仅作用于缩零场景，`1 ↔ N` 防抖必须配 HPA `behavior`）。
4. **指标断连 Fail-Safe `fallback`**：
   - 配置 `fallback: { failureThreshold: 2, replicas: 2 }` 与 `ignoreNullValues: "false"`，在 Prometheus 不可达或指标缺失时保底维持 2 副本，恢复后自动退出降级。

## Alternatives Considered

- **方案 A：使用原生 CPU/Memory HPA**  
  *放弃原因*：LLM 瓶颈在显存 KV Cache、GPU 算力与并发槽位，CPU 利用率滞后且失真。
- **方案 B：仅依赖 vLLM 引擎内部 `sum(vllm:num_requests_waiting)`**  
  *放弃原因*：在 Gateway 开启有界准入与 503 反压后，超载压力优先积压在 Gateway 队列而非后端引擎内部，仅看引擎队列会掩盖真实入口积压与拒流。生产环境中可将两者作为互补指标或叠加 `gateway_inflight_requests + gateway_queue_depth`。
- **方案 C：设置 `minReplicaCount: 0`（缩零）**  
  *放弃原因*：大模型权重冷启动耗时数分钟，线上交互服务缩零会导致首请求灾难性延迟。

## Consequences & Evidence

- **正向收益（本地 `kind` + KEDA v2.21.1 四项实机实验铁证）**：
  - **同 Burst 固定 1 副本 vs KEDA 弹性 `1 → 4`**：平均队列深度降 **`-61.5%`**（`12.87 → 4.95`），全程 503 拒绝率降 **`-79.9%`**（`46.15% → 9.29%`），扩容稳态窗（`20–55s`）实现 **`0.00%` 零拒绝**、TTFT P99 降 **`-75.7%`**、E2E P99 降 **`-67.1%`**；
  - **防抖验证**：队列与 30s 窗口归零后稳态保持 4 副本 `26.66s` 不抖动，随后严格按 `4 → 3 → 2 → 1` 步进缩容；
  - **Fail-Safe 容错**：停掉 Prometheus 后 `23.14s` 激活 `FALLBACK=True`、`36.55s` 保底拉起至 `2/2` Ready，恢复后 `24.75s` 自动自愈。
- **已验证的风险边界**：
  - 初期 `0–20s` 仍存在 `25.76%` 503 且全程 E2E P99 仍为 `10.9s`；当模拟 `15s` 慢冷启动时，HPA 出现 `3` 个未就绪 Pod 堆积的冷启动过冲、35s 突发拒绝率达 `28.45%`——印证**扩容绝不能替代入口准入限流**；
  - 本实验验证的是 Mock 控制面弹性闭环，不代表真实多 GPU 推理吞吐收益。
