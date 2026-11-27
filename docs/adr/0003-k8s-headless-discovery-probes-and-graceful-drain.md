# ADR-0003: Kubernetes Headless Service DNS 发现、三探针隔离与两阶段 `preStop` 优雅下线

- **Status**: Accepted
- **Date**: 2026-11-26
- **Related Code & Evidence**: [`gateway/discovery.py`](../../gateway/discovery.py), [`mock_backend/server.py`](../../mock_backend/server.py), [`k8s/mock-vllm.yaml`](../../k8s/mock-vllm.yaml), [`docs/k8s_lifecycle_report.md`](../k8s_lifecycle_report.md), [`results/k8s/summary.json`](../../results/k8s/summary.json)

## Context

将 LLM Gateway + 多推理副本部署到 Kubernetes 后，面临三类典型有状态/长连接痛点：
1. 普通 `ClusterIP` Service 由 `kube-proxy` 在 L4 做随机分发，会直接绕过 Gateway 的 `ReplicaRegistry` 并摧毁 Prefix-Aware 缓存局部性；
2. 大模型加载权重与 CUDA Graph 冷启动耗时极长（数十秒至数分钟），且高并发满载时延迟会上升；如果探针设计混用，Kubelet 会误杀正在加载或健康满载的 Pod，引发 **级联重启雪崩**；
3. Pod 终止时 Kubernetes 异步执行 Endpoints 摘除与容器停止；实测初版单阶段 `preStop`（立刻拒新 503）+ `lifespan` 覆盖 `SIGTERM` 信号处理器，导致滚动更新出现 3 次在途 `503`（96.67%）以及旧 Pod 悬挂 45s 直到 `SIGKILL`。

## Decision

1. **Headless Service (`clusterIP: None`) + Gateway DNS 动态发现**：
   - `mock-vllm-headless` 直接返回所有 Ready Pod 的真实 IP；
   - Gateway 内 [`HeadlessDNSReplicaDiscoverer`](../../gateway/discovery.py) 每 `1.0s` 同步 DNS，新 IP 立即注册；消失的 IP 立即标为 `healthy=False`（停止派发新请求），等待其在途 SSE 归零（`inflight == 0`）后再正式 `unregister` 并清理其过期前缀缓存索引。
2. **三探针职责严格解耦**：
   - `startupProbe` (`/startupz`)：保护慢模型冷启动，通过前挂起其余探针；
   - `readinessProbe` (`/readyz`)：只摘流、绝不重启容器（未加载完、排空中或手动隔离返回 `503`）；
   - `livenessProbe` (`/healthz`)：极轻量进程保活，**严禁与推理队列或响应延迟耦合**。
3. **两阶段 `preStop` + 原生 Uvicorn `SIGTERM` 流式优雅下线**：
   - Phase 1：`preStop` 调用 `POST /admin/drain?reject_new=false`（立刻让 `/readyz → 503` 触发 Endpoints/CoreDNS 摘除，同时在 `sleep 4` 传播窗口内继续服务已在途的尾单，消除异步 DNS 竞态）；
   - Phase 2：4s 后调用 `POST /admin/drain?reject_new=true` 并退出 `preStop`，由 Kubelet 发送 `SIGTERM` 给 PID 1（保留 Uvicorn 原生 `--timeout-graceful-shutdown 35`，**不覆盖 `SIGTERM` 处理器**），待在途 SSE 吐完 `data: [DONE]` 后以退出码 `0` 干净退出。

## Alternatives Considered

- **方案 A：Gateway 直接转发到标准 `ClusterIP` Service**  
  *放弃原因*：失去单 Pod 可见性，Prefix-Aware Routing 与单副本熔断全部失效。
- **方案 B：使用 Kubernetes API Watch `EndpointSlice` 替代 Headless DNS 轮询**  
  *放弃原因*：需额外配置 `ServiceAccount`/`RBAC` 与 K8s 客户端依赖；Headless DNS 轮询零 K8s RBAC 耦合、1s 级收敛，配合两阶段 `preStop` 4s 宽限期已实测达成 100% 零丢包滚动发布（P1 可进一步升级为 Watch 以降低收敛延迟至百毫秒级）。

## Consequences & Evidence

- **正向收益（本地 `kind` 集群三大实时演练铁证）**：
  - **Drill 1（强杀后端 Pod）**：Endpoints `0.67s` 摘除、`6.56s` 自愈恢复至 3 Ready，流量 `41/41 (100.0%)` 成功；
  - **Drill 2（滚动发布 `v1 → v2`）**：由初版 `96.67%` 提升至 **`88/88 (100.0%)` 零丢包零断流**，并实证捕捉到 `marked_unready → 1s 后流排空再 unregistered`；
  - **Drill 3（12s 冷启动 + Readiness 隔离）**：冷启动全程 `restarts = 0`；手动摘流 `3.08s` 生效、隔离期 `0` 命中、`0` 容器重启、`1.55s` 自动回流。
- **代价与约束**：
  - 每个终止中的 Pod 会多停留 `4s` 宽限期；`terminationGracePeriodSeconds` 必须大于 `preStop 宽限期 + 最长单个 SSE 请求容忍耗时`。
