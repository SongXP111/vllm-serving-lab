# Architecture Decision Records (ADR)

本目录记录 `vllm-serving-lab` 在 M1 → M3 演进过程中最核心的 4 项架构决策，按标准 **Context / Decision / Alternatives Considered / Consequences & Evidence** 格式沉淀。

| 编号 | 决策主题 | 状态 | 关联模块与证据 |
| :--- | :--- | :---: | :--- |
| [ADR-0001](0001-gateway-bounded-admission-queue.md) | 独立异步流式网关与有界准入队列（分离 `429`/`503` 与引擎调度队列） | Accepted | `gateway/admission.py`, `gateway/app.py`, `docs/slo_benchmark_regression_report.md` |
| [ADR-0002](0002-cache-load-aware-routing-and-metric-separation.md) | 带热点衰减的 `cache+load-aware` 路由及估计/真实缓存指标严格隔离 | Accepted | `gateway/router.py`, `gateway/registry.py`, `docs/gateway.md` |
| [ADR-0003](0003-k8s-headless-discovery-probes-and-graceful-drain.md) | Kubernetes Headless Service DNS 发现、三探针隔离与两阶段 `preStop` 排空 | Accepted | `gateway/discovery.py`, `k8s/mock-vllm.yaml`, `docs/k8s_lifecycle_report.md` |
| [ADR-0004](0004-keda-queue-driven-autoscaling-and-dynamic-capacity.md) | 基于 `gateway_queue_depth` 的 KEDA 扩缩容、动态并发上限联动与 Fail-Safe Fallback | Accepted | `k8s/keda-scaledobject.yaml`, `gateway/admission.py`, `docs/keda_autoscaling_report.md` |
