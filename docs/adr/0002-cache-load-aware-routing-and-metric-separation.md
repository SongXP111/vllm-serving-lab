# ADR-0002: 带热点衰减的 `cache+load-aware` 路由及估计/真实缓存指标严格隔离

- **Status**: Accepted
- **Date**: 2026-11-26
- **Related Code & Evidence**: [`gateway/router.py`](../../gateway/router.py), [`gateway/registry.py`](../../gateway/registry.py), [`docs/gateway.md`](../gateway.md), [`results/gateway/m2_mock_policy_comparison.json`](../../results/gateway/m2_mock_policy_comparison.json), [`results/gateway/m2_real_vllm_verification.json`](../../results/gateway/m2_real_vllm_verification.json)

## Context

在多副本 vLLM 集群中，开启 Automatic Prefix Caching（APC）后，实例选择面临一个核心工程冲突：
1. 传统无状态策略（`random` / `least-load`）将同一会话或公共 System Prompt 打散到多台副本，导致单机 LRU 不断相互驱逐，集群有效 KV Cache 容量退化为 $1 \times C$ 而非 $N \times C$；
2. 纯 `prefix-aware` 策略在热点倾斜（Hot Prefix Skew，大量并发共享同一长前缀）时会把 100% 流量砸向单台热机，导致热机队列爆炸而其余副本完全空转。
3. 同时，网关侧无法以零延迟同步每张 GPU 内部实时的 Radix Tree 块表，只能基于历史路由记录做**近似估计**。

## Decision

1. **统一 `Filter → Score → Pick` 路由框架与 4 种可切换策略**：支持 `random`、`least-load`、`prefix-aware` 与默认推荐策略 **`cache+load-aware`**，所有策略可通过配置或 `/admin/routing/policy` 热切换，便于同 Workload 对照评估。
2. **链式伪 Token 块哈希 + 热点衰减（Hotspot Damping）**：
   - 使用 16-token 窗口链式哈希（`compute_prefix_block_hashes`）在网关热路径轻量估算前缀亲和度（`EstimatedPrefixCacheIndex`，有界 LRU）；
   - `cache+load-aware` 综合打分 $w_{\text{prefix}} \cdot S_{\text{prefix}} + w_{\text{load}} \cdot S_{\text{load}}$；当热机负载超过 `hotspot_load_threshold (0.75)` 或 KV 占满至 `kv_hotspot_threshold (0.85)` 时衰减其前缀得分，主动将超额并发溢出至空闲副本（既止损排队，又顺势把热门前缀复制到第二台副本实现水平扩容）。
3. **强制指标语义隔离（红线原则）**：
   - 网关侧估计指标明确命名为 `X-Estimated-Prefix-Affinity` 与 `gateway_router_estimated_prefix_affinity_avg`；
   - 真实缓存命中率必须通过抓取后端真实 vLLM `/metrics`（`vllm:prefix_cache_hits_total / queries_total`）独立记录在 `gateway_replica_observed_vllm_prefix_cache_hit_rate`，两者绝不混用。

## Alternatives Considered

- **方案 A：仅使用 `least-load` 负载均衡**  
  *放弃原因*：M2 多轮共享前缀场景实测命中率仅 `16.67%`，TTFT P50 停留在冷启动 `109.0ms`。
- **方案 B：使用纯 `prefix-aware` 粘性哈希**  
  *放弃原因*：M2 热点倾斜场景下 8/8 并发全部堆在一台机器，Queue/TTFT P99 恶化至 `109.0ms`（最差），相对吞吐最差。
- **方案 C：在网关引入完整 HuggingFace Tokenizer 与实时 KV 块同步**  
  *放弃原因*：网关热路径 CPU/内存开销过高，且真实引擎 KV 驱逐事件高频变化；轻量伪 Token 链式哈希 + 异步抓取真实 `/metrics` 校验在工程复杂度与准确度间更优。

## Consequences & Evidence

- **正向收益**：
  - M2 多副本 Mock 对比（场景 1）：`cache+load-aware` 将前缀命中率从 `16.67% → 75.00%`，TTFT P50 缩短 **`6.8×`**（`109ms → 16ms`）；热点倾斜（场景 3）下相比纯 `prefix-aware` 将 Queue/TTFT P99 降低 **`56.9%`**（`109ms → 47ms`）；
  - RTX 5080 单卡真机校验：约 2,185 Token 共享前缀命中真实 vLLM KV Cache 后 TTFT 缩短 **`21.04×`**（`1,682.3ms → 79.97ms`），且网关估计命中均值（`74.78%`）与真实 vLLM `/metrics` 观测值（`74.14%`）误差仅 **`0.64%`**。
- **代价与约束**：
  - Mock 多副本实验仅用于路由算法正确性与相对权衡验证，不代表真实多 GPU 吞吐；若多副本网关并存，估计索引在各网关实例间独立维护，收敛速度略慢于集中式索引。
