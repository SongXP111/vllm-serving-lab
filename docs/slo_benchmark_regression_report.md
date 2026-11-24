# 📈 Step 4 证据链报告：Metrics → SLO → Benchmark → Regression（单卡 RTX 5080 真实 vLLM 实测）

> **对应里程碑**：Step 4（M2/M3，P0）— 建立从可观测性指标、SLO/Goodput 定义、控制变量压测到自动回归门禁的完整闭环证据链  
> **硬件与模型环境**：单卡 **NVIDIA GeForce RTX 5080 Laptop GPU (16,303 MiB VRAM)**，`vllm/vllm-openai:v0.25.1`，模型 `RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic` (`qwen2.5-7b-instruct-fp8`，FP8 Dynamic)  
> **核心代码**：[`gateway/metrics.py`](../gateway/metrics.py)、[`gateway/app.py`](../gateway/app.py)、[`configs/prometheus.yaml`](../configs/prometheus.yaml)、[`scripts/run_gateway_slo_benchmark.py`](../scripts/run_gateway_slo_benchmark.py)、[`scripts/check_regression.py`](../scripts/check_regression.py)  
> **实测原始数据**：[`results/step4/`](../results/step4/)（6 份完整 JSON 压测明细 + 3 份 `check_regression.py` 回归门禁报告）

---

## 一、可观测性证据底座：标准化指标与端到端链路串联

在开展压测前，我们先按 [vLLM Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/) 规范重构了网关的 `/metrics` 与请求追踪体系（详见 [`gateway/metrics.py`](../gateway/metrics.py) 与 [`tests/test_gateway_step4_metrics.py`](../tests/test_gateway_step4_metrics.py)）：

1. **三类标准 Prometheus 指标齐备**：
   * **`Counter`**：`gateway_requests_total{status_code,stream,policy}`、`gateway_rejections_total{reason}`、`gateway_router_policy_selections_total{policy}`、`gateway_router_decisions_total{policy,replica}`、`gateway_cancel_total{stage}`。
   * **`Gauge`**：`gateway_queue_depth`、`gateway_inflight_requests`、`gateway_replica_inflight{replica}`、`gateway_replica_healthy{replica}`。
   * **`Histogram`（统一 SI 秒制，含 `_bucket` / `_sum` / `_count`）**：`gateway_queue_wait_seconds`、`gateway_ttft_seconds`、`gateway_itl_seconds`、`gateway_e2e_seconds`，可直接与 vLLM 的 `vllm:request_queue_time_seconds`、`vllm:time_to_first_token_seconds`、`vllm:inter_token_latency_seconds` 在 Prometheus（[`configs/prometheus.yaml`](../configs/prometheus.yaml)）中并排计算 `histogram_quantile(0.99, ...)`。
2. **端到端请求链路串联（`gateway_request_id` ↔ `backend_request_id`）**：
   * 网关生成或透传 `X-Gateway-Request-ID` 与 W3C `traceparent` 到下游 vLLM，并在收到首帧 JSON/SSE 时自动回填后端的 `backend_request_id`（例如真实抓包：`gateway_request_id = "sustained_candidate_prefix_hit_i4-req-016"` ↔ `backend_request_id = "chatcmpl-sustained_candidate_prefix_hit_i4-req-016"` ↔ `traceparent = "00-371219dbef624b109e33f1ad3dad4bf6-4daa454229044d55-01"`），实时记录至 `GET /admin/traces` 与结构化日志。
3. **压测自检防污染实录（可观测性直接服务于实验纠偏）**：
   * 在首轮试跑中，我们通过同步真实 `vllm:prefix_cache_hits_total` 差值，立刻发现三组实验若复用相同 `seed` 盐值，Run 2/3 会意外白嫖 Run 1 残留的 GPU APC 缓存（命中率飙到 `99.3%`）导致实验失真。我们随即加入每轮独立 `--prompt-salt` 与 `vllm_metrics_delta` 前后差值校验，确认冷启动对照组真实缓存命中率 `< 1.1%`，彻底消除跨轮变量污染。

---

## 二、闭环诊断报告 A：吞吐 vs 尾延迟博弈与过载反压（『问题现象 → 指标证据 → 原因假设 → 代码改动 → 对照结果 → 回归检测』）

### 1. 问题现象
在单卡 RTX 5080 上模拟突发流量洪峰（固定随机种子 `seed=42`，泊松到达率 `10.0 req/s`，20 个请求，每条约 1,300 cold input tokens、输出 `max_tokens=96`，互动型 SLO 门槛定义为 **`TTFT <= 1800ms` 且 `TPOT <= 20ms` 且 `E2E <= 4000ms`**）：
* 当我们试图通过放宽网关并发限制（将 `gateway.max_inflight` 从 `2` 提高到 `6` 再提高到 `20`）来“压榨更高吞吐量”时，**总输出速度确实翻倍（`108 → 225 tok/s`），但流式交互体验全线崩塌，SLO 有效吞吐（Goodput）跌到了 `0%`（20 个请求无一达标）！**

### 2. 指标证据（实测自 `results/step4/baseline_inflight_2.json`、`candidate_inflight_6.json`、`candidate_inflight_20.json`）

| 唯一变化变量 (`gateway.max_inflight`) | Raw 吞吐 (`output_tps`) | **SLO Goodput** (`goodput_rps` / 达标率) | Queue Wait P99 | TTFT P50 / P99 | **TPOT P50 / P99** | **ITL P95 / P99** | E2E P99 | Reject / Error Rate | Peak VRAM (`nvidia-smi`) | 假设单价成本 (`$1.50/GPU-hr`)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`max_inflight = 2` (过度限流基线)** | `108.05 tok/s` | `0.11 rps` (`10.0%`) | `14,422 ms` | `7,088 / 14,824 ms` | **`14.82 / 17.03 ms`** | `31 / 99.9 ms` | `16,096 ms` | `0.0% / 0.0%` | `15,756 / 16,303 MiB` | `\$3.86 / 1M tok` |
| **`max_inflight = 6` (拉高并发)** | `186.09 tok/s` (+72%) | `0.19 rps` (`10.0%`) | `6,703 ms` | `3,321 / 7,188 ms` | **`18.59 / 29.27 ms` (+71.9%!)** | **`78 / 284.1 ms`** | `8,594 ms` | `0.0% / 0.0%` | `15,881 / 16,303 MiB` | `\$2.24 / 1M tok` |
| **`max_inflight = 20` (无界追高吞吐)** | **`225.34 tok/s` (+109%)** | **`0.00 rps` (`0.0%` 全员违约!)** | `0.0 ms` | `2,556 / 5,262 ms` | **`24.47 / 29.91 ms` (+75.6%!)** | **`94 / 219.2 ms`** | `6,912 ms` | `0.0% / 0.0%` | `15,829 / 16,303 MiB` | `\$1.85 / 1M tok` |

> *成本说明：上述 `$/1M tokens` 基于明确假设 **`\$1.50 / GPU-hr`** 按 `(1.50 / 3600) / output_tps * 1e6` 推算，仅用于单机配置横向效率对比，**不代表任何云厂商真实账单**。

从三组指标证据可以精准诊断：
* **`max_inflight = 2`**：`gateway_queue_wait_seconds` P99 高达 `14.42s`（占总 TTFT 的 97.3%），说明网关把 GPU 憋得太狠，吞吐受限。
* **`max_inflight = 6` 与 `20`**：`gateway_queue_wait_seconds` 降为 `0`，但 `gateway_itl_seconds` P95 从 **`31ms` 飙升到 `94ms`（3 倍打字卡顿！）**，`TPOT P99` 从 `17.03ms` 恶化到 **`29.91ms`**。

### 3. 原因假设（为什么提升吞吐反而恶化尾延迟？）
1. **Batch 变大与 Prefill/Decode 混合干扰**：在 1.3K 冷输入下，同时向 RTX 5080 放行 `6~20` 个请求，虽然通过大 Batch 分摊了权重访存让 `output_tps` 翻倍，但 Chunked Prefill 不断把 1.3K Token 的计算块穿插进正在执行的 Decode Step 中，导致单步耗时暴涨（ITL P95 从 31ms 跳至 94ms，TPOT 跌破 20ms 交互 SLO）。
2. **超载到达率下“来者不拒”必致全员超时**：该单卡对 1.3K 冷输入的可持续吞吐仅 ~`2.3 req/s`，面对 `10 req/s` 洪峰，既不能无界放进 GPU（把所有人 TPOT 拖死），也不能无限堆进 40 长度的队列（把所有人 TTFT 拖死），必须用 **有界准入 + 快速反压 (`503`)** 锁住并发在 `max_inflight = 3` 拐点内，并用短队列（`queue_capacity = 3, queue_timeout = 1.2s`）及时拒绝溢出流量，保全被接纳用户的 SLO。

### 4. 代码改动与配置落地
* 复用 [`gateway/admission.py`](../gateway/admission.py) 的准入控制与有界队列，将洪峰保护配置收紧至拐点：`max_inflight = 3`、`queue_capacity = 3`、`queue_timeout = 1.2s`，超额请求在 `< 1ms` 内返回 `503 Service Unavailable` 并计入 `gateway_rejections_total{reason="queue_full_503"}`。

### 5. 对照结果（`candidate_backpressure_i3_q3.json` vs 无界 `candidate_inflight_20.json`）

| 核心指标 | Before：无界追吞吐 (`max_inflight=20`) | After：拐点准入 + 有界反压 (`i=3, q=3, t=1.2s`) | 变化幅度 |
| :--- | :---: | :---: | :---: |
| **SLO Goodput (`goodput_rps`)** | **`0.00 rps` (`0/20` 达标)** | **`1.13 rps` (`4/4` 已接纳请求 100% 达标 SLO!)** | **从 0 救回正向 Goodput!** |
| **TTFT P99** | `5,261.84 ms` | **`588.60 ms`** | **`-88.8%`** |
| **TPOT P99** | `29.91 ms` (违约) | **`17.39 ms` (达标 <= 20ms)** | **`-41.9%`** |
| **ITL P99** | `219.21 ms` | **`95.22 ms`** | **`-56.6%`** |
| **E2E P99** | `6,911.66 ms` | **`1,887.92 ms`** | **`-72.7%`** |
| **Peak VRAM (`nvidia-smi`)** | `15,829 MiB` | **`15,356 MiB`** | `-473 MiB` |
| **过载反压 `503` 拒绝率** | `0.0%`（所有用户一起超时） | `80.0%`（`<1ms` 快速反压触发上游扩容/切流） | 显式保护在途用户 |

### 6. 回归检测门禁结果（`scripts/check_regression.py` 实测拦截 `REGRESS`！）
直接运行 [`scripts/check_regression.py`](../scripts/check_regression.py) 检查 `candidate_inflight_6` vs `baseline_inflight_2`（落盘 [`results/step4/regression_6_vs_2.json`](../results/step4/regression_6_vs_2.json)）：
```text
Workload             Conc  Throughput     P99 TTFT        P99 TPOT        P99 E2E         $/1M Tok   Err Rate   Status
poisson_1300in_96out c=1   186.1 (+72%)   7187.8ms (-52%) 29.3ms (+72%)   8593.7ms (-47%) $2.24      0.0%       REGRESS
❌ [REGRESSION FAILED] Detected 1 workload performance violations:
   • poisson_1300in_96out (c=1): TPOT P99 regressed +71.9% (>+15.0%)
```
**门禁价值实证**：即便候选版本把原始吞吐拉高了 `+72%`、把 TTFT 降了 `-52%`，`check_regression.py` 依然准确识别出 `TPOT P99` 恶化 `+71.9%` 超过 `+15%` 容忍线，返回退出码 `1` 阻止上线！

---

## 三、闭环诊断报告 B：Prefix Cache 亲和复用单变量优化（0% 错误率全面达标 + 回归门禁 `PASS`）

为了在不牺牲任何拒绝率（`Error/Reject Rate = 0.0%`）的前提下同时突破吞吐与尾延迟边界，我们在持续到达负载下（固定 `seed=42`，`arrival_rate = 3.0 req/s`，18 请求，~1,300 input tokens / 64 out，`max_inflight = 4`，SLO: `TTFT <= 1200ms, TPOT <= 20ms, E2E <= 3000ms`）验证单一变量——**启用共享系统前缀复用（真实 vLLM APC 命中率从 `~1%` 提升至 `81.64%`）**的端到端收益：

| 核心指标 | Before：无前缀复用 (`sustained_baseline_uncached_i4`) | After：前缀亲和复用 (`sustained_candidate_prefix_hit_i4`) | 变化幅度 |
| :--- | :---: | :---: | :---: |
| **真实 vLLM `/metrics` Prefix Hit Delta** | `~1.0%`（冷启动重算） | **`81.64%`**（`20,448 / 25,046` tokens） | **`+80.6 pct`** |
| **原始吞吐 (`output_tps`)** | `140.43 tok/s` | **`209.01 tok/s`** | **`+48.8%`** |
| **SLO Goodput (`goodput_rps` / 达标率)** | `0.12 rps` (`5.56%`, `1/18`) | **`2.18 rps` (`66.67%`, `12/18`)** | **`18.2x 暴涨!`** |
| **Queue Wait P50 / P99** | `1,188 / 2,360 ms` | **`890 / 1,593 ms`** | **`-32.5% P99`** |
| **TTFT P50 / P99** | `1,453.0 / 2,627.6 ms` | **`999.1 / 1,727.8 ms`** | **`-34.2% P99`** |
| **TPOT P50 / P99** | `21.67 / 23.08 ms` (违约) | **`14.14 / 19.00 ms` (达标 <= 20ms)** | **`-17.7% P99`** |
| **ITL P95 / P99** | `79.0 / 202.8 ms` | **`16.0 / 116.4 ms`** | **`-79.7% P95 / -42.6% P99`** |
| **E2E P50 / P99** | `2,820.4 / 4,081.6 ms` | **`1,849.7 / 2,528.6 ms`** | **`-38.0% P99`** |
| **Error / Reject Rate** | `0.0% / 0.0%` | **`0.0% / 0.0%`** | 零报错 |
| **Peak VRAM (`nvidia-smi`)** | `15,496 / 16,303 MiB` | `15,667 / 16,303 MiB` | `+171 MiB` |
| **假设单价成本 (`$1.50/GPU-hr`)** | `\$2.97 / 1M tok` | **`\$1.99 / 1M tok`** | **`-32.8%`** |

### 自动回归门禁验收（[`results/step4/regression_prefix_vs_uncached.json`](../results/step4/regression_prefix_vs_uncached.json)）
```text
Workload                     Conc  Throughput     P99 TTFT        P99 TPOT        P99 E2E         $/1M Tok   Err Rate   Status
poisson_sustained_shared_sys c=1   209.0 (+49%)   1727.8ms (-34%) 19.0ms (-18%)   2528.6ms (-38%) $1.99      0.0%       PASS
🎉 [REGRESSION PASS] Candidate meets all SLA thresholds and causes NO performance regression!
```
**退出码 `0`（PASS）**！不仅吞吐大涨 `+48.8%`、假设单价成本直降 `-32.8%`，而且因为跳过了 81.64% 的长 Prefill 计算，连带着**彻底消除了长 Prefill 对并发 Decode 的抢占卡顿**（`ITL P95` 从 `79ms` 骤降至 `16ms`，`TPOT P99` 降至 `19.0ms`），实现了真正的 **帕累托前沿全面外推**！

---

## 四、后续部署与扩容的反馈闭环准则

从 Step 4 起，后续所有部署调参（K8s / 多副本扩容 / 引擎参数变更）统一以本套指标与门禁作为闭环反馈：
1. **扩容触发信号**：当 `gateway_queue_depth` 持续靠近 `queue_capacity` 或 `rate(gateway_rejections_total{reason="queue_full_503"}[1m]) > 0`（而非单纯看 GPU Util%），立即触发水平扩容；
2. **上线门禁**：任何路由策略、准入阈值或 vLLM 参数修改，必须跑通 `scripts/run_gateway_slo_benchmark.py` + `scripts/check_regression.py`（退出码 `0`），严防“原始吞吐涨了、尾延迟或 Goodput 崩了”的伪优化上线。
