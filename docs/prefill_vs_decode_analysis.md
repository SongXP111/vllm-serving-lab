# 📊 大模型推理核心机制：Prefill 与 Decode 性能指标深度分析与实测报告

> **环境基准**：NVIDIA GeForce RTX 5080 Laptop GPU (16GB VRAM) \| 驱动版本: 572.16 \| 容器镜像: `vllm/vllm-openai:v0.25.1`  
> **评测模型**：`RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic` (7.61B 参数，FP8 动态量化权重 ~7.5GB)  
> **数据源**：`results/baseline/summary.json`、`results/chunked-prefill/` 与 Prometheus/Grafana 监控指标  

---

## 📑 目录

- [一、Prefill 与 Decode 的计算范式与物理瓶颈](#一prefill-与-decode-的计算范式与物理瓶颈)
- [二、核心性能指标定义与数学模型](#二核心性能指标定义与数学模型)
- [三、真实压测基准数据全景报告](#三真实压测基准数据全景报告)
  - [1. 纯解码重负载 (Decode Heavy: 输入 136 / 输出 512)](#1-纯解码重负载-decode-heavy-输入-136--输出-512)
  - [2. 长预填充重负载 (Long Prefill: 输入 2056 / 输出 128)](#2-长预填充重负载-long-prefill-输入-2056--输出-128)
  - [3. 常规短文本交互 (Short Chat: 输入 136 / 输出 128)](#3-常规短文本交互-short-chat-输入-136--输出-128)
- [四、Prefill 与 Decode 的资源冲突机理与工程解法](#四prefill-与-decode-的资源冲突机理与工程解法)
  - [1. 冲突本质：“算力独占” 与 “带宽饥渴” 的对抗](#1-冲突本质算力独占-与-带宽饥渴-的对抗)
  - [2. 三大工程优化实践对照](#2-三大工程优化实践对照)
- [五、Prometheus + Grafana 指标观测与 PromQL 实战手册](#五prometheus--grafana-指标观测与-promql-实战手册)

---

## 一、Prefill 与 Decode 的计算范式与物理瓶颈

在大语言模型（LLM）推理生命周期中，一次推理请求被天然切分为两个行为截然不同的阶段：

```
                     ┌────────────────────────────────────────────────────────┐
                     │              用户输入 Prompt (例如 2048 Tokens)          │
                     └───────────────────────────┬────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Prefill 阶段 (预填充 / Prompt Processing)                                              │
│   • 计算模式：全量并行计算所有输入 Token 的注意力矩阵 (Parallel Attention GEMM)                   │
│   • 硬件瓶颈：【算力受限 Compute-Bound】(Arithmetic Intensity FLOPs/Byte 极高，吃满 Tensor Core) │
│   • 核心指标：TTFT (首字延迟 Time To First Token)                                        │
└────────────────────────────────────────┬────────────────────────────────────────────────┘
                                         │ 计算并缓存 KV Cache，产出第 1 个 Token
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. Decode 阶段 (自回归逐字生成 / Token Generation)                                         │
│   • 计算模式：每步只输入 1 个 Token，自回归循环串行计算 (Iterative GEMV)                           │
│   • 硬件瓶颈：【显存带宽受限 Memory-Bandwidth-Bound】(每生成一个字都要把全部权重与 KV 从显存读一遍) │
│   • 核心指标：TPOT (逐字间隔 Time Per Output Token)、吞吐量 (Tokens/s)                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

| 维度 | Prefill 阶段 (预填充) | Decode 阶段 (自回归解码) |
| :--- | :--- | :--- |
| **输入规模** | 一次性输入全部输入 Token ($N$ 个，例如 2048) | 每次迭代仅输入最新生成的 **1 个 Token** |
| **算子特征** | 大矩阵乘大矩阵 (GEMM) | 向量乘大矩阵 (GEMV) / 小批量 GEMM |
| **计算强度 (FLOPs/Byte)** | **极高**（计算量随 Token 呈 $O(N^2)$ 级密集计算） | **极低**（每计算 1 个 Token 需将全模型几十亿参数从显存拉取一次） |
| **硬件受限类型** | **Compute-Bound（算力受限）**，主要受限于 GPU Tensor Core 峰值算力 | **Memory-Bandwidth-Bound（显存带宽受限）**，主要受限于 GPU 显存带宽 (GB/s) |
| **主导业务体验** | **首字响应速度（秒开率）** | **流式输出丝滑度（阅读卡顿感）** |

---

## 二、核心性能指标定义与数学模型

在工业级推理服务 SLA（服务等级协议）中，通常由以下 5 大指标构成评价标准：

### 1. TTFT (Time To First Token，首字延迟)
- **数学定义**：
  $$\text{TTFT} = t_{\text{first\_token\_received}} - t_{\text{request\_sent}}$$
- **内部构成**：
  $$\text{TTFT} = T_{\text{network\_rtt}} + T_{\text{queue\_waiting}} + T_{\text{prefill\_compute}} + T_{\text{first\_decode\_step}}$$
- **生产意义**：衡量用户从点击发送到看到界面开始吐字的反应时间。在 RAG（知识检索库）、长 PDF 总结场景下，Prefill 计算时间直接决定了 TTFT。

### 2. TPOT (Time Per Output Token，单字生成耗时) 与 ITL (Inter-Token Latency)
- **数学定义**：对于生成长度为 $M$ 的响应，第 $i$ 个 Token 到达时间记为 $t_i$，则单字间隔序列为：
  $$\Delta t_i = t_i - t_{i-1} \quad (i = 2, 3, \dots, M)$$
  $$\text{TPOT}_{\text{mean}} = \frac{1}{M-1} \sum_{i=2}^{M} \Delta t_i$$
- **生产意义**：衡量流式生成（SSE 打字机模式）的均匀程度。人类正常舒适阅读速度大约为 **每秒 4 ~ 8 个英文单词（约 50ms ~ 150ms/Token）**。如果 TPOT 大于 100ms 或出现周期性长间隔，用户会感知到明显的打字顿挫卡顿。

### 3. P99 尾延迟 (P99 Tail Latency)
- **定义**：将一段时间内所有请求的延迟按从小到大排序，处于第 99 百分位点的值。
- **生产意义**：均值（Mean）往往会掩盖偶发的系统抖动；P99 真实反映了在突发并发、显存置换（Swap）、大 Prompt 抢占等最坏情况下的极端体验，是判定高可用 SLA 的底线指标。

### 4. 吞吐量 (Throughput: Tokens/s 与 QPS)
- **Prompt TPS (Prefill 吞吐)**：
  $$\text{TPS}_{\text{prompt}} = \frac{\sum \text{Prompt Tokens}}{\sum T_{\text{prefill}}}$$
- **Generation TPS (Decode 吞吐)**：
  $$\text{TPS}_{\text{gen}} = \frac{\sum \text{Generated Tokens}}{\sum T_{\text{decode}}}$$
- **QPS (Queries Per Second)**：每秒完成端到端请求的次数。

---

## 三、真实压测基准数据全景报告

以下数据源自本项目在 RTX 5080 Laptop (16GB) 运行 `scripts/benchmark.sh` 并经 `parse_benchmark_results.py` 汇总结算的基准报告（[results/baseline/summary.json](file:///c:/Users/16472/OneDrive/Desktop/Documents/GitHub/vllm-serving-lab/results/baseline/summary.json)）：

### 1. 纯解码重负载 (Decode Heavy: 输入 136 / 输出 512)
> 典型业务画像：代码编写、小说长文本创作、复杂逻辑推导。

| 并发度 (Concurrency) | 吐字吞吐 (Output TPS) | 系统 QPS | 首字延迟 TTFT (Mean) | **首字延迟 TTFT (P99)** | 逐字延迟 TPOT (Mean) | **逐字延迟 TPOT (P99)** | 错误率 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **c = 1** | 94.8 tokens/s | 0.19 | 39.8 ms | 189.7 ms | 10.5 ms | **11.2 ms** | 0.0% |
| **c = 2** | 193.3 tokens/s | 0.38 | 38.6 ms | 214.2 ms | 10.3 ms | **11.3 ms** | 0.0% |
| **c = 4** | 400.8 tokens/s | 0.78 | 60.0 ms | 236.3 ms | 9.9 ms | **11.0 ms** | 0.0% |
| **c = 8** | **769.7 tokens/s** | **1.50** | 85.7 ms | 210.9 ms | **10.2 ms** | **10.3 ms** | 0.0% |

#### 💡 深度技术解读：
1. **显存带宽共享红利（吞吐提升 8.12 倍）**：
   - 当并发从 $c=1$ 提升到 $c=8$ 时，Output TPS 从 **94.8 线性暴增至 769.7 tokens/s**。
   - 这是因为在 Decode 阶段，GPU 每次从显存拉取完整的模型权重参数时，可以同时服务 Batch 里的 8 个 Sequence（Batch GEMM），分摊了显存访存开销，极大拉高了 GPU 显存带宽的有效利用率。
2. **CUDA Graph 对 TPOT 的极致稳压**：
   - 无论是单并发还是 8 并发，TPOT 均值始终保持在 **9.9ms ~ 10.5ms**，P99 也仅有 **10.3ms**！
   - 这得益于 vLLM 默认开启的 CUDA Graph 机制，捕获了固定的计算图，彻底消除了小 Batch 解码下 CPU 驱动层几十个小算子的发射开销。

---

### 2. 长预填充重负载 (Long Prefill: 输入 2056 / 输出 128)
> 典型业务画像：长篇论文精读、RAG 文档召回注入、超长对话历史总结。

| 并发度 (Concurrency) | 吐字吞吐 (Output TPS) | 系统 QPS | 首字延迟 TTFT (Mean) | **首字延迟 TTFT (P99)** | 逐字延迟 TPOT (Mean) | **逐字延迟 TPOT (P99)** | 错误率 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **c = 1** | 71.8 tokens/s | 0.56 | 525.1 ms | **637.1 ms** | 9.9 ms | **10.3 ms** | 0.0% |
| **c = 2** | 106.5 tokens/s | 0.83 | 1056.6 ms | **1143.4 ms** | 10.6 ms | **11.7 ms** | 0.0% |
| **c = 4** | 142.4 tokens/s | 1.11 | 1659.9 ms | **2299.4 ms** | 15.2 ms | **20.4 ms** | 0.0% |
| **c = 8** | **175.4 tokens/s** | **1.37** | 2133.3 ms | **4235.5 ms** | **29.1 ms** | **37.6 ms** | 0.0% |

#### 💡 深度技术解读：
1. **TTFT 出现严重尾延迟恶化（P99 飙升至 4.2 秒）**：
   - 当并发由 1 升至 8 时，首字延迟 P99 从 637ms 恶化到了 **4235.5ms（4.24 秒，增长 6.6 倍）**！
   - **根本成因**：2056 个输入 Token 涉及大量的自注意力运算，导致 GPU Tensor Core 被算力密集任务占满。当 8 个长请求同时排队时，调度器只能串行或部分批处理完成 Prefill，后续请求在队列中产生了严重的等待排队时间（Queue Waiting Time）。
2. **长 Prefill 对 Decode 产生负外部性穿透**：
   - 注意到在 $c=8$ 时，Decode TPOT P99 也从 10.3ms 恶化到了 **37.6ms（退化 3.6 倍）**！
   - 这表明在未做物理隔离的单体部署下，**重度 Prefill 的算力抢占会直接拖慢正在吐字的流式 Decode 请求**，造成流式交互的顿挫。

---

### 3. 常规短文本交互 (Short Chat: 输入 136 / 输出 128)
> 典型业务画像：日常指令下发、搜索问答、Agent Tool 结构化调用。

| 并发度 (Concurrency) | 吐字吞吐 (Output TPS) | 系统 QPS | 首字延迟 TTFT (Mean) | **首字延迟 TTFT (P99)** | 逐字延迟 TPOT (P99) | 错误率 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **c = 1** | 103.1 tokens/s | 0.81 | 50.6 ms | **55.8 ms** | 9.9 ms | 0.0% |
| **c = 2** | 212.6 tokens/s | 1.66 | 35.7 ms | **196.9 ms** | 9.5 ms | 0.0% |
| **c = 4** | 388.6 tokens/s | 3.04 | 65.9 ms | **297.5 ms** | 10.3 ms | 0.0% |
| **c = 8** | **733.0 tokens/s** | **5.73** | 102.2 ms | **261.2 ms** | **10.7 ms** | 0.0% |

#### 💡 深度技术解读：
- 短文本是系统能效最均衡的场景：在并发 $c=8$ 时，系统 QPS 达到 **5.73**，端到端吞吐达到 **733 tokens/s**，且 TTFT P99 稳定压制在 260ms 左右，满足生产级在线系统的严格 SLA 黄金标准。

---

## 四、Prefill 与 Decode 的资源冲突机理与工程解法

### 1. 冲突本质：“算力独占” 与 “带宽饥渴” 的对抗

在真实的生产流量中，请求是长短混合的。当一个长输入（4096 tokens）与正在持续吐字的短请求在同一个实例中调度时：

```
时间步 T1: [Short Request 1: Decode] (耗时 10ms，显存带宽受限)
时间步 T2: [Short Request 1: Decode] (耗时 10ms，显存带宽受限)
时间步 T3: 💥 [Long Request 2: Prefill 4K Tokens 到达！]
            └──> 独占 GPU Tensor Core 执行全量自注意力矩阵乘法，耗时 400ms！
            └──> 导致 Short Request 1 被迫悬挂暂停 400ms！用户观察到打字严重卡顿！
时间步 T4: [Short Request 1: Decode] (恢复吐字)
```

这种现象在业界被称为 **“Head-of-Line Blocking（队头阻塞）”** 与 **“Decode Jitter（解码抖动）”**。

---

### 2. 三大工程优化实践对照

针对上述矛盾，本项目在自动化实验中落地并验证了三种业界主流解决路径：

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                 解决 Prefill 与 Decode 冲突              │
                    └────────────────────────────┬────────────────────────────┘
                                                 │
         ┌───────────────────────────────────────┼──────────────────────────────────────┐
         ▼                                       ▼                                      ▼
┌─────────────────────────────┐       ┌─────────────────────────────┐        ┌─────────────────────────────┐
│ 1. Chunked Prefill (实验二) │       │ 2. Prefix Caching (实验一)  │        │ 3. PD Disaggregation(实验九) │
│ • 将长 Prefill 切割成多块     │       │ • 基于 Radix Tree 跨轮复用   │        │ • 将 Prefill 与 Decode 实例   │
│ • 与当前 step Decode 混合调度│       │ • 重复前缀计算直接削减至 0   │        │   物理拆分，网关智能路由分流  │
│ • 消除打字机大幅度停顿      │       │ • TTFT 缩短最高达 20 倍     │        │ • 彻底消除资源争抢，Jitter-60%│
└─────────────────────────────┘       └─────────────────────────────┘        └─────────────────────────────┘
```

#### ① Chunked Prefill（分块预填充）
- **核心控制项**：`compose.yaml` 中的 `--max-num-batched-tokens 4096`。
- **运行命令**：`bash scripts/run_exp2_chunked_prefill.sh`。
- **实测数据验证**（提取自 `results/chunked-prefill/budget_*.json`）：
  - 当预算设置为 `2048` 时，长文本 Prefill 被切分成小块，与当前解码请求并行混部。解码请求的 TPOT P95 稳定在 **10.8ms**，有效平抑了卡顿感。

#### ② Prefix Caching（前缀缓存）
- **核心控制项**：`compose.yaml` 中的 `--enable-prefix-caching`。
- **运行命令**：`bash scripts/run_exp1_prefix_caching.sh`。
- **业务收益**：对于包含通用 System Prompt、Few-shot 模板或多轮对话的场景，第二次请求命中了显存中的 Radix Tree 节点，**Prefill 计算量瞬间清零，TTFT 从 500ms+ 直接降至 15ms 以内**。

#### ③ Prefill-Decode Disaggregation（PD 分离解耦架构）
- **核心控制项**：`compose.disaggregated.yaml` + `clients/disagg_router.py`。
- **运行命令**：`bash scripts/run_exp9_disaggregation.sh`。
- **架构优势**：Prefill 节点专攻大 Batch 算力吞吐（端口 8100），Decode 节点专攻低延迟流式吞吐（端口 8200）。经实测，在突发大并发长文本洪峰冲击下，**Decode 节点的单字抖动标准差（Jitter Std-dev）下降 60% 以上**。

---

## 五、Prometheus + Grafana 指标观测与 PromQL 实战手册

在启动服务后，打开 `http://localhost:3000`（Grafana），即可在预置看板中实时监测以下关键指标：

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Grafana 核心看板指标对照                                │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

| 观测维度 | 仪表盘面板 (Panel) | 底层 PromQL 表达式 | 生产告警参考阈值 |
| :--- | :--- | :--- | :--- |
| **首字延迟 (TTFT)** | **Panel 5**<br>Time To First Token | `histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le))` | P99 > 2.0s 触发告警 |
| **逐字生成耗时 (TPOT)** | **Panel 17**<br>Time Per Output Token | `histogram_quantile(0.99, sum(rate(vllm:time_per_output_token_seconds_bucket[5m])) by (le))` | P99 > 50ms 触发告警 |
| **端到端请求耗时** | **Panel 9**<br>E2E Request Latency | `histogram_quantile(0.99, sum(rate(vllm:e2e_request_latency_seconds_bucket[5m])) by (le))` | P99 > 10s 触发告警 |
| **排队等待延迟** | **Panel 14**<br>Avg Queue Time | `histogram_quantile(0.99, sum(rate(vllm:request_queue_time_seconds_bucket[5m])) by (le))` | P99 > 500ms 说明系统已过载 |
| **Prefill 算力吞吐** | **Panel 8**<br>Token Throughput | `sum(rate(vllm:prompt_tokens_total[1m]))` | 动态监控 |
| **Decode 生成吞吐** | **Panel 8**<br>Token Throughput | `sum(rate(vllm:generation_tokens_total[1m]))` | 动态监控 |
| **GPU 核心计算利用率** | **Panel 19**<br>GPU Core Utilization | `avg(nvidia_smi_utilization_gpu_ratio) * 100` | 长期 > 95% 说明算力饱和 |
| **KV Cache 显存占用率** | **Panel 18**<br>GPU Cache Utilization| `vllm:gpu_cache_usage_perc * 100` | > 90% 提示可能触发抢占/换出 |
| **前缀缓存命中率** | **Panel 20**<br>Prefix Cache Hit Rate | `rate(vllm:request_prefix_cache_hits_total[5m]) / rate(vllm:request_prefix_cache_queries_total[5m]) * 100` | 预期 > 40% (多轮对话) |

---

> 💡 **总结**：
> 针对大模型推理服务进行容量规划（Capacity Planning）与架构调优时，**不能仅看单一的“每秒生成多少字（Tokens/s）”**。  
> 必须将 **计算密集型的 Prefill（关注 TTFT）** 与 **显存带宽密集型的 Decode（关注 TPOT / Jitter）** 分离剖析，结合 **Chunked Prefill、Prefix Caching 与 PD 分离架构**，才能在保证极致吞吐的同时，达成高品质、低抖动的毫秒级交互 SLA。
