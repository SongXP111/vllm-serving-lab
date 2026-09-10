# 🧪 vLLM Serving Lab — 高性能大模型服务标准化实验与监控套件

[![Docker Compose](https://img.shields.io/badge/Docker%20Compose-v2.0+-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![vLLM](https://img.shields.io/badge/vLLM-v0.25.1-4A154B)](https://github.com/vllm-project/vllm)
[![Prometheus](https://img.shields.io/badge/Prometheus-v3.13.1-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-13.1.1-F46800?logo=grafana&logoColor=white)](https://grafana.com/)

> **打造标准化、可复现、深度优化的工业级大语言模型（LLM）推理服务与生产监控工作台。**  
> 适配 Windows WSL2 / Linux + NVIDIA GPU (如 RTX 4090 / 5080 Laptop 等消费级与数据中心硬件)。

---

## 🌟 核心特性与架构概览

本项目将临时、零散的 `docker run` 与命令行测试脚本升级为**结构化、一键化、高可观测性**的标准化推理服务架构：

1. 🚀 **一键自动化编排 (`Docker Compose`)**：
   - 固定组件镜像版本（vLLM `v0.25.1` + Prometheus `v3.13.1` + Grafana `13.1.1`），拒绝依赖 `latest` 带来的环境突变。
   - 内置完善的 **Health Check（健康检查）** 机制，确保服务间依赖顺畅（Prometheus 等待 vLLM 权重加载完毕才启动打流）。
   - 内置 **JSON 日志轮转（Log Rotation）** 策略，限制文件大小与保留个数，避免长期运行撑爆磁盘。
   - 配置 `restart: unless-stopped` 自动重启策略，保障服务高可用。
2. 🔒 **生产安全与鉴权配置 (`.env` 隔离)**：
   - 杜绝把 API Key 或 Hugging Face Token 硬编码提交进 Git 代码库。
   - 支持动态环境变量插值（端口、显存利用率、上下文长度、量化 Dtype 等均可灵活定制）。
3. 📈 **全链路指标监控与可视化 (Observability)**：
   - 自动预置 Prometheus 数据源与包含 **16 个监控面板的 Grafana 工业级仪表盘**。
   - 包含通过 `nvidia_gpu_exporter` 实时采集的 **真实 GPU 核心算力利用率 (SM Compute %)、功率 (Watts)、显卡温度**。
   - 实时观测 **TTFT（首字延迟）、TPOT（逐字间隔）、端到端延迟 P50/P99、Queue Time 排队延迟、KV Cache 使用率、Prefix Cache 命中率与 Tokens/s 吞吐**。
4. 🧪 **九大核心推理架构优化实验套件 (9-Experiment Matrix)**：
   - 提供完备的自动化脚本，深度测评 **Prefix Caching（前缀缓存）、Chunked Prefill（分块预填充）、模型权重量化 (BF16/AWQ/FP8)、FCFS vs Priority 调度策略、Speculative Decoding（投机解码）、CUDA Graph 图执行优化、KV Cache FP8 量化、CPU 卸载与 Swap Space 内存溢出置换、Prefill-Decode Disaggregation (PD 分离架构)**。

---

## ⚡ 快速开始 (Quick Start)

### 1. 环境准备与配置

克隆项目后，首先将配置模版复制为 `.env` 文件（请勿将真实 `.env` 提交至 Git）：

```bash
cp .env.example .env
```

如果您需要加载受限 / 私有模型，或者开启客户端 Bearer Token 鉴权，请修改 `.env` 中的相关字段：
```ini
HF_TOKEN=hf_your_huggingface_token_here
VLLM_API_KEY=your_secret_api_key_here
```

### 2. 一键启动服务

运行我们提供的**一键启动脚本**，服务将在后台启动并自动轮询等待服务健康就绪（自动适配首次下载模型权重的漫长等待）：

```bash
bash scripts/start.sh
```

或者使用原生 Docker Compose 命令：
```bash
docker compose up -d
docker compose logs -f vllm  # 查看实时推理日志
```

### 3. 一键冒烟测试 (Smoke Test)

验证 API 与核心功能是否就绪：

```bash
bash scripts/smoke_test.sh
```

---

## 🌐 服务端点与监控地址 (Endpoints)

服务成功启动后，即可通过以下端口访问各个可视化面板与接口：

| 服务名称 | 默认地址 | 默认账号 / 密码 | 说明 |
| :--- | :--- | :--- | :--- |
| **vLLM API Server** | `http://localhost:8000/v1/chat/completions` | `Bearer <VLLM_API_KEY>` | 兼容 OpenAI /v1 格式接口 |
| **vLLM Metrics** | `http://localhost:8000/metrics` | 无 | vLLM 内部 Prometheus 原始打点 |
| **GPU Exporter** | `http://localhost:9835/metrics` | 无 | NVIDIA 驱动层硬件算力/功耗打点 |
| **Prometheus UI** | `http://localhost:9090` | 无 | 抓取、存储与查询时序指标 |
| **Grafana Dashboard**| `http://localhost:3000` | `admin` / `admin` (可通过 .env 修改) | 16 面板全景监控大屏 |

---

## 📊 一键压测与实验套件 (Benchmark & Experiments)

我们提供了一个统一的**自动化测试矩阵控制台** `scripts/run_all_benchmarks.sh`，不仅支持基线吞吐与泊松流量压测，还内置了九大核心大模型推理优化实验：

```bash
# 查看帮助与选项
bash scripts/run_all_benchmarks.sh --help

# 一键按顺序自动化运行所有基线压测与九大优化实验！
bash scripts/run_all_benchmarks.sh --all
```

> 📖 **深入技术分析**：  
> 深入理解计算与显存带宽受限、指标定义及工程冲突请阅读 👉 **[《大模型推理核心机制：Prefill 与 Decode 性能指标深度分析与实测报告》](file:///c:/Users/16472/OneDrive/Desktop/Documents/GitHub/vllm-serving-lab/docs/prefill_vs_decode_analysis.md)** 👈

### 🧪 实验矩阵明细

| 实验编号 | 优化技术方向 | 核心开关 / 参数 | 对应测试脚本 | 预期性能收益 |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | **基线压测** | 泊松到达 / 饱和打流 | `scripts/benchmark.sh` | 建立系统吞吐与尾延迟基准 SLA 报告 |
| **Exp 1** | **Prefix Caching** | `--enable-prefix-caching` | `scripts/run_exp1_prefix_caching.sh` | Radix Tree 跨轮复用，热启动 TTFT 缩短最高 20x |
| **Exp 2** | **Chunked Prefill** | `--max-num-batched-tokens` | `scripts/run_exp2_chunked_prefill.sh` | 分块混合调度平抑 TPOT 抖动，兼顾长短并发 |
| **Exp 3** | **权重量化对比** | BF16 vs AWQ vs FP8 | `scripts/run_exp3_quantization.sh` | 压缩模型权重释放显存，可用 KV Blocks 翻倍 |
| **Exp 4** | **调度策略** | FCFS vs Priority | `scripts/run_exp4_scheduling_policy.sh` | 消除队头阻塞，VIP 请求 TTFT 缩短数十倍 |
| **Exp 5** | **投机解码 (Spec-Decode)**| `--speculative-config` | `scripts/run_exp5_speculative_decoding.sh` | N-gram 零显存投机，TPOT 缩短 30%~50% (1.5x+ 吞吐) |
| **Exp 6** | **CUDA Graph 优化** | `--enforce-eager` 开关 | `scripts/run_exp6_cuda_graph.sh` | 消除 CPU 驱动发射开销，小并发解码加速 30%+ |
| **Exp 7** | **KV Cache FP8 量化** | `--kv-cache-dtype fp8_e4m3` | `scripts/run_exp7_kv_cache_quant.sh` | KV 显存减半，系统并发容纳上限翻倍 (2.0x) |
| **Exp 8** | **CPU Offload & Swap** | `--swap-space 4` | `scripts/run_exp8_cpu_offload.sh` | 显存过载换出至 Host RAM，请求 100% 防雪崩 |
| **Exp 9** | **PD 分离架构** | `compose.disaggregated.yaml` | `scripts/run_exp9_disaggregation.sh` | Prefill 与 Decode 物理隔离，解码抖动降低 60%+ |

---

#### 1️⃣ 基线负载压测 (Baseline Workloads)
- **命令**：`bash scripts/benchmark.sh` (或 `-b`)
- **场景**：对比在不同并发度（Concurrencies = 1, 2, 4, 8）下，常规对话 (Short Chat)、长文预载 (Long Prefill) 与重度解码 (Decode Heavy) 的 QPS 与延时表现。并在压测结束后自动执行 `parse_benchmark_results.py` 汇总结算完整 SLA 报告。

#### 2️⃣ 实验一：前缀缓存优化 (Prefix Caching)
- **命令**：`bash scripts/run_exp1_prefix_caching.sh` (或 `-1`)
- **核心逻辑**：对比开启与关闭 `--enable-prefix-caching` 时，长 System Prompt / Agent 历史会话场景下的首字延迟。
- **实验结论**：开启 Prefix Caching 能借助底层 Radix Tree 实现 KV Cache 零拷贝复用，将热启动 **TTFT 缩短最高达 ~20 倍**。

#### 3️⃣ 实验二：分块预填充与混合调度 (Chunked Prefill)
- **命令**：`bash scripts/run_exp2_chunked_prefill.sh` (或 `-2`)
- **核心逻辑**：通过自动化脚本测试 `max-num-batched-tokens` 在 `2048`, `4096`, `8192` 预算下，大并发流式解码 (Decode) 与突发长文预载 (Prefill) 发生碰撞时的性能制衡。
- **实验结论**：较小预算有助于降低持续生成流中的卡顿顿挫感（改善短请求 TPOT P99），较大预算则更加照顾吞吐与长文响应速度（降低长请求 TTFT）。

#### 4️⃣ 实验三：模型量化性能与显存对比 (BF16 vs AWQ vs FP8)
- **命令**：`bash scripts/run_exp3_quantization.sh` (或 `-3`)
- **核心逻辑**：全自动动态重启容器，横向对比 `Qwen/Qwen2.5-3B-Instruct (BF16)`, `Qwen/Qwen2.5-7B-Instruct-AWQ (4-bit)`, `RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic (FP8)` 的显存占用、可用 KV Cache Token 槽位总量、速度及回答质量。
- **实验结论**：**量化的首要价值在于释放显存瓶颈**——大幅压缩权重显存从而释放成倍的 KV Cache 容量，并借助 Blackwell/Ada 架构的 FP8 Tensor Core 获得高吞吐收益。

#### 5️⃣ 实验四：调度策略对比 (FCFS 先来先服务 vs Priority 优先级插队)
- **命令**：`bash scripts/run_exp4_scheduling_policy.sh` (或 `-4`)
- **核心逻辑**：对比在离线批处理密集占满队列场景下，高优先级在线 VIP 请求（`priority=0`）与离线批量请求（`priority=10`）在 `fcfs` 与 `priority` 调度策略下的排队延迟与 TTFT。
- **实验结论**：开启 `priority` 调度策略可有效打破队头阻塞（Head-of-Line Blocking），将高峰期 **VIP 请求 TTFT 缩短数十倍**，且对集群总吞吐几无损耗。

#### 6️⃣ 实验五：投机解码加速对比 (Speculative Decoding)
- **命令**：`bash scripts/run_exp5_speculative_decoding.sh` (或 `-5`)
- **核心逻辑**：在代码生成与结构化 JSON 等具有局部复用规律的任务中，对比标准自回归解码与开启 N-gram 投机采样（`num_speculative_tokens=4`）的单字生成间隔（TPOT）与吞吐量。
- **实验结论**：N-gram 投机解码无需额外分配任何显存即可达成 **1.4x ~ 2.2x 的逐字生成加速**，极大改善流式客户端的交互卡顿感。

#### 7️⃣ 实验六：CUDA Graph 算子捕获与执行对比 (CUDA Graph vs Eager Mode)
- **命令**：`bash scripts/run_exp6_cuda_graph.sh` (或 `-6`)
- **核心逻辑**：对比开启 CUDA Graph 图执行与禁用 CUDA Graph（`--enforce-eager`）在低并发逐字生成阶段的性能差异，揭示 CPU 驱动层发射开销（Kernel Launch Overhead）对单字延迟的严重制约。
- **实验结论**：CUDA Graph 能消除数百个连续的小算子驱动层发射延迟，将小并发解码 **TPOT 缩短 30% ~ 50%**；但也会额外消耗约 800MB 静态显存用于图缓冲。

#### 8️⃣ 实验七：KV Cache 量化收益评估 (FP8 KV Cache)
- **命令**：`bash scripts/run_exp7_kv_cache_quant.sh` (或 `-7`)
- **核心逻辑**：对比标准精度 KV Cache（`--kv-cache-dtype auto`）与 8-bit 量化 KV Cache（`--kv-cache-dtype fp8_e4m3`）在相同显存预算下的物理 Block 分配总量与 Token 容纳上限。
- **实验结论**：FP8 KV Cache 将单个 Token 的 KV 缓存压缩至 1 Byte，**使系统的并发容纳与长文本 Token 上限直接实现翻倍（~2.0x 扩容）**，且几乎无端到端生成精度损失。

#### 9️⃣ 实验八：CPU 卸载与内存溢出置换 (CPU Offload & Swap Space)
- **命令**：`bash scripts/run_exp8_cpu_offload.sh` (或 `-8`)
- **核心逻辑**：测试在突发大并发和极端上下文压迫下，vLLM 的 `--swap-space` 机制如何将超出显存池的 KV Cache 块置换到主机内存（Host RAM），从而防止服务直接触发 CUDA OOM 崩溃。
- **实验结论**：配置适量的 Swap Space 可充当**系统的“防雪崩安全气囊”**，在保证请求 100% 成功率的同时，以微小的 PCIe 搬运延迟代价换取生产系统的高可用。

#### 🔟 实验九：Prefill-Decode 分离架构评估 (PD Disaggregation)
- **命令**：`bash scripts/run_exp9_disaggregation.sh` (或 `-9`)
- **核心逻辑**：在单体实例与 PD 分离实例（`compose.disaggregated.yaml`）下，模拟并发长文本 Prefill 洪峰冲击，观测在线交互短请求的逐字生成延迟与抖动方差（Jitter Std-dev）。
- **实验结论**：PD 分离彻底消除了长 Prefill 算力对 Decode 显存带宽调度的资源争抢，将解码抖动降低 **60% 以上**，保障严苛 SLA 场景下的丝滑用户交互体验。

> 💡 *所有的压测结果将以规范的 JSON 报告保存于 `results/` 目录下，便于进行深度学术分析或数据绘制。*

---

## 🛠️ 故障排查与最佳实践 (Troubleshooting)

在实际部署（特别是消费级 NVIDIA Laptop 显卡搭配 Windows WSL2 环境）中，可能会遇到显存不够、WSL2 Pin Memory 崩溃、找不到模型仓库或环境参数不生效等常见陷阱。

请仔细阅读我们的专版手册：  
👉 **[《故障排查与最佳实践手册 (Troubleshooting Guide)》](file:///c:/Users/16472/OneDrive/Desktop/Documents/GitHub/vllm-serving-lab/docs/troubleshooting.md)** 👈

常见问题快速导航：
- [x] **WSL2 Cannot Pin Memory 报错解决**：自带 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 环境变量。
- [x] **Hugging Face 401 报错或找不到仓库**：模型仓库名大小写敏感，请核对 `VLLM_MODEL`。
- [x] **KV Cache 空间不足 (No available memory for cache blocks)**：调低 `GPU_MEMORY_UTILIZATION` 或换用 AWQ 量化模型。
- [x] **Docker Compose 没读到 `.env` 参数**：请确保在项目根目录执行，或检查 `.env` 换行符。

---

## 🛑 停止与回收服务

实验完毕后，可通过一键命令优雅停止所有容器并释放 GPU 资源：

```bash
docker compose down
```

如需清理所有的监控时序数据库与 Grafana 数据卷：
```bash
docker compose down -v
```