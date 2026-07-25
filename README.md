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
   - 自动预置 Prometheus 数据源与 Grafana 仪表盘。
   - 实时观测 **TTFT（首字延迟）、TPOT（逐字间隔）、GPU VRAM / KV Cache 命中率、吞吐量 (Tokens/s) 与请求排队时间**。
4. 🧪 **三大深度优化实验套件**：
   - 提供完备的 automated 脚本，深度测评 **Prefix Caching（前缀缓存）、Chunked Prefill（分块预填充）与 AWQ 量化 (Quantization)** 核心技术落地收益。

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
| **vLLM Metrics** | `http://localhost:8000/metrics` | 无 | Prometheus 原始打点数据 |
| **Prometheus UI** | `http://localhost:9090` | 无 | 抓取、存储与查询时序指标 |
| **Grafana Dashboard**| `http://localhost:3000` | `admin` / `admin` (可通过 .env 修改) | 核心业务监控大屏 |

---

## 📊 一键压测与实验套件 (Benchmark & Experiments)

我们提供了一个统一的**自动化测试矩阵控制台** `scripts/run_all_benchmarks.sh`，不仅支持基线吞吐压测，还内置了三大核心大模型推理优化实验：

```bash
# 查看帮助与选项
bash scripts/run_all_benchmarks.sh --help

# 一键按顺序自动化运行所有基线压测与三大优化实验！
bash scripts/run_all_benchmarks.sh --all
```

### 🧪 实验矩阵明细

#### 1️⃣ 基线负载压测 (Baseline Workloads)
- **命令**：`bash scripts/benchmark.sh` (或 `-b`)
- **场景**：对比在不同并发度（Concurrencies = 1, 2, 4, 8）下，常规对话 (Short Chat)、长文预载 (Long Prefill) 与重度解码 (Decode Heavy) 的 QPS 与延时表现。

#### 2️⃣ 实验一：前缀缓存优化 (Prefix Caching)
- **命令**：`bash scripts/run_exp1_prefix_caching.sh` (或 `-1`)
- **核心逻辑**：对比开启与关闭 `--enable-prefix-caching` 时，长 System Prompt / Agent 历史会话场景下的首字延迟。
- **实验结论**：开启 Prefix Caching 能借助底层 Radix Tree 实现 KV Cache 零拷贝复用，将热启动 **TTFT 缩短最高达 ~20 倍**。

#### 3️⃣ 实验二：分块预填充与混合调度 (Chunked Prefill)
- **命令**：`bash scripts/run_exp2_chunked_prefill.sh` (或 `-2`)
- **核心逻辑**：通过自动化脚本测试 `max-num-batched-tokens` 在 `2048`, `4096`, `8192` 预算下，大并发流式解码 (Decode) 与突发长文预载 (Prefill) 发生碰撞时的性能制衡。
- **实验结论**：较小预算有助于降低持续生成流中的卡顿顿挫感（改善短请求 TPOT P99），较大预算则更加照顾吞吐与长文响应速度（降低长请求 TTFT）。

#### 4️⃣ 实验三：模型量化性能与显存对比 (AWQ vs BF16)
- **命令**：`bash scripts/run_exp3_quantization.sh` (或 `-3`)
- **核心逻辑**：全自动动态重启容器，横向对比 `Qwen/Qwen3-4B (BF16)`, `Qwen/Qwen3-4B-AWQ`, `Qwen/Qwen3-8B-AWQ` 的显存占用、可用 KV Cache Token 槽位总量、速度及回答质量。
- **实验结论**：**AWQ 量化的首要价值在于解决显存容量瓶颈**——大幅压缩权重显存从而释放几万乃至几十万的 KV Cache 容量；但在计算延时上，由于解量化算子开销，并不保证在所有硬件和 Batch Size 下都绝对快于 BF16。

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