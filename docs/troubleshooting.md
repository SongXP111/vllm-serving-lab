# vLLM Serving Lab - 故障排查与最佳实践手册 (Troubleshooting Guide)

在本项目（消费级硬件 NVIDIA RTX 5080 Laptop + Windows WSL2 + Docker Compose 架构）的搭建与深度优化实验中，我们总结了以下常见问题的根因与极速解决建议。

---

## 1. WSL2 显存固定报错 (CUDA Cannot Pin Memory)

### 🔴 症状现象
运行容器时 vLLM 报出 `RuntimeError: Cannot pin memory` 或系统直接报 CUDA OOM 崩溃。

### 🔍 问题根因
Windows WSL2 的虚拟内存机制在分配 Page-locked / Pinned Memory 时可能与 PyTorch 的常规分配策略发生冲突。

### 🛠️ 解决方案
1. **本项目已在 `compose.yaml` 中默认内置修复环境变量**：
   ```yaml
   environment:
     - VLLM_WSL2_ENABLE_PIN_MEMORY=1
   ```
2. 如果宿主机内存紧张，请在 Windows 用户目录下的 `.wslconfig` 中适度增大内存配置（如 `memory=16GB`）。

---

## 2. Hugging Face 401 鉴权错误与找不到模型 (Repository Not Found)

### 🔴 症状现象
容器日志显示 `httpx.HTTPStatusError: Client error '401 Unauthorized' for url '.../config.json'`，或者报 `Repository Not Found`。

### 🔍 问题根因
1. **模型仓库 ID 大小写错误**：Hugging Face 仓库名称严格区分大小写。把 `Qwen/Qwen3-8B-AWQ` 错写成小写的 `qwen3-8b-awq` 会导致 401 或找不到仓库。
2. **访问受限/私有模型未配置 Token**：例如 Llama-3 或未公开权重需要鉴权。

### 🛠️ 解决方案
1. 检查 `.env` 或 `compose.yaml` 中的 `VLLM_MODEL` 变量，确保完整且严格大小写一致（如 `Qwen/Qwen3-8B-AWQ`）。
2. 将您的 HF Token 复制到 `.env` 文件中：`HF_TOKEN=hf_xxxxxxxxxxxx`。我们的 Docker Compose 会自动将其挂载入容器环境变量。

---

## 3. vLLM 显存分配不足与启动 OOM (No Available Memory for Cache Blocks)

### 🔴 症状现象
日志提示 `ValueError: No available memory for the cache blocks. Try increasing gpu_memory_utilization...`。

### 🔍 问题根因
消费级显卡（如 16GB VRAM）除了运行模型权重外，Windows 桌面窗口管理器 (DWM) 及其它进程可能会占据一定基础显存。如果模型权重太大（如 8B BF16 占 ~16GB），会导致无法分配足够空间给 KV Cache 内存池。

### 🛠️ 解决方案
1. **采用 AWQ 4-bit 量化模型**：如实验 3 结论所示，`Qwen/Qwen3-8B-AWQ` 仅占用 ~5GB 权重显存，为 KV Cache 留下了高达 **47,616 个 tokens** 的海量容量。
2. **安全调整参数**：在 `.env` 中设置 `GPU_MEMORY_UTILIZATION=0.85` 或更保守的 `0.80`，避免试图占满 100% 显存导致宿主机系统抖动。
3. 终端执行 `nvidia-smi` 查看是否有残留的无关 Python 进程序列，及时 `kill`。

---

## 4. Prefix Caching 测试出现“假冷启动”或不提速

### 🔴 症状现象
在测试前缀缓存时，第一轮冷启动 TTFT 就只有 20~30ms，或者开启缓存后第二轮热启动仍需要 500ms+。

### 🔍 问题根因
1. **假冷启动**：上一轮压测残留的缓存未清理，导致第一轮测试直接命中了Radix Tree（基数树）。
2. **热不提速**：测试打流时随机生成的 Prompt 没有使用完全一致的公共前缀，导致 Radix Tree 节点无法重用。

### 🛠️ 解决方案
1. **注入扰动强制冷启动**：我们在 `test_prefix_caching.py` 中通过在 System Prompt 末尾追加动态时间戳 `[CACHE-RESET-UUID]`，保证了每轮新实验都是纯净的 0 命中冷启动。
2. **容器状态隔离**：在进行严谨对比实验时（参见 `scripts/run_exp1_prefix_caching.sh`），脚本会在测量前自动 `docker compose restart vllm`，确保基数树物理清空。

---

## 5. TPOT 吞吐指标失真或异常极小 (P95 比 Mean 还低)

### 🔴 症状现象
流式接口统计到的 TPOT 仅 1ms~5ms，或 P95 分位数值低于平均值，与真实感知不符。

### 🔍 问题根因
1. **未按实际 Token 数归一化**：流式通信中，某些 chunk 可能会一次性包含多个 tokens（例如思考标签 `<think>` 或标点符号短语）。如果单纯用 `网络数据包到时间隔` 等同于 `TPOT`，会导致时间被强行稀释。
2. **预填充噪音 (Prefill / Warmup Noise)**：首字流式返回的前几个时间窗口常伴随 CUDA Runtime 的瞬时队列抖动。

### 🛠️ 解决方案
1. **基于真实 Usage 计算吞吐**：我们的压测脚本（`test_chunked_prefill.py`）会解析 stream 最后一包的 `usage.completion_tokens`，除以总生成时间得出精准的 `tokens/s` 吞吐。
2. **剔除 Warmup 窗口**：在统计算法中显式切除前 5 个非稳定包 intervals：`tpots[TPOT_WARMUP_TOKENS:]`。

---

## 6. API Key Bearer 鉴权失败 (`401 Unauthorized`)

### 🔴 症状现象
客户端调用 `http://localhost:8000/v1/chat/completions` 返回 `{"error": "Unauthorized"}`。

### 🔍 问题根因
`.env` 中设置了 `VLLM_API_KEY`，但调用打流脚本时 HTTP 请求头中没有携带鉴权 Token。

### 🛠️ 解决方案
1. 检查请求头必须包含：`"Authorization": "Bearer <YOUR_KEY>"`。
2. 如果是在纯本地研发压测，可随时将 `.env` 中的 `VLLM_API_KEY=` 留空，vLLM 会自动切换为免鉴权模式。

---

## 7. 端口被占用与网络打通冲突

### 🔴 症状现象
`docker compose up -d` 报错 `Bind for 0.0.0.0:8000 failed: port is already allocated`。

### 🔍 问题根因
宿主机有旧的 Python 服务、Jupyter Lab 或旧容器正在监听 8000 / 9090 / 3000 端口。

### 🛠️ 解决方案
1. 在 WSL2 终端执行：`sudo lsof -i :8000` 或 `netstat -tlpn | grep 8000` 找出旧进程 PID 终止。
2. 或直接通过 `.env` 修改对外映射端口，例如：`VLLM_PORT=8080`，做到灵活多环境无缝隔离。

---

## 8. Docker Compose 不识别 `.env` 中的变量

### 🔴 症状现象
`docker compose up -d` 后，容器使用的参数仍然是默认值（例如模型仍是 `Qwen/Qwen3-8B-AWQ`），而非 `.env` 中设置的自定义值。

### 🔍 问题根因
1. **`.env` 文件不在 `compose.yaml` 同一目录下**：Docker Compose 默认只从当前工作目录（即运行 `docker compose` 命令的目录）读取 `.env` 文件。
2. **`.env` 文件有 BOM 头或 Windows 换行符 (CRLF)**：可能导致变量值末尾包含不可见的 `\r` 字符。

### 🛠️ 解决方案
1. **始终从项目根目录执行命令**：`cd ~/projects/vllm-serving-lab && docker compose up -d`。
2. **验证变量是否被正确解析**：`docker compose config | grep model` 查看实际展开后的参数。
3. **转换换行符**：`dos2unix .env`（如果没有 dos2unix，可用 `sed -i 's/\r$//' .env`）。

