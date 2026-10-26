# 🔬 大模型推理深度下钻 Profiling 实战手册 (Nsight Systems / Nsight Compute / PyTorch Profiler)

> **导读**：当宏观服务指标（Prometheus/Grafana）提示存在延迟或吞吐瓶颈时，必须借助微观性能分析工具链，深入到 **GPU 硬件内核执行（Kernel Execution）**、**显存搬运（Memory Memcpy）** 与 **CPU↔GPU 同步屏障（Host-Device Synchronization）**，才能准确定位代码或算子级根因。

---

## 📑 目录

- [一、三级下钻性能分析全景图](#一三级下钻性能分析全景图)
- [二、一级下钻：PyTorch Profiler 算子级链路追踪](#二一级下钻pytorch-profiler-算子级链路追踪)
  - [1. 快速触发与 Chrome Tracing 导出](#1-快速触发与-chrome-tracing-导出)
  - [2. vLLM 原生 Profiler 接入流程](#2-vllm-原生-profiler-接入流程)
  - [3. Perfetto 可视化关键指标解读](#3-perfetto-可视化关键指标解读)
- [三、二级下钻：NVIDIA Nsight Systems 全系统时序追踪](#三二级下钻nvidia-nsight-systems-全系统时序追踪)
  - [1. Docker 容器内一键抓取指令](#1-docker-容器内一键抓取指令)
  - [2. 关键时序病症辨识：Kernel 间隙、同步阻塞与换出](#2-关键时序病症辨识kernel-间隙同步阻塞与换出)
- [四、三级下钻：NVIDIA Nsight Compute 单算子 Roofline 极限分析](#四三级下钻nvidia-nsight-compute-单算子-roofline-极限分析)
  - [1. PagedAttention 与 GEMV 算子精准捕捉](#1-pagedattention-与-gemv-算子精准捕捉)
  - [2. Roofline 模型：判断算子是否逼近硬件天花板](#2-roofline-模型判断算子是否逼近硬件天花板)
- [五、一键脚本与排查总结](#五一键脚本与排查总结)

---

## 一、三级下钻性能分析全景图

```
┌────────────────────────────────────────────────────────────────────────┐
│               大模型推理三级下钻 Profiling 工具链协同架构                │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 【第 1 级：服务宏观三角定位 (Prometheus & Grafana)】                     │
│ • 工具：vLLM /metrics、nvidia_gpu_exporter                              │
│ • 目标：10秒内判定排队瓶颈、Prefill 瓶颈或 Decode 瓶颈                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ 发现特定阶段性能劣化
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 【第 2 级：深度学习算子级追踪 (PyTorch Profiler)】                       │
│ • 工具：torch.profiler、vLLM API Profiler、Perfetto / Chrome Tracing    │
│ • 目标：定位耗时最长的 PyTorch 算子、显存瞬时暴增点与 Python 调度开销        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ 发现 GPU 内核存在异常空隙或同步阻塞
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 【第 3 级：硬件系统级与驱动发射 (NVIDIA Nsight Systems - nsys)】        │
│ • 工具：nsys CLI、Nsight Systems GUI                                   │
│ • 目标：观测 CPU 发射空隙、cudaStreamSynchronize 阻塞、PCIe H2D/D2H 换出  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ 发现特定 Attention / GEMM 效率偏低
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 【第 4 级：硬件微架构单算子剖析 (NVIDIA Nsight Compute - ncu)】          │
│ • 工具：ncu CLI、Nsight Compute GUI                                    │
│ • 目标：Roofline 模型、SM 流处理器利用率、Tensor Core 活跃度、L2 缓存命中率 │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 二、一级下钻：PyTorch Profiler 算子级链路追踪

### 1. 快速触发与 Chrome Tracing 导出
本项目提供了即开即用的本地与容器内 PyTorch Profiler 实战脚本：
```bash
# 模式 A: 本地执行真实 Transformer 解码层前向微架构 Profile (Prefill 2048 + Decode 阶段)
python3 scripts/torch_profiler.py

# 模式 B: 连接运行中的 vLLM 生产服务触发引擎原生 Profiler (/start_profile & /stop_profile)
python3 scripts/torch_profiler.py --server --base-url http://localhost:8000
```
该脚本在 RTX 5080 Laptop GPU 上执行完整的 Transformer Decoder Block（RMSNorm、RoPE、4:1 GQA 注意力层带 KV Cache 拼接、SwiGLU MLP）：
* 分别对 **Prefill 阶段 (2048 tokens)** 与 **Decode 逐字生成阶段 (1 token + 2048 KV Cache)** 划分 Profiler 作用域。
* 记录 CUDA Kernel Time、CPU Launch Time、Memory Allocation 与调用栈。
* 自动导出追踪文件：`results/profiles/torch_trace.json` (715 KB)。
* 打印消耗最大的 Top 算子表格（展示真实 Tensor Core CUTLASS GEMM 与 GEMV 算子）：

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------
Name                                                     Self CPU %    Self CUDA %   CUDA Total    CUDA Avg
-------------------------------------------------------  ------------  ------------  ------------  ------------
LLM_PREFILL_STAGE (2048 tokens)                          3.12%         82.76%        13.555 ms     13.555 ms
aten::mm (SwiGLU & QKV Linear)                           3.10%         73.70%        12.336 ms     293.7 us
_ZN7cutlass7Kernel2I68cutlass_80_tensorop_bf16_s16816...  0.00%         55.61%         9.108 ms       1.822 ms
LLM_DECODE_STEP (1 token + 2048 KV cache)                3.53%         20.45%         3.349 ms       3.349 ms
aten::bmm (Batched Attention Q*K^T & Attn*V)             2.22%         11.37%         1.920 ms     159.9 us
_ZN8internal5gemvx6kernelIii13__nv_bfloat16... (GEMV)    0.00%         10.59%         1.735 ms     115.6 us
aten::softmax (Scaled Dot-Product Attention)             0.03%          5.09%         0.833 ms     138.8 us
-------------------------------------------------------  ------------  ------------  ------------  ------------
Self CPU time: 75.22ms | Self CUDA time: 16.38ms (NVIDIA RTX 5080 Laptop GPU, PyTorch 2.11.0+cu128)
```

### 2. vLLM 原生 Profiler 接入流程
vLLM 官方镜像内嵌了 PyTorch Profiler 支持：
1. 在 `.env` 或 `compose.yaml` 中添加环境变量：
   ```ini
   VLLM_TORCH_PROFILER_DIR=/results/profiles
   ```
2. 启动服务后，向目标接口注入 profile 标记，或通过管理端点触发：
   ```bash
   # 触发开始打点
   curl -X POST http://localhost:8000/start_profile
   # 注入测试流量 (例如并发压测)
   bash scripts/benchmark.sh
   # 触发结束并存盘
   curl -X POST http://localhost:8000/stop_profile
   ```
3. 容器内部会自动将 `.pt.trace.json` 写入 `./results/profiles/` 目录。

### 3. Perfetto 可视化关键指标解读
在浏览器打开 [https://ui.perfetto.dev](https://ui.perfetto.dev) 并拖入生成的 JSON 文件：
* **CPU vs GPU 轨道重合度**：若 CPU 轨道上的算子执行完毕后，GPU 轨道有长段无核执行区域，说明系统受限于 CPU Python 解释器或调度器逻辑。
* **Memory 轨迹突刺**：检查 `Allocated` 显存曲线，若在 Prefill 阶段瞬时突刺逼近显存硬顶，说明需要引入分块预填充（Chunked Prefill）。

---

## 三、二级下钻：NVIDIA Nsight Systems 全系统时序追踪

### 1. Docker 容器内一键抓取指令
Nsight Systems 是分析大模型推理底层系统瓶颈的**终极核武器**。在容器内运行如下命令抓取 15 秒实时推理轨迹：

```bash
docker exec -it vllm-server nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=/results/profiles/vllm_nsys_trace \
  --duration=15 \
  --force-overwrite=true
```

### 2. 关键时序病症辨识

在 Nsight Systems GUI 中打开导出的 `.nsys-rep` 文件，重点检查以下三大典型“性能杀手”：

```
[病症 1：CPU Kernel Launch Overhead (驱动发射开销)]
CUDA Stream:  [Kernel 1] ─── 空白间隙 (5-10µs) ─── [Kernel 2] ─── 空白间隙 ─── [Kernel 3]
原因: CPU 单个发射小算子的速度赶不上 GPU 执行速度。
解法: 启用 CUDA Graph（本套件实验六），将整个 Decode 循环录制为单图发射！

[病症 2：CPU↔GPU Synchronization Stalls (同步屏障阻塞)]
CPU Thread:   [cudaStreamSynchronize (红色长条等待)] ════════════════════════════════
CUDA Stream:  [................ 漫长的计算 ................]
原因: Host 代码中意外调用了 .item()、.cpu() 或显式 cudaStreamSynchronize 造成流水线排空打断。
解法: 全链路异步执行，将结果拷贝与采样步骤推迟合并。

[病症 3：KV Cache CPU Thrashing (内存置换频繁)]
Memory Track: [DtoH Memcpy 800MB] ──── [HtoD Memcpy 800MB] ──── [DtoH Memcpy]
原因: 显存池被撑爆，触发了 CPU Swap Space 颠簸。
解法: 开启 FP8 KV Cache 量化（本套件实验七），直接将物理槽位翻倍。
```

---

## 四、三级下钻：NVIDIA Nsight Compute 单算子 Roofline 极限分析

### 1. PagedAttention 与 GEMV 算子精准捕捉
当怀疑某个特定算子（如 PagedAttention 解码内核）效率不高时，可使用 `ncu` 进行微架构采样：

```bash
ncu --set full \
    --kernel-name regex:.*paged_attention.* \
    --target-processes all \
    -o /results/profiles/ncu_paged_attention \
    <launch_command>
```

### 2. Roofline 模型：判断算子是否逼近硬件天花板
Nsight Compute 会自动绘制 **Roofline（屋顶模型）图**：
* **计算强度横坐标**：$\text{FLOPs/Byte}$。
* **吞吐纵坐标**：$\text{GFLOPs/s}$ 或 $\text{TFLOPs/s}$。
* **判定法则**：
  * 若算子位于 **左侧斜坡（Memory Bandwidth Roof）** 上：说明该算子已榨干了当前 GPU 物理显存带宽（如 1000 GB/s），计算优化无法提升它，**必须通过量化（减少字节搬运）来解决**。
  * 若算子位于 **右侧平顶（Peak Tensor Core Compute Roof）** 下方很远处：说明该算子算力受限但效率不高，存在寄存器溢出（Spill）、共享内存冲突（Bank Conflict）或并行度不足问题。

---

## 五、一键脚本与排查总结

本项目将上述多级排查动作封装为一键执行脚本：

```bash
# 1. 运行宏观服务三角定位
bash scripts/profile_service.sh --macro

# 2. 运行 PyTorch Profiler 本地微观追踪
bash scripts/profile_service.sh --torch

# 3. 打印 Nsight Systems 容器端抓取与排查指导
bash scripts/profile_service.sh --nsys

# 4. 打印 Nsight Compute 算子级屋顶模型分析指导
bash scripts/profile_service.sh --ncu
```
