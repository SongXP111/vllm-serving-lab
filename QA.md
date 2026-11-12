# 💡 大模型推理与 AI Infra 核心问题与深度问答集 (QA Collection)

> **导读**：本问答集系统化沉淀在大模型服务（LLM Serving）与 AI Infra 学习、实战及架构设计中遇到的高频思考、经典认知误区与工业级工程权衡博弈。

---

## 📑 目录

- [Q1: 工业界是如何设定 `max-num-batched-tokens` 的？为什么不直接把预算设得更小以保护流式打字体验？](#q1-工业界是如何设定-max-num-batched-tokens-的为什么不直接把预算设得更小以保护流式打字体验)
- [Q2: 为什么在大并发或长 Prefill 场景下，W4A16（AWQ/GPTQ）反而不如 W8A8（FP8）香？](#q2-为什么在大并发或长-prefill-场景下w4a16awqgptq反而不如-w8a8fp8香)
- [Q3: 投机解码（Speculative Decoding）中，大模型验证不也是算一遍吗，为什么比自己生成快？为什么小模型每次只猜 3~5 个词而不是一次性全写完？](#q3-投机解码speculative-decoding中大模型验证不也是算一遍吗为什么比自己生成快为什么小模型每次只猜-35-个词而不是一次性全写完)
- [Q4: 在 PD 分离架构中，跨机传输 KV Cache 难道不费时间和带宽吗？如果 Decode 显存满了接不住怎么办？普通机房如何选型？](#q4-在-pd-分离架构中跨机传输-kv-cache-难道不费时间和带宽吗如果-decode-显存满了接不住怎么办普通机房如何选型)
- [Q5: 既然 vLLM 自带了 HTTP API Server，为什么生产环境必须先挡一层自己的 Gateway？为什么 M1 第一步规定“先只连 1 台 vLLM，不做智能路由”？](#q5-既然-vllm-自带了-http-api-server为什么生产环境必须先挡一层自己的-gateway为什么-m1-第一步规定先只连-1-台-vllm不做智能路由)
- [Q6: 在流式推理（SSE）中，已经发出 HTTP Headers 后为什么不能再随意返回另一份 JSON 错误？M1 Step 1 的五大失败测试如何证明零资源泄漏？](#q6-在流式推理sse中已经发出-http-headers-后为什么不能再随意返回另一份-json-错误m1-step-1-的五大失败测试如何证明零资源泄漏)
- [Q7: 为什么必须在 Gateway 层决定「接受、排队还是拒绝」后 Router 才有可用的负载状态？llm-d EPP 的 Flow Control 四大核心概念（Saturation / Flow / Priority / Fairness）是什么？](#q7-为什么必须在-gateway-层决定接受排队还是拒绝后-router-才有可用的负载状态llm-d-epp-的-flow-control-四大核心概念saturation--flow--priority--fairness是什么)
- [Q8: 既然 vLLM 自己内部已经有 `waiting` 和 `running` 调度队列，为什么 Gateway 的队列必须独立于它们？](#q8-既然-vllm-自己内部已经有-waiting-和-running-调度队列为什么-gateway-的队列必须独立于它们)

---

### Q1: 工业界是如何设定 `max-num-batched-tokens` 的？为什么不直接把预算设得更小以保护流式打字体验？

#### ❓ 问题背景与直觉假说
在学习分块预填充（Chunked Prefill）时，一种非常符合直觉的想法是：
> *“小预算似乎更合理。如果用大预算（如 8192），某个用户输入超大 Prompt 会霸占算力并严重影响其他流式打字的用户；而如果用极小预算（如 512 或 256），只有输入长 Prompt 的用户会感觉到慢。但这合情合理，因为他输入的内容本来就很长，心理预期等待时间更久是完全正常的。为什么工业界不普遍采用更小预算？”*

---

#### 💡 核心解析与第一性原理

你的直觉在**“用户体验 SLA（服务等级协议）”与“调度公平性”**维度是**完全正确且敏锐的**，直接呼应了操作系统经典的短作业优先（SJF）和尾延迟防抖原则。

然而，AI Infra 绝非仅看单一用户体验，还受制于**“底层硬件物理架构（Hardware Reality）”**与**“数据中心财务成本（Economics）”**。工业界之所以没有将预算无脑压到极低，是因为面临着以下三座大山：

---

#### 一、为什么预算不能设得太小？（底层物理与工程成本）

##### 1. Tensor Core 算力“断崖式坍塌”（GEMM 矩阵吃不饱）
现代 GPU（如 NVIDIA Hopper H100、Blackwell B200、Ada Lovelace RTX 4090/5080）拥有 100+ 个流式多处理器（SM）和数百个专用 Tensor Core。
* 在大模型注意力和 FFN 的矩阵乘法（GEMM：$M \times N \times K$）中，**$M$ 就是当前 Batch 内的总 Token 数**。
* **当 $M \ge 2048$ 时**：数据规模充足，GPU 内部有足够的线程块（Thread Blocks）和 Warp 能铺满所有 SM 核心，Tensor Core 处于高饱和度利用状态（Model FLOPs Utilization 达 60%~80%）。
* **当 $M$ 骤降到 512 甚至 256 时**：GEMM 矩阵极度“瘦窄”，根本分不出足够的线程块去喂饱上百个 SM。大量计算单元处于空转摸鱼状态，**硬件实际计算效率会断崖式暴跌至 20%~30%**。

##### 2. 调度器摩擦与“千刀万剐”反噬
假设一个用户传入了一个 **8192 Tokens** 的长文档：
* **预算设为 4096**：GPU **只需 2 次前向迭代** 就能完成全部 Prefill。
* **预算设为 512**：该请求被迫切成整整 **16 个微小碎块**！
  * CPU/调度器必须在 Host 和 Device 之间轮询 16 次、调度发射 16 轮底层的 CUDA Kernels，并频繁维护 16 次中间块表状态；
  * 这种调度开销和低效 GEMM 产生叠加摩擦，导致长文本用户的首字延迟（TTFT）**从原本的 1 秒暴增到 4~5 秒甚至更久**，甚至极易触发上层 API 网关的请求超时错误（Gateway Timeout）。

##### 3. 数据中心算力账（单位 Token 成本翻倍）
以单台 8 卡 H100 云服务器为例，月租金通常在 \$20,000 ~ \$30,000。
* 如果因为预算设得太小导致 Tensor Core 利用率折半，集群每秒能产出的总 Token 吞吐量（Tokens/s）直接腰斩；
* 为了扛住相同的业务洪峰，企业被迫采购双倍数量的服务器，**每月多烧几十万机房租金与电力成本**。

---

#### 二、工业界是如何科学设定的？

工业界通常综合以下两套方法论进行科学定参：

##### 1. 寻找硬件算力饱和拐点（The Knee of the Curve）
通过 Profiling 绘制硬件吞吐效率与 Token 预算的对应曲线：

```
Tensor Core 算力饱和度 (%)
 100% │                              
  80% │                    ╭─────────── [饱和平原: 2048 ~ 8192]
  60% │               ╭────╯            (进一步加大吞吐增益微弱，卡顿风险骤增)
  40% │          ╭────╯
  20% │     ╭────╯ [坍塌区: < 1024]
   0% └─────┴────────┴────────┴────────┴──
           512      1024     2048     4096   (max-num-batched-tokens)
```
* **结论**：低于 2048，算力断崖下跌；高于 4096，吞吐提升微乎其微但对打字流的冲击成倍放大。
* 因此，**2048 或 4096** 成为了现代开源引擎（vLLM、TensorRT-LLM）兼顾硬件效率与交互延迟的**“黄金平衡点（Default Sweet Spot）”**。

##### 2. 依据业务形态实施 SLA 物理分级

| 业务形态 | 核心矛盾 | 工业界配置推荐 | 选型考量 |
| :--- | :--- | :--- | :--- |
| **C 端流式对话 (ChatGPT 类)** | 用户紧盯屏幕，打字顿挫零容忍 | `max-num-batched-tokens: 2048`<br>*(甚至极度严格的 1024)* | 适当牺牲边缘吞吐，绝对保证流式打字 P95 TPOT 稳定在 15ms 以内。 |
| **离线批量 / 数据合成 / 总结** | 背后批处理运行，无交互阅读诉求 | `max-num-batched-tokens: 8192`<br>*(甚至更高)* | 算力完全吃满，追求单位成本下产出的最大总 Tokens 数。 |
| **超长文本智能体 (Agent 32k~128k)** | 单请求上下文巨大，切块太多易超时 | 动态弹性切块 或 **PD 分离** | 单机单参数已达物理极限，依赖集群拓扑解耦。 |

---

#### 三、架构演进：为什么单机混部最终走向了 PD 分离？

在单体服务（Monolithic Serving）架构下，无论工程师如何精巧地微调 `max-num-batched-tokens`，本质上都是在**“保流式打字流畅”**与**“保长文本 Prefill 效率”**之间走钢丝。

为了彻底消除这一固有妥协，现代大模型基础设施诞生了终极演进形态——**Prefill-Decode Disaggregation (PD 分离架构)**：
* **Prefill 算力节点**：把预算拉到最大（如 16384），让 Tensor Core 满负荷轰鸣，极速算完大矩阵；
* **Decode 带宽节点**：只接自回归请求，专注于低延迟单字生成，彻底杜绝打字卡顿；
* **高速 KV 搬运**：两类节点之间通过 RDMA 网络与 KV Connector（如 Mooncake / NIXL）在毫秒级无损转运 KV Cache。

---

### Q2: 为什么在大并发或长 Prefill 场景下，W4A16（AWQ/GPTQ）反而不如 W8A8（FP8）香？

#### ❓ 问题背景与核心洞察
在对比模型量化方案时：
* **W4A16（如 AWQ / GPTQ）**：将权重压缩到极致的 4-bit（0.5 Byte），但激活值（Activation）依然保持 16-bit（FP16/BF16）；
* **W8A8（如 FP8 Dynamic）**：权重与激活值同时量化为 8-bit（1 Byte）。

直觉上 4-bit 比 8-bit 省了一半显存，为什么在大并发（High Batch Size）或长文本 Prefill 阶段，工业界数据中心反而更推崇 W8A8 (FP8)？
> **核心洞察**：“如果激活值 A 不量化，大并发的时候计算量会很大，系统会从 Memory-Bound 变成 Compute-Bound，此时必须把激活值也一起量化以调用更高算力的低精度 Tensor Core！”

---

#### 💡 底层硬件物理机制深度拆解

##### 1. W4A16（只量化权重）的真实硬件执行过程
现代 GPU 硬件中其实**并不存在**直接计算 `4-bit 权重 × 16-bit 激活值` 的原生 Tensor Core 指令。因此 W4A16 在 GPU 内部的真实流水线是：
1. **省显存带宽**：将 4-bit 权重从高延迟显存（VRAM）搬运到片上高速缓存（SRAM / 寄存器），此步骤节省了 75% 的显存带宽搬运时间。
2. **反量化开销（Dequantize ALU Overhead）**：在寄存器中，必须调用 CUDA 核心执行反量化算子，将 4-bit 整数**现场解压还原回 16-bit 浮点数（FP16/BF16）**。
3. **16-bit 矩阵乘**：最后送入 **FP16 Tensor Core**，与 16-bit 激活值执行常规的 `16-bit × 16-bit` 矩阵乘法（峰值算力为 $1\times$）。

##### 2. 屋顶模型（Roofline Model）的瓶颈跃迁：从 Memory-Bound 到 Compute-Bound
* **小并发 Decode 阶段（Batch = 1 ~ 4）**：
  * 此时计算强度极低，系统完全卡死在**显存带宽（Memory-Bound）**，算力严重过剩。
  * W4A16 省下了 75% 的权重搬运时间，而过剩的算力可以毫无压力地掩盖掉“4-bit 解压回 16-bit”的反量化开销，因此小并发下 W4A16 加速极其显著。
* **大并发（Batch = 64+）或长文本 Prefill 阶段**：
  * 大量 Token 拼成大矩阵，计算强度（FLOPs/Byte）急剧飙升，系统越过屋顶模型的拐点，彻底演变为**算力受限（Compute-Bound）**！
  * 此时，W4A16 **不仅无法提升 Tensor Core 的计算吞吐上限（依然受困于 16-bit Tensor Core），其额外的反量化解压指令反而争抢了宝贵的片上执行周期，拖累了整体计算速度！**

##### 3. W8A8 (FP8) 为什么在大并发下封神？
* **零反量化解压**：权重是 FP8，激活值也是 FP8，无需还原回 16-bit。
* **原生 2 倍硬件峰值算力**：直接送入现代显卡（NVIDIA Ada Lovelace RTX 4090/5080、Hopper H100、Blackwell B200）硬件原生的 **FP8 Tensor Core**。在同一张显卡上，**FP8 Tensor Core 的硬件峰值算力（TFLOPs）正好是 FP16 Tensor Core 的整整 2 倍！**
* **双重收益**：既省下了 50% 的显存带宽，又直接将 Compute-Bound 阶段的算力天花板翻了一倍。

---

#### 📊 W4A16 vs W8A8 (FP8) 工业选型对照表

| 评估维度 | W4A16 (AWQ / GPTQ) | W8A8 (FP8 Dynamic) |
| :--- | :--- | :--- |
| **权重体积** | **0.5 Byte** (节省 75% 权重显存) | 1 Byte (节省 50% 权重显存) |
| **激活值精度 (Activation)** | 16-bit (2 Bytes，不量化) | **8-bit (1 Byte，同步量化)** |
| **反量化 (Dequant) 开销** | 有（需在寄存器内实时解压回 16-bit） | **无（硬件原生直算）** |
| **调用的硬件计算单元** | **FP16/BF16 Tensor Core (1x 基准算力)** | **FP8 Tensor Core (2x 翻倍峰值算力)** |
| **最佳适用场景** | **单卡显存极度受限 / 小并发流式生成**<br>（纯靠极致压缩体积破除显存带宽瓶颈） | **数据中心高并发 / 长 Prefill / 大吞吐生产集群**<br>（兼顾显存带宽减半与计算峰值算力翻倍） |

---

### Q3: 投机解码（Speculative Decoding）中，大模型验证不也是算一遍吗，为什么比自己生成快？为什么小模型每次只猜 3~5 个词而不是一次性全写完？

#### ❓ 问题连环追问
1. **灵魂追问一**：投机解码时，小模型猜出 4 个词交由大模型验证，“大模型一算发现全猜对了”，这不也相当于大模型把这 4 个词算了一遍吗？为什么把 4 个词拼成矩阵喂给 GPU 就能并行算，而平时自己生成却不行？
2. **灵魂追问二**：既然大模型并行验证这么快，工业界为什么不让小模型一口气把整篇 500 字的回答全写完，再交给大模型一次性验证？
3. **灵魂追问三**：什么是多词预测头（MTP）？作为 AI Infra 工程师需要掌握到什么深度？

---

#### 💡 核心解析一：为什么验证 4 个词能并行，而自己生成 4 个词必须串行？

**大模型做的矩阵乘法计算量确实完全一样！核心区别在于：自己生成要“串行搬运 4 遍权重（耗时 40ms）”，而验证草稿只需“并行搬运 1 遍权重（耗时 10ms）”！**

##### 1. 平时自己生成（常规 Decode）：卡在“不知道下一个位置的输入是什么”
假设已有 `"今天天气真"`，要生成 `"不"`、`"错"`、`"呀"`、`"！"`：
* 没算出第 1 个词 `"不"` 之前，第 2 行的输入是**空白**！
* 大模型被迫串行走 4 步：每走一步，只填 1 行数据，从显存搬运 1 遍 14GB 权重（耗时 10ms），走完 32 层在出口拿到 1 个词，再填回下一轮的第一层入口。4 个词累计搬运 $14\text{GB} \times 4 = 56\text{GB}$，耗时 **40ms**。

##### 2. 投机解码并行验证：小弟提前填好了每一行的输入，彻底解除串行等待！
当小模型提前猜出 `["不", "错", "呀"]` 后，`["真", "不", "错", "呀"]` 这 4 行的字面输入**已经全部已知**，像做小型 Prefill 一样一次性送入 GPU：
* **第一步（乘权重矩阵 $W$）**：算第 3 行 $\text{向量(“错”)} \times W$ 完全不需要等第 2 行 $\text{向量(“不”)} \times W$！GPU 只从显存**搬运 1 遍 14GB 权重（耗时 10ms）**，成千上万个线程在同一纳秒内把 4 行同时乘完，生成各自的 $Q, K, V$。
* **第二步（下三角因果掩码自注意力）**：通过 $4 \times 4$ 的下三角掩码（Causal Mask），GPU 的 4 组线程**同时并行计算** 4 个位置的注意力，并在第 32 层出口处同时吐出 4 个位置的预测结果：
  * 位置 1（只看 `"真"`）$\to$ 预测出 `"不"`（对照草稿 `"不"` ✅）
  * 位置 2（看 `"真不"`）$\to$ 预测出 `"错"`（对照草稿 `"错"` ✅）
  * 位置 3（看 `"真不错"`）$\to$ 预测出 `"呀"`（对照草稿 `"呀"` ✅）
  * 位置 4（看 `"真不错呀"`）$\to$ 顺手预测出 `"！"`（🎁 白送第 4 个词！）
* **哪怕中间猜错了也零精度损失**：若小弟第 2 个词猜成 `"赖"`，大模型在位置 2 会直接算出正确词 `"错"`，当场丢弃 `"赖"` 及之后的草稿并替换为 `"错"`，数学上保证输出与大模型逐字生成 **100% 完全等价**。

---

#### 💡 核心解析二：为什么采用“每次猜 3~5 个词的乒乓接力”，而不是让小模型一次性全写完？

**原因：自回归生成的“多米诺骨牌崩塌效应（一错毁所有）”与命中率指数衰减定律！**

1. **骨牌崩塌效应**：
   * 假设提问 `"法国的首都是哪里？"`，呆萌的小模型一口气写了 500 字：`"是【伦敦】，那里有大本钟、泰晤士河……"`。
   * 大模型在批改第 2 个词时直接将 `"伦敦"` 修正为 **`"巴黎"`**。
   * 一旦第 2 个词被改写，小模型后面基于“伦敦”写出的 498 个词（大本钟、泰晤士河）**上下文彻底作废，必须 100% 扔进垃圾桶**！小模型白忙活 500 步，大模型也白白浪费了验证 500 个词的算力。
2. **命中率的指数衰减**：
   * 假设小模型单字猜对率高达 $\alpha = 80\%$：
   * 连续猜对 $K=4$ 个词的概率：$0.8^4 \approx 41\%$（非常划算）；
   * 连续猜对 $K=20$ 个词的概率：$0.8^{20} \approx 1.1\%$；
   * 连续猜对 $K=100$ 个词的概率：$0.8^{100} \approx 0.00000002\%$（等于零）。
3. **工业界标准“乒乓接力（Ping-Pong Loop）”**：
   * 小模型每次仅往前猜 **$K = 3 \sim 5$ 个词** $\to$ 大模型立刻验证并修正错误词 $\to$ 小模型站在大模型修正后的正确词基础上，继续猜下 $3 \sim 5$ 个词。

---

#### 💡 核心解析三：工业界三大投机流派与多词预测头（MTP）

| 流派演进 | 谁来当“猜词小弟”？ | 核心机制与工业界代表 |
| :--- | :--- | :--- |
| **1. 独立小模型**<br>*(Draft Model)* | 同词表的轻量级独立模型<br>*(如 Llama-70B 配 Llama-1B)* | 早期方案。需在显存中同时维护大小两个模型与两套 KV Cache，工程调度复杂。 |
| **2. N-gram 历史匹配**<br>*(Prompt Lookup)* | **无需模型**，直接从上文/Prompt 中匹配重复词组抄写 | **零额外显存开销**（本项目 `run_exp5` 采用）。在写代码、改写文档、生成 JSON 时极快。 |
| **3. 多词预测头**<br>*(MTP / EAGLE / Medusa)* | **大模型自己长出几个“小脑袋”**<br>在主干网络最后一层挂载轻量预测头 | **🔥 当前工业界终极形态（DeepSeek-V3 / R1 核心技术）**。大模型跑完 32 层沉重身体提炼出深层思维向量（Hidden State）后，除了用主头吐出第 1 个词，还顺手用串联的轻量单层小头（MTP Head）直接预判出第 2、3 个词，下一轮再由主干网络并行验证。站在巨人肩膀上猜词，命中率高达 80%~90%，且零独立模型运维负担！ |

> 📌 **AI Infra 工程师的决策边界（何时开 / 何时关）**：
> * ✅ **必开场景**：**低并发（小 Batch）** + **高确定性任务**（代码生成、JSON 结构化输出、文档总结）。此时显存带宽受限而算力闲置，投机解码可提速 **1.5x ~ 2.5x**。
> * ❌ **必关场景**：**大并发满载（如 Batch Size = 64+）** 或 **高随机性创意写作**。大并发下 GPU 算力本来就已经被打满（Compute-Bound），此时再塞入额外的草稿词做验证，会直接争抢紧缺算力，反而导致全集群延迟恶化！

---

### Q4: 在 PD 分离架构中，跨机传输 KV Cache 难道不费时间和带宽吗？如果 Decode 显存满了接不住怎么办？普通机房如何选型？

#### ❓ 问题连环追问
1. **传输开销拷问**：Prefill 节点算完把几百 MB 甚至上 GB 的 KV Cache 传给 Decode 节点，难道不会消耗大量网络时间和带宽吗？
2. **反压死锁拷问（Mooncake 核心痛点）**：如果 Prefill 算完了准备发货，但 Decode 节点的 KV Cache 显存刚好满了接不住，会发生什么？怎么解决？
3. **机房选型拷问**：如果公司只有普通机房（没有昂贵的 RDMA 高速网卡，只有普通万兆以太网），是不是应该直接用“应用层网关分流”而不是跨机真·PD 分离？

---

#### 💡 核心解析一：工业界如何将 KV Cache 跨机传输延迟“藏”起来？

哪怕拥有 $200 \sim 400 \text{ Gbps}$ 的 RDMA 高速网络，如果傻等 32 层 Transformer 全部算完再一次性打包传输，依然会产生几十毫秒的停顿。工业界通过两大绝招将其隐形：

##### 1. 逐层流水线异步重叠（Layer-by-Layer Compute-Communication Overlap）
Prefill 节点**绝不等 32 层全算完才发货**，而是算完一层、立刻在后台异步传一层：
```
时间轴 ──────────────────────────────────────────────────────────►
GPU 计算核心:  [算第1层] ──► [算第2层] ──► [算第3层] ... ──► [算第32层]
RDMA 网卡(DMA):          └─► [传第1层KV] ─► [传第2层KV] ... ─► [传第31层KV] ─► [传第32层KV]
```
* 当 GPU 的 Tensor Core 刚算完 **第 1 层**、转身去算 **第 2 层** 的同一瞬间，网卡（GPUDirect RDMA DMA 引擎，完全不占用 GPU 计算核心）已经在后台把 **第 1 层的 KV Cache** 直接发往 Decode 节点显存了。
* **效果**：当 GPU 算完第 32 层时，**前 31 层的 KV Cache 早就已经躺在 Decode 节点的显存里了！** 最终暴露出来的网络延迟仅仅是传最后 1 层所需的不到 1 毫秒！

##### 2. 从源头压缩包裹体积（FP8 KV Cache 与 DeepSeek MLA 架构）
* **工程量化压缩**：开启 `--kv-cache-dtype fp8_e4m3`，直接将待传 KV Cache 体积砍半（节省 50% 网络带宽）。
* **模型架构级压缩（DeepSeek-V3/R1 发明 MLA 的核心动因）**：DeepSeek 之所以摒弃传统注意力而发明 **MLA（多头潜在注意力）**，很大程度上就是为了配合其内部大规模 **PD 分离架构**——MLA 通过低秩矩阵投影将 **KV Cache 体积暴砍了 93.3%**，让跨节点传输包裹缩水十倍以上，彻底解除网络带宽瓶颈！

---

#### 💡 核心解析二：如果 Decode 显存满了接不住怎么办？（防止全集群级联反压死锁）

如果不加控制：Prefill 算完捏着几 GB 的 KV Cache 发不出去，不敢丢弃只能憋在自己显存里等待；几秒钟后 Prefill 自己的显存也被待发包裹彻底塞满，引发 **全集群级联死锁（Cascading Backpressure Deadlock）**。

工业界（如 Kimi 的 Mooncake 架构与 vLLM V1）采用 **三大防爆机制** 根治这一问题：

1. **“先占座，后上车”（Pre-Allocation / 预留握手协议）**：
   * 在请求刚到达、**Prefill 甚至还没开始计算前**，全局调度器会先向目标 Decode 节点发起握手，**提前锁死预留该请求所需的物理 Block 槽位**。
   * 若 Decode 显存不足无法预留，该请求直接在队列外等待，绝不允许“算完了却没地方发货”。
2. **设立“丰巢快递柜”（分布式内存缓冲池 / Mooncake KV Pool）**：
   * 若 Decode 节点由于正在生成超长回答而暂时显存吃紧，Prefill 节点绝不把 KV Cache 憋在昂贵的 GPU 显存里，而是瞬间将其卸载到 **服务器庞大的主机内存池（Host DRAM，数百 GB）** 构成的分布式缓冲池中，立刻释放 Prefill 显卡；
   * 待 Decode 节点空出显存后，再从内存缓冲池中拉取（Pull）过来继续生成。
3. **超时丢弃与弹性重算（Fallback / Recompute）**：
   * 极端洪峰下若缓冲池亦满载，果断丢弃未送达的 KV 块并在稍后触发轻量重计算，或者由 Prefill 节点临时客串原地完成 Decode，确保集群绝对不发生死锁。

---

#### 💡 核心解析三：普通机房（无 RDMA 高速网络）的务实架构选型

| 机房网络与硬件条件 | 推荐架构形态 | 工程选型理由 |
| :--- | :--- | :--- |
| **普通千兆/万兆以太网 (TCP/IP)**<br>无跨机 RDMA，单卡或少卡部署 | **段位 1：应用层长短分流网关 (`disagg_router.py`)**<br>+ 单实例内开启 **`Chunked Prefill`** | 跨机网速太慢（传 1GB 需数秒），强行跨机传 KV 会造成灾难级延迟。按 Prompt 长度将长短请求分流至不同预算的独立实例，零网络改造成本即可隔离打字抖动。 |
| **单台 8 卡服务器内部**<br>具备主板级 **NVLink / PCIe P2P** | **单机内算子级真·PD 分离**<br>（如前 4 卡 Prefill，后 4 卡 Decode） | 虽然没有跨机 RDMA，但机内卡间 NVLink 带宽高达 $400 \sim 900 \text{ GB/s}$，通过 `PyNcclConnector` 在同机不同卡间瞬移 KV Cache 极快。 |
| **顶级数据中心集群**<br>配备 **InfiniBand / RoCE RDMA ($200\text{G}+$) ** | **段位 2：跨机大规模算子级真·PD 分离**<br>（结合 Mooncake / NIXL 异构弹性池化） | 充分发挥 RDMA 零拷贝直传优势，实现 Prefill 与 Decode 异构显卡混搭与独立弹性扩缩容。 |

---

### Q5: 既然 vLLM 自带了 HTTP API Server，为什么生产环境必须先挡一层自己的 Gateway？为什么 M1 第一步规定“先只连 1 台 vLLM，不做智能路由”？

#### ❓ 问题背景
在从单机推理实验（M0）向生产级推理平台（Serving Platform M1）演进时，学习计划强调：
> *“同样一个请求先经过你自己的 HTTP 网关，才有地方控制鉴权、请求生命周期、超时和取消；此时先只连一台 vLLM，不做智能路由。”*

为什么不能让客户端直接连 vLLM 的 8000 端口？为什么第一步严格限制只连 1 台 vLLM？

---

#### 💡 核心解析一：为什么不能让流量直接裸连 vLLM，必须加一层自己的 Gateway？

vLLM 内置的 FastAPI Server 职责是**“专心将请求喂给底层 Scheduler 与 GPU”**，而非抗住公网复杂的业务生命周期。若没有独立 Gateway 接管请求生命周期，生产环境会出现 **四大灾难**：

##### 1. 管“取消（Cancel Propagation）” —— 防止用户走了，GPU 还在白烧钱！
* **场景**：用户请求生成 2000 字长文（需 20 秒），但刚看了前 20 个字就点击了“停止生成”或直接关掉了网页标签页。
* **裸连或劣质代理灾难**：若上游未实时感知客户端断开并向下游传播取消信号，后端 vLLM 会在毫不知情的情况下继续在 GPU 上生成剩下的 1980 个字，白白霸占宝贵的 KV Cache 显存块与算力整整 20 秒！
* **Gateway 职责**：实时监控客户端断连（`request.is_disconnected()`），一旦断开，**立即主动关闭（Abort/Close）连向 vLLM 的上游 HTTP 流**，触发 vLLM Engine 当场将该请求从 `running` 队列驱逐并瞬间回收物理显存块。

##### 2. 管“超时（Deadline / Timeout）” —— 消灭“僵尸废品算力（Zero Goodput）”
* **场景**：客户端设置了 10 秒超时，但因系统拥堵，请求在队列中排了 12 秒才轮到 GPU 执行。
* **灾难**：第 10 秒时客户端早已超时退出，但 vLLM 在第 12 秒依然耗费巨量算力将其算完，产出的 Token 无人接收（有效吞吐 Goodput 归零）。
* **Gateway 职责**：维护全链路 Deadline，一旦在网关排队超时或首字响应超时（TTFT Timeout），当场拦截清理，绝不将已过期的“僵尸请求”放进 GPU。

##### 3. 管“流式三态生命周期（SSE State Machine）” —— 发出 HTTP 200 后无法反悔！
普通 HTTP 接口失败可直接返回 `500`。但大模型 SSE 流式输出（`stream=true`）具有严格的 **三阶段不可逆状态机**：
```
[阶段 1: 未开始 (Pending)] ──► 尚未收到后端首字。若后端挂了，Gateway 可直接返回标准 HTTP 502/503/504 JSON 错误。
        │
        ▼ (收到第 1 个 Token！Gateway 已向客户端发出 HTTP 200 OK 响应头！)
[阶段 2: 流式传输中 (Streaming)] ──► 正在逐字透传。若吐到第 50 个字时后端崩溃，
        │                           ⚠️ 此时 HTTP 200 状态码早已发出，绝对不可能再改回 500 状态码！
        │                           Gateway 必须按协议规范发射流内错误事件并优雅闭合流，防止客户端死等。
        ▼
[阶段 3: 已结束 (Completed/Aborted)] ──► 无论正常结束、客户端断连还是后端异常，必须确保底层连接池（httpx）零句柄泄漏。
```

##### 4. 管“多租户鉴权、配额与过载反压（Auth / Quota / Backpressure）”
vLLM 自带的 `--api-key` 仅能校验单一静态密码，无法实现租户级 Token 速率限制（Rate Limit）、有界队列（Bounded Queue）以及区分 `429 Too Many Requests`（租户超限）与 `503 Service Unavailable`（系统满载反压）。

---

#### 💡 核心解析二：为什么 M1 第一步规定“先只连 1 台 vLLM，绝对不做智能路由”？

这是系统工程中最核心的 **“控制变量与分层解耦（Isolation of Concerns）”** 原则：

1. **与 M0 基线进行 1:1 照妖镜比对（验证零缓冲流式透传）**：
   * **M0 基线**：`Client ──► 单台 vLLM`（例如实测 `TTFT = 39ms`，`TPOT = 10ms`）。
   * **M1 Step 1**：`Client ──► 你的 Gateway ──► 同一台 vLLM`。
   * **照妖镜作用**：因为后端完全是同一台 vLLM，你可以直接对比加了 Gateway 前后的延迟：
     * 若 `TTFT` 依然是 `~40ms`，证明你的异步流式透传开销仅 `~1ms`，管道完全合格；
     * 若 `TTFT` 暴增到 `5000ms`（等同于 E2E 总耗时），当场即可抓出新手写网关最常犯的致命 Bug：**不小心把后端的流在网关内存里做了全量缓冲（Buffering），等整篇回答全生成完才一次性吐给客户端，把真流式写成了假流式！**
2. **先堵死“单根水管”的泄漏，再装“多路分叉阀门”**：
   * 异步流式代理隐藏着连接池耗尽、协程悬挂、断连未清理等诸多底层 IO 陷阱。
   * 如果一上来把“异步流式代理 + 排队限流 + 多节点 Prefix 路由”揉在一起写，一旦压测卡死或延迟飙升，根本无法定位是 **路由算法选错了节点** 还是 **网关 IO 管道本身漏水**。
   * 先在 `1 Gateway → 1 vLLM` 的极简拓扑下通过全部异常测试（断连、慢客户端、后端中途崩溃、并发连接池复用），把单根管道锤炼得滴水不漏后，后续叠加 Bounded Queue（Step 2）与 Prefix-aware Router（M2）就会水到渠成。

---

### Q6: 在流式推理（SSE）中，已经发出 HTTP Headers 后为什么不能再随意返回另一份 JSON 错误？M1 Step 1 的五大失败测试如何证明零资源泄漏？

#### ❓ 问题背景
在实现 `gateway/app.py`（M1 Step 1）时，验收要求明确规定：
> *“必须做的失败测试：客户端收到若干 token 后主动断开；慢客户端读取；backend 在首字前失败与流式输出中途失败；多请求并发。记录是否资源泄漏、连接是否及时关闭，以及已经发出 HTTP headers 后为什么不能再随意返回另一份 JSON 错误。”*

---

#### 💡 核心解析一：已经发出 HTTP Headers 后，为什么不能再返回另一份 HTTP 500/502 JSON 错误？

这源于 **HTTP/1.1 字节流协议（Wire Protocol）与 ASGI 状态机的单向不可逆性**：

1. **HTTP 报文在 TCP 网线上的物理顺序是严格固定的**：
   ```
   HTTP/1.1 200 OK\r\n                        <── 1. 状态行（Status Line，只能发一次！）
   Content-Type: text/event-stream\r\n        <── 2. 响应头（Headers）
   Transfer-Encoding: chunked\r\n
   \r\n                                       <── 3. 空行（代表 Header 彻底结束，进入 Body 阶段！）
   data: {"choices":[{"delta":{"content":"你好"}}]}\n\n   <── 4. 响应体（Body Chunks）
   ```
2. **首字前失败（`RequestState.NOT_STARTED`）—— 还没发空行，可以随时悔改**：
   * 在 `gateway/app.py` 中，我们通过 `first_chunk = await anext(chunk_iter)` **在构造 `StreamingResponse` 之前先预取第 1 个 Token Chunk**。
   * 若后端在 Prefill 阶段超时或崩溃（还没吐出第 1 个字），Gateway 此时**尚未向客户端发送 `HTTP/1.1 200 OK` 状态行**！因此 Gateway 可以名正言顺地改发 `HTTP/1.1 502 Bad Gateway` 或 `504 Gateway Timeout` + 标准错误 JSON。
3. **流中途失败（`RequestState.STREAMING`）—— 覆水难收，严禁二次发送状态头**：
   * 一旦第 1 个字吐出，ASGI 协议的 `http.response.start` 事件（携带 `status=200`）已经刷入操作系统网卡缓冲区发给了客户端。
   * 如果吐到第 50 个字时后端 vLLM 进程突然崩溃（抛出 `httpx.ReadError`）：
     * **在协议层**：TCP 流已经处于第 4 阶段（Chunked Body 传输中），你**绝对不可能时光倒流去撤回客户端已经收到的 `HTTP/1.1 200 OK`**；如果强行在 ASGI 中再次调用 `http.response.start` 或抛出未捕获异常，Uvicorn/Starlette 会直接报 `RuntimeError: Expected ASGI message 'http.response.body', but got 'http.response.start'` 并粗暴断开 TCP 连接，导致客户端报 `ChunkedEncodingError / IncompleteRead`。
     * **工业级正确处理**：保持 HTTP 200 流不断，在生成器 `except` 块中向 SSE 流内注入一条**带内错误事件帧（In-band SSE Error Event）**：
       ```
       data: {"error": {"message": "Upstream stream interrupted...", "type": "mid_stream_error", "request_id": "..."}}\n\n
       data: [DONE]\n\n
       ```
       随后在 `finally:` 中调用 `await upstream_resp.aclose()` 优雅闭合流。

---

#### 💡 核心解析二：M1 Step 1 五大强制失败测试与零泄漏实测证据链

测试代码位于 `tests/test_gateway_step1.py`，实测结果已自动持久化至 `results/gateway/step1_failure_tests.json`：

| 失败测试场景 | 对应测试函数 | 核心验证指标与实测结果 (`step1_failure_tests.json`) | 是否有资源/连接泄漏？ |
| :--- | :--- | :--- | :---: |
| **1. 首字前失败**<br>*(Backend 在吐第 1 个字前超时或崩溃)* | `test_3_pre_first_token_failure_returns_http_error_json` | • TTFT 超时返回标准 `HTTP 504` JSON<br>• 首字前流崩溃返回标准 `HTTP 502` JSON<br>• `upstream_stream_closed: true` | **零泄漏**<br>(`active_requests = 0`) |
| **2. 流中途失败**<br>*(已发 HTTP 200 头，吐完 2 个字后崩溃)* | `test_4_mid_stream_failure_after_headers_sent` | • 状态码保持 `200`<br>• 成功注入 `mid_stream_error` SSE 帧与 `[DONE]`<br>• `upstream_stream_closed: true` | **零泄漏**<br>(`active_requests = 0`) |
| **3. 客户端主动断连**<br>*(后端计划吐 100 字，客户端收 2 字后断连)* | `test_5_client_disconnect_cancels_and_closes_upstream` | • 客户端在收到 `2` 个 chunk 后发送 `http.disconnect`<br>• 后端实际仅生成 **`2` 个 Token** 即被 `aclose()` 掐断（节省 98% 无效算力！）<br>• `upstream_stream_closed: true` | **零泄漏**<br>(`active_requests = 0`) |
| **4. 慢客户端读取**<br>*(客户端每 chunk 延迟 15ms 慢速消费)* | `test_6_slow_client_reading_and_high_concurrency_no_leak` | • 逐块按需拉取，无网关内存暴涨<br>• 读取完毕后 `all_upstream_streams_closed: true` | **零泄漏**<br>(`active_requests = 0`) |
| **5. 多请求高并发**<br>*(25 路并发流式请求混合快慢客户端)* | `test_6_slow_client_reading_and_high_concurrency_no_leak` | • 25 条上游流 `100%` 执行 `aclose()` 归还连接池<br>• 压测结束后 `leaked_active_requests: 0` | **零泄漏**<br>(`active_requests = 0`) |

---

### Q7: 为什么必须在 Gateway 层决定「接受、排队还是拒绝」后 Router 才有可用的负载状态？llm-d EPP 的 Flow Control 四大核心概念（Saturation / Flow / Priority / Fairness）是什么？

#### ❓ 问题背景
在进入 **M1 Step 2（准入控制、有界队列、公平性与过载反压）** 时，学习计划提出两条核心指引：
1. **为什么现在学**：*“当到达请求超过 backend 能处理的并发能力时，单有 Gateway 只会无限等待或超时；必须先决定「接受、排队还是拒绝」，Router 才有可用的负载状态。”*
2. **去哪学**：*“先复用 Step 1 的 asyncio/HTTPX 超时与测试；参考 [llm-d EPP](https://llm-d.ai/docs/dev/architecture/core/router/epp/) 的 Flow Control 概念（`flow`、`priority`、`fairness`、`saturation`），只读设计思想，不复制它的整个框架。”*

为什么不能把请求一股脑全部直接塞进 vLLM 自己的队列里？为什么不做网关层准入控制，后面的多节点 Router（M2）就会失效？

---

#### 💡 核心解析一：为什么不能把请求全塞给 vLLM？什么是“调度悔恨（Scheduling Regret）”？

##### 1. 单纯透传（Unbounded Forwarding）在洪峰下的雪崩
假设单台 vLLM 显卡同时最多能无抢占流畅运行 **8 个并发请求**（`max_in_flight = 8`）。若瞬间涌入 **100 个请求**，而 Gateway 不做拦截直接透传：
* **有效吞吐（Goodput）归零**：92 个请求在 vLLM 内部等待队列里积压数十秒。当排到第 80 个请求时，客户端早已触发超时断开；若未及时在网关层排队阶段将其剔除，GPU 仍会为已超时的死请求耗费算力。
* **队头阻塞（Head-of-Line Blocking）**：第 100 个高优在线实时对话请求，被迫排在 99 个离线长文批处理请求后面等待。

##### 2. 为什么说“必须先在网关决定接受、排队还是拒绝，Router 才有可用的负载状态”？
在分布式推理架构（如 [llm-d EPP](https://llm-d.ai/docs/dev/architecture/core/router/epp/)）中，把未获得执行槽位的请求过早发往某个具体 Backend 实例，会导致严重的 **“调度悔恨（Scheduling Regret / 过早绑定）”**：

```text
❌ 过早绑定（不在网关排队，直接发往后端）：
Client (100 reqs) ──► Router ──┬──► vLLM-A (瞬间塞入 50 个，全锁死在 A 的本地队列里)
                               └──► vLLM-B (瞬间塞入 50 个，全锁死在 B 的本地队列里)
```
* **灾难推演**：
  * 假设第 0 毫秒时，Router 把 50 个请求发给了 `vLLM-A`，50 个发给了 `vLLM-B`。
  * 结果 `vLLM-A` 碰巧遇到 5 个 32K 超长上下文请求，需要算 60 秒；而 `vLLM-B` 全是短对话，2 秒钟就把 50 个请求全部算完，**从第 3 秒起 `vLLM-B` 彻底空闲**！
  * 然而，那 45 个已经发往 `vLLM-A` 本地队列的请求**已经被死死锁在 `vLLM-A` 进程里，无法改派到空闲的 `vLLM-B`**！
* **正确解法（延迟绑定 Late Binding + 网关层准入控制）**：
  * **接受（Accept）**：网关严格控制每台 vLLM 的在途并发上限 `in_flight <= max_in_flight`；未饱和时直接放行（0 排队延迟）。
  * **排队（Queue in Gateway）**：当所有后端实例均已达到 `max_in_flight` 饱和水位时，多出来的请求**暂存在网关自己的有界内存队列（Bounded Queue）中**，暂不绑定死某台后端。一旦 `vLLM-B` 率先释放出 1 个槽位，网关立即从队头取出最高优请求派发给 `vLLM-B`！
  * **拒绝（Reject / Shedding）**：当租户超配额时当场返回 `429 Too Many Requests`；当网关有界队列已满或排队等待超时时，当场返回 `503 Service Unavailable`，保护后端引擎不发生级联 OOM 雪崩。
  * **结论**：只有网关把每台 Backend 的并发水位严格控制在 `max_in_flight` 以内，每台节点的负载状态才是干净、真实、可度量的，后续 M2 的智能 Router 才能在“真正有槽位空出的一瞬间”做出最优路由决策。

---

#### 💡 核心解析二：llm-d EPP 的 Flow Control 四大核心概念拆解

**llm-d**（Kubernetes 原生分布式推理栈）中的 **EPP（Endpoint Picker）** 将网关流控抽象为以下四个正交维度：

| 核心概念 | 解决的核心痛点 | 运行机制与工程语义 |
| :--- | :--- | :--- |
| **1. `Saturation`<br>（饱和度检测）** | 平时闲着的时候为什么要排队白加延迟？ | **Work-Conserving（有空即走）**：当后端 `in_flight < max_in_flight` 时，流控器完全旁路，请求 0 延迟直通后端；**仅当 `in_flight == max_in_flight`（进入饱和态）时**，才激活网关内存队列。 |
| **2. `Flow`<br>（逻辑流分类）** | 所有请求混作一团，分不清是谁发的、急不急 | 为每个进入网关的请求打上身份标签 **`FlowKey = (Tenant_ID, Priority)`**（例如通过请求头 `X-Tenant-ID` 与 `X-Priority`），将匿名洪流切分为可独立计量与调度的逻辑流。 |
| **3. `Priority`<br>（优先级分档）** | 离线跑批任务把在线 VIP 用户的交互卡死 | **严格优先级分档（Priority Bands，如 `high` > `normal` > `low`）**：<br>• **出队顺序**：后端释放槽位时，严格先清空高优分档，再处理低优分档。<br>• **过载丢弃（Shedding）**：队列接近满载时，优先拒绝低优先级请求返回 `503`。 |
| **4. `Fairness`<br>（同级租户公平性）** | **防“吵闹邻居（Noisy Neighbor）”**：同一优先级下，租户 A 狂发 100 个请求，把只发 1 个请求的租户 B 堵死 | **同级内按租户轮转（Per-Flow Round-Robin / WFQ）**：<br>• 在同一个 Priority 分档内，每个租户维护独立子队列 `Queue[tenant_id]`。<br>• 每次释放槽位时，在有积压请求的租户之间 **Round-Robin 轮流取 1 个请求**（先放 A 的第 1 个 → 紧接着放 B 的第 1 个 → 再放 A 的第 2 个），彻底消除单租户霸占队列。 |

---

#### 💡 核心解析三：为什么“只读设计思想，不复制它的整个框架”？

* **原版框架的复杂度**：`llm-d EPP` 是用 Go 编写的 Kubernetes 重型组件，深度耦合了 Envoy `ext_proc` gRPC 流协议与 K8s CRD 控制器，代码量达上万行。
* **我们在 Python Gateway 中的精炼落地（< 200 行原生 `asyncio`）**：
  我们只需提取上述四大思想，在 `gateway/` 中实现一个轻量内聚的 **`AdmissionController`**：
  1. **租户配额（Tenant Quota）**：限制单租户最大并发/速率，超限立即返回 **`429 Too Many Requests`**。
  2. **饱和与有界队列（Saturation & Bounded Queue）**：跟踪 `in_flight` 与 `max_in_flight`；饱和时进入有界队列（上限 `max_queue_size`），队列满或等待超过 `queue_timeout` 立即返回 **`503 Service Unavailable`**。
  3. **Priority + Per-Tenant Round-Robin 调度器**：按优先级分桶，同优先级内按 `tenant_id` 循环轮转唤醒等待中的 `asyncio.Future`；若客户端在排队等待期间主动断连，立即从队列中移除该条目，绝不浪费后续释放的后端槽位。

---

### Q8: 既然 vLLM 自己内部已经有 `waiting` 和 `running` 调度队列，为什么 Gateway 的队列必须独立于它们？

#### ❓ 问题背景
在 **M1 Step 2（准入控制与有界队列通关验收）** 中，核心考察点是：
> *“能解释 Gateway 队列为何独立于 vLLM 的 `waiting` / `running`；有代码与自动化测试、`queue_depth` / `queue_wait` / `reject` / `cancel` 指标；无效请求不会进入后续推理队列。”*

既然 vLLM 的 Scheduler 内部本来就维护了 `waiting`（等待队列）和 `running`（执行队列），为什么不能直接把所有请求全扔进 vLLM 的 `waiting` 队列，而非要在 Gateway 层再独立维护一层有界队列？

---

#### 💡 核心解析：两层队列管理的物理资源与职责边界完全不同

```text
                        【第 1 道闸门：业务流控、多租户治理与集群级延迟绑定】
客户端洪峰 ──► [ Gateway 独立有界队列 (queue_depth <= queue_capacity) ]
                 │  • 管什么：400 Schema 校验、429 租户配额/速率桶、Priority 防饥饿、租户 Round-Robin、排队断连摘除
                 │  • 调度单位：HTTP JSON 请求 (未分词、未绑定特定 GPU 节点、未占显存)
                 ▼ (严格限流 max_inflight，仅放行拿到槽位的有效请求！)
              [ vLLM 内部 Scheduler (waiting / running) ]
                    • 管什么：Token 级 Continuous Batching、PagedAttention 物理 KV 显存块分配与抢占 (Preemption)
                    • 调度单位：SequenceGroup / Token IDs & Physical KV Cache Blocks
```

Gateway 队列必须独立于 vLLM `waiting` / `running` 队列的 **四大核心原因**：

##### 1. 职责与调度粒度不同（业务多租户治理 vs 物理显存块拼图）
* **vLLM 的 `waiting` / `running` 是“GPU 显存块调度器”**：
  * 请求进入 vLLM 后先做 Tokenizer 分词放入 `waiting`；每个 Step（约 10~20ms），vLLM Scheduler 根据 **物理 GPU KV Cache Block** 剩余数量决定能否将请求从 `waiting` 拉入 `running`；若 Decode 途中显存耗尽，还会将 `running` 中的请求抢占（Preempt）退回 `waiting`。
  * vLLM 的 `waiting` 队列**不具备多租户业务感知能力**：它不知道什么叫“单租户超配额返回 `429`”、不知道“同优先级下租户 A 与租户 B 必须 Round-Robin 交替执行防吵闹邻居”、也不知道“高优先级连续放行 3 次后必须让 1 次给低优先级防永久饥饿”。
* **Gateway 的队列是“多租户业务流控器”**：
  * 在请求尚未进入推理引擎之前，基于 `FlowKey = (tenant_id, priority)` 完成配额限流、优先级提权与租户间公平调度。

##### 2. 将“无效 / 过期 / 排队断连”请求 100% 拦截在分词与推理队列之外
* 若没有 Gateway 独立队列，所有请求直接涌入 vLLM：
  * 哪怕请求 JSON 格式非法（`400`）、租户超限（`429`）、进门前或排队期间 Deadline 已过期（`504`）、或者客户端在排队中途早已断开连接（`499`），vLLM 进程仍需耗费 CPU 执行 HTTP 解析、甚至调用 HuggingFace Tokenizer 对长文本分词并塞入 `waiting` 队列。
* **有了 Gateway 独立队列**：
  * 非法请求（`400`）、超额请求（`429`）、队列满溢出请求（`503`）、过期请求（`504`）与排队中途断连请求（`499`）**100% 在 Gateway 层被当场拦截或从队列摘除（`invalid_and_rejected_entered_backend_count == 0`）**，绝不污染后端的推理队列。

##### 3. 在多实例场景（M2 Router）下消除“调度悔恨（Scheduling Regret / 过早绑定）”
* vLLM 的 `waiting` 队列是**单实例私有的（Instance-Local）**。一旦过早把请求塞进 `vLLM-A` 的 `waiting` 队列，即使隔壁 `vLLM-B` 瞬间空闲下来，该请求也无法从 `vLLM-A` 迁移到 `vLLM-B`。
* Gateway 的有界队列是**集群级中心化、延迟绑定（Late-Binding）**的：请求在 Gateway 队列等待时不绑定具体节点，直到某台 vLLM 真正释放 `in_flight` 槽位的那一瞬间，才由 Router 派发过去。

##### 4. 保护 vLLM 主进程 Event Loop 与 CPU 分词线程，避免产生 GPU 调度气泡
* vLLM API Server 与 Tokenizer 运行在 CPU 上。如果让成百上千个并发请求直接堆积在 vLLM 内部，海量挂起协程与分词任务会争抢 CPU 与 GIL，拖慢主进程向 GPU 发射下一个 Step Kernel 的速度，导致正在 `running` 的请求出现 **Step 间气泡（TPOT 劣化）**。Gateway 用 `max_inflight` 锁死进入每台 vLLM 的并发上限，让 vLLM 的 `waiting` 队列常年保持极低水位，专心满负荷驱动 GPU。







