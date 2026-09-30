# 单 Batch 专家组通信–计算流水设计

状态：Draft v0.8（使用 DeepEP v2 原生异步接口，CUDA Graph 外执行 overlap）
目标系统：SGLang MoE + Expert Parallelism
方案名称：Expert-Group Pipeline，简称 EGP

## 1. 背景

当前 MoE 层按以下顺序执行：

```text
Router/TopK
    ↓
Dispatch 全部专家路由
    ↓
计算全部本地专家
    ↓
Combine 全部结果
```

对应延迟近似为：

\[
T_{\text{baseline}} = T_D + T_C + T_K
\]

其中 \(T_D\) 是 dispatch 通信，\(T_C\) 是专家计算，\(T_K\) 是 combine 通信。

Two-Batch Overlap 通过拆分 token batch，在两个 batch 之间交错通信和计算。本设计采用另一种方式：保持 batch 不变，将每张 GPU 上的物理专家划分成多个 group，在同一个 MoE 层内部构造流水线。

```text
Dispatch G0
Dispatch G1  || Compute G0
Combine G0   || Compute G1
Combine G1
```

目标是隐藏部分 dispatch 和 combine 延迟，同时避免 TBO 引入的 batch 切分和跨层调度复杂度。

## 2. 目标与非目标

### 2.1 目标

1. 在单个 MoE batch 内实现通信–计算 overlap。
2. 保持 MoE 数学语义不变。
3. CUDA Graph capture/replay 留作后续目标，当前不支持。
4. 与 EPLB 的 logical-to-physical 映射兼容。
5. 初始实现支持两个专家组。
6. 保证所有 EP rank 以相同顺序执行 collective。
7. 可以通过配置关闭，并可靠回退到现有 MoE 路径。

### 2.2 非目标

首版不考虑：

- 与 TBO 同时开启；
- 动态调整 group 数量；
- CPU 根据每个 batch 的 TopK 结果生成路由；
- 任意非连续专家分组；
- 所有 A2A 和 MoE runner backend；
- 在首版中修改 EPLB 的布局优化目标。

## 3. 初始支持范围

建议 MVP 范围：

| 维度 | MVP |
|---|---|
| 平台 | NVIDIA CUDA |
| A2A backend | 仅 DeepEP v2 (`deepep_v2` / `ElasticBuffer`) |
| DeepEP v2 模式 | 沿用 `deepep_v2_mode`；decode 使用 expanded/masked，prefill 使用 contiguous |
| 量化 | 128×128 blockwise FP8，dynamic activation scaling |
| MoE runner | DeepGEMM masked/contiguous |
| Group 数量 | 固定为 2 |
| CUDA Graph | 当前禁用 decode/prefill graph |
| EPLB | 支持基于物理槽位分组 |
| TBO | 与 EGP 互斥 |
| Shared expert | 首版保持现有独立路径 |
| 动态 shape | 通过 padding/capture-size bucket 处理 |

建议的临时配置参数：

```text
--enable-moe-expert-group-overlap
--moe-expert-group-count 2
```

阶段一实现已加入这两个实验参数；默认关闭，当前仅支持 group count 为 2。

启用 EGP 后统一通过 `ExpertGroupExecutor.run()` 执行双流 overlap，
不再提供执行方式选项或串行分支。仅支持 `--moe-a2a-backend deepep_v2`，
不依赖旧版 `--deepep-mode`。启用 overlap 时自动禁用 decode 和 prefill
CUDA Graph。普通 deepep_v2
路径的 decode CUDA Graph 支持保持不变。EPLB 尚未支持。
Prefill 的 contiguous dispatch 仍有 CPU 同步，实际重叠程度需要 GPU profiling。

```bash
# 在原有模型、TP/EP 启动参数上追加：
--enable-moe-expert-group-overlap \
--moe-expert-group-count 2 \
--moe-a2a-backend deepep_v2 \
--moe-runner-backend deep_gemm \
--disable-shared-experts-fusion
```

数值对照时关闭 `--enable-moe-expert-group-overlap`，使用普通 DeepEP v2
路径，并保持相同输入、权重、路由和通信 dtype；对照路径也关闭 CUDA Graph。

## 4. 专家分组模型

设：

```text
P = 全局物理专家数
R = EP rank 数
L = P / R，每个 rank 的本地物理专家数
G = group 数
S = L / G，每个 rank、每个 group 的专家数
```

要求：

```text
P % R == 0
L % G == 0
```

以 32 个全局物理专家、EP2、两个 group 为例：

```text
P = 32
R = 2
L = 16
G = 2
S = 8
```

分组如下：

```text
group0:
  rank0 physical  0~7
  rank1 physical 16~23

group1:
  rank0 physical  8~15
  rank1 physical 24~31
```

每个 group 必须在所有 rank 上都包含专家。不能将 group0 定义为“GPU0 上的所有专家”，否则不同 GPU 之间会产生计算空洞。

### 4.1 分组依据

首版按照本地物理槽位分组：

\[
group(p) =
\left\lfloor
\frac{p \bmod L}{S}
\right\rfloor
\]

这样做有以下优点：

- 不依赖逻辑专家 ID；
- EPLB 迁移专家后仍然有效；
- 每组权重在本地 tensor 中连续；
- 方便直接切分专家权重；
- 适合 grouped GEMM。

## 5. 路由 ID 压缩

每个 group 独立执行 dispatch 时，将专家 ID 压缩到该 group 的专家空间。

对于物理专家 ID \(p\)：

```text
rank       = p // L
local_slot = p % L
group      = local_slot // S
offset     = local_slot % S
```

该专家在 group 内的压缩全局 ID 为：

\[
compact(p) = rank \times S + offset
\]

每个 group dispatcher 使用：

```text
num_global_experts = R * S
num_local_experts  = S
```

例如 physical expert 26：

```text
rank       = 26 // 16 = 1
local_slot = 26 % 16 = 10
group      = 10 // 8 = 1
offset     = 10 % 8 = 2
compact_id = 1 * 8 + 2 = 10
```

group1 dispatch 时使用 expert ID 10，而不是 26。

## 6. Graph-safe 路由拆分

### 6.1 输入

路由拆分发生在 EPLB logical-to-physical remap 之后：

```text
physical_topk_ids: [T, K]
topk_weights:      [T, K]
```

其中：

- `T` 是 capture 后的 padded token 数；
- `K` 是 TopK；
- `-1` 表示无效或 padding 路由。

现有 logical-to-physical 入口是 `python/sglang/srt/eplb/expert_location_dispatch.py` 中的 `topk_ids_logical_to_physical()`。

### 6.2 输出

首版使用固定 shape：

```text
group_topk_ids:     [G, T, K]
group_topk_weights: [G, T, K]
```

对于不属于当前 group 的路由：

```text
topk_id     = -1
topk_weight = 0
```

### 6.3 GPU kernel

伪代码：

```python
def split_physical_routes(
    physical_topk_ids,
    topk_weights,
    num_local_experts,
    experts_per_group,
    num_groups,
    output_ids,
    output_weights,
):
    for token, slot in parallel:
        physical_id = physical_topk_ids[token, slot]
        weight = topk_weights[token, slot]

        for group in range(num_groups):
            output_ids[group, token, slot] = -1
            output_weights[group, token, slot] = 0

        if physical_id < 0:
            return

        rank = physical_id // num_local_experts
        local_id = physical_id % num_local_experts
        group = local_id // experts_per_group
        offset = local_id % experts_per_group
        compact_id = rank * experts_per_group + offset

        output_ids[group, token, slot] = compact_id
        output_weights[group, token, slot] = weight
```

正式实现使用单个 Triton/CUDA kernel，不使用多个 `torch.where` kernel。

### 6.4 CUDA Graph 要求

路由拆分必须满足：

- 逐 token 路由完全在 GPU 计算；
- 输入、输出地址固定；
- 输出 shape 固定；
- 不调用 `nonzero()` 或动态 boolean indexing；
- 不将 route count 搬到 CPU；
- 不根据 group 是否为空执行 Python 分支；
- kernel 对 Torch Compile 注册为 custom op。

## 7. 执行流水线

定义：

```text
Dg = group g dispatch
Cg = group g expert compute
Kg = group g combine
```

两个 group 的目标时序：

```text
时间 ─────────────────────────────────────────────>

Comm stream:
        D0 ───────── D1 ───────── K0 ───────── K1
                       ▲             ▲
                       │             │
Compute stream:
             wait(D0)  C0 ── wait(D1) C1
```

依赖关系：

```text
D0 -> C0 -> K0
D1 -> C1 -> K1

D0 -> D1           通信流顺序
D1 || C0           第一处 overlap
K0 || C1           第二处 overlap
K0,K1 -> final sum
```

理想延迟近似为：

\[
T_{\text{EGP}}
\approx
D_0 +
\max(D_1,C_0) +
\max(K_0,C_1) +
K_1 +
T_{\text{split}} +
T_{\text{extra}}
\]

其中 `T_extra` 包含额外 kernel launch、metadata 和小 GEMM 效率损失。

## 8. Dispatcher 状态设计

当前 DeepEP v2 dispatcher 将通信状态保存在实例成员中：

```text
self._impl._handle
self._impl._pad_empty_combine
```

因此一个 dispatcher 实例不能同时保存多个 in-flight group。

### 8.1 MVP 方案

每个 group 创建独立 dispatcher context：

```python
class ExpertGroupPipelineDispatcher:
    dispatchers: list[DeepEPv2Dispatcher]
```

初始化：

```python
dispatchers = [
    DeepEPv2Dispatcher(
        num_experts=ep_size * experts_per_group,
        num_local_experts=experts_per_group,
        ...
    )
    for _ in range(num_groups)
]
```

每组 dispatcher 注入独立的 `DeepEPv2Buffer` pool，使用不同 runtime resource key，
并为 `ElasticBuffer` 设置 `prefer_overlap_with_compute=True`。普通 v2 路径仍使用默认 pool。

### 8.2 长期方案

将通信状态从 dispatcher 实例移到输出对象：

```python
@dataclass
class GroupDispatchOutput:
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    dispatch_handle: object
    ready_event: object
    group_id: int
```

然后使用显式 handle：

```python
combine(group_output, group_output.dispatch_handle)
```

这样 dispatcher 可以无状态化，支持任意数量的 in-flight 通信。但此重构不属于 MVP。

## 9. 建议执行接口

```python
class ExpertGroupPipeline:
    def split_routes(
        self,
        physical_topk_ids,
        topk_weights,
    ) -> GroupedTopKOutput:
        ...

    def dispatch_async(
        self,
        group_id,
        hidden_states,
        group_topk_output,
    ) -> GroupDispatchState:
        ...

    def compute(
        self,
        group_id,
        dispatch_state,
    ) -> GroupCombineInput:
        ...

    def combine_async(
        self,
        group_id,
        combine_input,
    ) -> GroupCombineState:
        ...

    def finalize(
        self,
        group_combine_states,
    ) -> torch.Tensor:
        ...
```

MoE forward 从当前的：

```python
dispatch_output = dispatcher.dispatch(hidden_states, topk_output)
combine_input = run_moe_core(dispatch_output)
output = dispatcher.combine(combine_input)
```

改为专用 EGP 路径：

```python
grouped_topk = pipeline.split_routes(
    physical_topk_ids=topk_output.topk_ids,
    topk_weights=topk_output.topk_weights,
)

d0 = pipeline.dispatch_async(0, hidden_states, grouped_topk[0])
d1 = pipeline.dispatch_async(1, hidden_states, grouped_topk[1])

c0 = pipeline.compute(0, d0)
k0 = pipeline.combine_async(0, c0)

c1 = pipeline.compute(1, d1)
k1 = pipeline.combine_async(1, c1)

output = pipeline.finalize([k0, k1])
```

## 10. 专家权重切分

当前实现由 `ExpertGroupExecutor` 统一持有 group count/size、expert slices、
分组 runner config、各组 runner 和 `ExpertGroupDispatcher`。
Executor 不创建通信 stream，使用 DeepEP 原生通信流及完成事件。
初始化时注入 dispatcher factory，避免 executor 反向依赖 `FusedMoE`。

`FusedMoE` 调用 `executor.run(layer, hidden_states, topk_output)`；executor
内部拆分路由，通过 `_compute_group()` 调用独立的
`get_expert_group_quant_info(..., expert_slice=...)` 获取当前权重的分组量化
信息并执行计算；普通 `get_moe_quant_info()` 接口保持原有语义。
只缓存 slice，不缓存权重 view 或 `quant_info`，以适配加载和量化后处理中的
权重张量替换。`layer.dispatcher` 指向 executor 持有的同一个 dispatcher，
保持量化配置设置入口。

DWDP、symmetric memory、输出裁剪和 all-reduce 留在层上；DWDP 的消费完成
事件在 executor 返回、通信流 join 已提交之后记录。


本地专家权重通常具有：

```text
[num_local_experts, ...]
```

连续分组时：

```python
start = group_id * experts_per_group
end = start + experts_per_group

gate_up_group = gate_up_weight[start:end]
down_group = down_weight[start:end]
```

要求：

1. weight view 不产生运行时复制；
2. view 在 CUDA Graph capture 前构造；
3. EPLB 更新仍然写入原 physical slot；
4. group 内 compact local ID 与 weight view 下标一致。

关键不变量：

```text
group compact local id i
    对应
原 local physical slot group_start + i
```

## 11. Combine 与最终输出

每个 group 仅携带属于自身的 top-k weight：

```text
非本组 route:
  expert_id = -1
  weight = 0
```

每个 combine 返回：

```text
partial_output_g: [T, H]
```

最终结果：

```python
output = partial_output_g0 + partial_output_g1
```

需要确认以下操作只执行一次：

- routed scaling factor；
- residual add；
- TP/EP all-reduce；
- shared expert add；
- padding token 截断。

建议 group combine 只完成 routed expert 的加权回传，所有 MoE 层级的后处理在 partial output 求和之后执行。

## 12. EPLB 协同

EGP 在 EPLB physical remap 之后工作：

```text
logical TopK
    ↓ EPLB
physical TopK
    ↓ EGP split
group + compact expert ID
```

因此 EPLB 不需要知道 dispatcher 的 compact ID。

### 12.1 首版行为

group 绑定物理槽位：

```text
group = local_physical_slot // experts_per_group
```

EPLB 可以自由改变：

```text
physical slot -> logical expert
```

EGP 分组不变。

### 12.2 潜在问题

当前 EPLB 平衡的是 GPU 总负载，并不保证 group 负载平衡：

```text
GPU0 group0 很热，group1 很冷
GPU1 group0 很冷，group1 很热
```

collective group 的完成时间由最慢 rank 决定，因此这种布局会降低流水线效果。

### 12.3 后续优化

增加 group-aware EPLB 目标：

1. 先平衡节点和 GPU 总负载；
2. 再在每个 GPU 内将物理副本二路 bin packing；
3. 让每个 group 的预计负载跨 rank 接近；
4. 保持每组物理槽位连续。

首版只记录 group load，不修改 EPLB 算法。

## 13. CUDA Graph 设计（后续方案，当前不支持）

### 13.1 Graph 内执行

以下操作进入 CUDA Graph：

```text
Router/TopK
logical-to-physical remap
physical-to-group split
D0/D1
C0/C1
K0/K1
partial output sum
```

### 13.2 CPU 负责

CPU 只负责：

- 配置 group 数量；
- 创建 dispatcher context；
- 创建固定 buffer；
- 生成静态 group policy；
- 选择 capture size；
- EPLB rebalance 后在安全点更新 metadata。

CPU 不负责当前 batch 的 `routes_g0/routes_g1`。

### 13.3 固定 buffer

每个 capture size 预分配：

```text
group_topk_ids[G,T,K]
group_topk_weights[G,T,K]
partial_outputs[G,T,H]
group events/handles
```

必须避免：

```python
routes = topk_ids[group_mask]
```

因为该操作会产生数据依赖的动态 shape。

### 13.4 多流 capture

多流 capture 必须形成完整 fork/join：

```text
capture stream
    ├── comm stream
    └── compute stream
          ↓
capture stream wait/join
```

capture 结束前，所有 side stream 必须通过 event 或 `wait_stream` 汇合回主 capture stream。

每个 group 的 DeepEP v2 graph state、handle 和 workspace 必须独立，或者经验证可以安全共享。

## 14. 通信放大风险

DeepEP v2 contiguous 路径可能按目标 rank 合并发送 hidden state。拆成 group 后，一个 token 如果在同一 rank 命中两个 group，可能发送两次 hidden state。

定义通信放大率：

\[
A_{\text{comm}} =
\frac{\text{EGP 实际发送字节}}
     {\text{原始 dispatch 实际发送字节}}
\]

必须在 benchmark 中记录该值。

建议准入条件：

```text
A_comm <= 1.2
```

具体阈值需实验确定。首版固定使用两个 group，避免 group 数过多导致通信快速膨胀。

## 15. 资源竞争

DeepEP v2 通信 kernel 可能消耗 SM，专家 GEMM 也会占用 SM。两个 kernel 出现在不同 stream 上不代表真正获得性能收益。

需要调节：

```text
SGLANG_DEEPEP_V2_NUM_SMS
DeepGEMM num_sms
comm stream priority
compute stream priority
```

观察指标：

- dispatch 单独执行时间与 overlap 后执行时间；
- GEMM 单独执行时间与 overlap 后执行时间；
- SM occupancy；
- NVLink/RDMA throughput；
- HBM bandwidth；
- 最终 MoE 层延迟。

如果 overlap 后两个 kernel 都显著变慢，则需要进一步限制各自的 SM 数量。

## 16. 正确性不变量

实现必须满足：

1. 每个有效 `(token, topk-slot)` 恰好属于一个 group。
2. 无效 route 在所有 group 中都保持无效。
3. 所有 group 的 route 并集等于原 route。
4. 不同 group 的有效 route 交集为空。
5. compact ID 可以唯一映射回原 physical ID。
6. compact local ID 与该 group 的权重切片一致。
7. 所有 EP rank 以相同顺序执行所有 group collective。
8. 即使某 rank 的某 group 没有 token，也必须参与 collective。
9. 所有 group partial output 求和等价于原 MoE output。
10. EPLB rebalance 后，物理槽位权重与 group weight view 保持一致。

## 17. 测试计划

### 17.1 单元测试

覆盖：

```text
EP2 / 32 physical experts / 2 groups
不同 top-k
包含 -1 padding
token 同时命中两个 group
token 在同一 rank 命中两个 group
某 group 全空
某 rank 的某 group 全空
EPLB replicated experts
```

验证：

```python
baseline_output ≈ egp_output
```

同时检查 route 并集、交集和 compact-ID 映射。

### 17.2 分布式正确性测试

至少覆盖：

- EP2 单机；
- EP8 单机；
- 多节点 EP；
- rank token 数不一致；
- 空 token rank；
- CUDA Graph capture/replay 多轮；
- EPLB rebalance 前后；
- 相同 graph 使用不同 TopK 路由数据。

### 17.3 数值容差

BF16/FP16：

```text
rtol = 1e-2
atol = 1e-2
```

FP8/FP4 使用现有 MoE backend 的测试容差。还需要检查确定性模式下，多次 replay 输出一致。

### 17.4 性能测试

Sweep：

```text
groups: 1, 2, 4
batch size
prefill token 数
decode batch size
top-k
hidden size
intermediate size
均匀/偏斜路由
EPLB on/off
CUDA Graph on/off
```

记录：

```text
MoE layer latency
端到端 token latency
dispatch/combine 时间
group compute 时间
通信字节数
通信放大率
GEMM occupancy
graph replay latency
额外显存
```

## 18. 回退条件

遇到以下情况自动回退到原始 MoE 路径：

- `num_local_physical_experts % group_count != 0`；
- backend 不支持多个 in-flight dispatcher；
- 当前 runner 不支持 weight slicing；
- batch/token 数超过已捕获 graph 范围；
- group workspace 初始化失败；
- CUDA Graph capture 失败；
- 与 TBO 同时启用；
- EPLB/elastic EP 处于布局切换阶段。

开发阶段建议直接报错，稳定后再增加自动回退。

## 19. 实施阶段

### 阶段一：串行正确性原型（历史验证方案）

当前生产调用已移除串行分支，统一使用 `ExpertGroupExecutor.run()`。
仍需在 NVIDIA EP2
环境完成 DeepEP v2 + DeepGEMM 的分布式数值对比。

实现：

```text
split G0/G1
D0 -> C0 -> K0
D1 -> C1 -> K1
sum
```

目标：验证 route split、compact ID、weight slicing 和 combine 语义。

### 阶段二：ExpertGroupExecutor overlap

实现状态：使用 DeepEP v2 原生 `async_with_compute_stream=True`，
Executor 不创建或切换外层 comm stream。普通 deepep_v2 的 dispatch/combine
接口继续在返回前提交完成等待，分组路径通过 `dispatch_a` 准备输入并提交通信，通过 `dispatch_b` 等待和整理结果。
`_dispatch_core` 仅作为 impl 的内部实现，不对 executor 暴露。
`async_with_compute_stream` 在初始化时存入 impl；分组 dispatcher 设置为 True，
普通路径默认为 False，dispatch/combine 调用不再传入该参数。
两组 `dispatch_a` 均在等待 D0 和提交 C0 之前执行。
combine 使用 `combine_a` 提交、`combine_b` 等待。

提交顺序：

```text
dispatch_a G0（准备输入、记录事件、提交 D0）
dispatch_a G1（准备输入、记录事件、提交 D1）
dispatch_b G0（wait D0、整理结果）-> C0
K0 async（在 C1 提交前发起，计算流依赖只包含 C0）
wait D1 -> C1 -> K1 async
wait K0/K1 -> sum
```

传入 `previous_event` 的 dispatch 同时使用 `allocate_on_comm_stream=True`，
避免 D1 的接收分配复用计算流上仍由 C0 使用的存储。combine 在当前计算流
分配输出，由原生内部通信流完成写入，最终消费输出前等待其完成事件。

每组独立持有 ElasticBuffer 和 handle，相同配置的层复用各组 buffer。
Executor 保留输入和结果引用，各组 dispatcher 持有 handle 直到 combine 提交。
`dispatch_b` / `combine_b` 直接等待 DeepEP 返回的 event，不封装 Pending 对象。DeepEP 原生异步接口
负责跨流分配器生命周期。所有 rank 执行 D0、D1、K0、K1，空组不跳过。
Overlap prefill 使用 `do_cpu_sync=False`，接收 buffer 按最大容量分配；
普通 prefill 保持 `do_cpu_sync=True`。接收计数的 `.item()` 和裁剪延迟到
`dispatch_b` 等待原生 event 之后的结果整理；非 expand 模式同步裁剪
handle 的 `recv_src_metadata` 并更新 `num_recv_tokens`，保证 combine 输入与元数据行数一致。
`dispatch_b` 的 `.item()` 仍有主机等待，但 D0/D1 已提交，且读取 D1 计数前已提交 K0。
实际 overlap 仍需 GPU profiling。

阶段二检查还发现阶段一未处理 EPLB 接收计数的 group→physical-slot
转换：现有统计器假定一次接收的是全部本地专家计数。因此目前显式拒绝
EPLB 和 expert distribution recorder，阶段四再补齐；单纯物理 ID 分组
不足以声称完整 EPLB 集成已支持。

当前简化版本验证：15 项隔离 CPU 检查通过，包含 1 项初始化配置检查、4 项模拟原生事件的
decode/prefill（含空 rank）检查和 10 项现有 buffer 生命周期检查。
模拟检查覆盖连续调用、D0/D1/K0/K1 顺序、D1/C0 与 K0/C1 无额外依赖、
消费前等待、prefill 计数读取及 combine 后 handle 清理。
这类测试不证明实际 GPU 并行或速度收益。本机缺少 Triton/CUDA，完整
SGLang pytest 收集及 EP2 DeepEP v2/DeepGEMM 数值测试仍待 GPU 环境执行。
验收时先 warm up，再用相同输入对照普通 DeepEP v2 路径与 overlap 路径，覆盖不同 rank token
数、空 rank、全空 group、跨组 top-k、多轮执行；随后用 Nsight Systems
确认实际并行，并测量层延迟与额外通信 buffer 显存。

增加：

- 两个 dispatcher context；
- DeepEP 内部 comm stream / 调用者 compute stream；
- 输入就绪事件和原生完成事件；
- `D1 || C0`；
- `K0 || C1`。

目标：用 Nsight 验证真实 overlap 和延迟收益。

### 阶段三：Decode CUDA Graph（已移除支持）

当前 overlap 仅在 CUDA Graph 外执行。参数解析时禁用 decode graph，
prefill graph 由 deepep_v2 后端禁用。原生异步接口用于 CUDA Graph 外的 overlap。
普通 deepep_v2 路径继续支持 decode graph。

本节之前列出的 capture/replay 方案和相关性能指标仅作为后续设计目标，
不代表当前实现支持或已通过验收。

### 阶段四：EPLB 与性能优化

增加：

- EPLB rebalance 测试；
- per-group load 统计；
- group-aware placement；
- 动态启停策略；
- group 数量 autotune。

## 20. 验收标准

功能验收：

- 所有正确性和分布式测试通过；
- CUDA Graph replay 使用不同路由时结果正确；
- EPLB rebalance 前后无需重新构建整个模型；
- 空 group/rank 不死锁；
- 关闭 EGP 后行为与当前版本一致。

性能验收建议：

```text
目标 workload 的 MoE 层延迟下降 ≥ 10%
端到端延迟无回退
通信放大率在可接受范围
额外显存可控
```

如果只能观察到 timeline overlap，但 MoE 层总延迟没有下降，则不视为性能验收通过。

## 21. 核心决策总结

最终建议采用：

```text
按本地物理专家槽位做连续分组
        ↓
EPLB remap 后使用 GPU fused kernel 拆 route
        ↓
每个 group 使用独立 DeepEP v2 dispatcher context
        ↓
固定 shape 的 GPU 路由拆分，当前在 CUDA Graph 外执行
        ↓
D1 || C0，K0 || C1
        ↓
各 group partial output 求和
```

这条路径对现有架构侵入相对可控：路由拆分发生在 TopK 后处理与 dispatcher 之间，通信状态复用 TBO 的多 dispatcher 思路，权重侧使用连续 view，CUDA Graph 则通过固定 buffer 和 GPU-side route splitting 保持完整 capture。
