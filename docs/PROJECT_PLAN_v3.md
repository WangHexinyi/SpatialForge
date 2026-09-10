# SpatialForge 项目计划书 v3.1 — Architecture Revision

> **文档定位**：跨 AI 项目记忆、当前状态快照、研究路线与工程执行指南。  
> **不是不可修改的纲领**：其中任何架构、门控、实验和实现细节都可以基于新证据讨论、调整、替换。  
> **需要保存的依据**：用户研究愿景、实验事实、已通过门控和核心边界；Git 状态是带日期的快照，接手时必须重新核对。
> **最近更新**：2026-09-08；架构修订基线为 `3fb0e65`，经 God View G1 人类 UAT 验收后完成路线图重构，接手时通过 Git 重新核对当前 HEAD。
> **建议仓库路径**：`docs/PROJECT_PLAN_v3.md`  
> **公开仓库注意**：本文不记录任何私有 SSH 地址、密钥、令牌、API Key、账号凭据或云实例敏感信息。

---

> ## ★ 2026-09-09 ARCHITECTURE REVISION (v3.1 → Embodied ProcTHOR Mainline)
>
> **Human UAT 判定**：既有 procedural scene 是 toy-level training env；camera-centric / orbital
> trajectory 路线不是最终训练范式；既有 Inspector 前端存在交互 bug。正式训练主线换轨为：
>
> ```
> ProcTHOR / AI2-THOR
>   → Embodied First-person Agent
>   → AgentAction → Environment Transition → New Observation
>   → Memory / Episode History
>   → Object Search (Find <category>)
>   → Teacher / Behavior Cloning (privileged)
>   → World Representation / Active Observation
>   → Interactive Object Search
> ```
>
> **AGENT 是主体；Camera 是 Agent 的眼睛；God View 是 researcher/teacher/verifier。**
> 固定镜头 / Canonical 6/14/26 / Trajectory Lab 仅保留用于 calibration、debug、
> visualization、research inspection，**不再作为正式训练主范式**。
>
> ### Deprecation 记录（本轮更新）
> - **Procedural Toy World**（`challenge.py` scene_challenge_* / S0–S3 /
>   cube/sphere/cylinder/torus）：`DEPRECATED AS TRAINING SOURCE`。保留极少量
>   synthetic geometry fixture 仅用于 unit tests / camera math / projection
>   regression（这些 fixture 不属于训练数据）。
> - **Static / Orbital Camera Training**：`DEPRECATED AS PRIMARY TRAINING METHOD`；
>   保留为 research/calibration/debug infrastructure（历史 gate 证据保留，见 §8–§9）。
> - **God View / Research Inspector**：重新定位为 research infrastructure。
> - **旧 G2.1 Dynamic Viewpoint & Scene Challenge**：不再作为训练主任务承接对象，
>   其轨迹 / camera 研究保留于 research 层。
>
> ### 本轮交付（wip branch，未经 Git 提交）
> 1. `spatialforge/embodied/` 具身运行时：数据契约、动作空间、model-input 泄漏防护、
>    teacher 最短路、确定性 CPU harness 后端、真实 ProcTHOR/AI2-THOR 后端。
> 2. i18n 层（zh-CN / en / bilingual），Inspector 新 UI：顶部极简全局栏 + 右侧 7-tab Sidebar
>    （Observation / Agent / Task / Environment / Training / Debug / Settings）。
> 3. 新 Inspector server `embodied_server.py` + 双语前端 `embodied_web/`。
> 4. 旧 toy 世界从正式训练/产品 Environment 流程移出（training 主入口不再挂 challenge map）。
>
> 详细报告与运行状态见 §28 末尾 "Embodied Vertical Slice Record（2026-09-09）"。

---

# 0. 新会话 / 新 AI 快速拉起

**CURRENT / VALIDATED**：当前实现是受控静态多视角空间推理实验基础设施；长期定位是 **Spatial Intelligence Research Infrastructure（空间智能研究基础设施）**。具身 Agent、动态 world model、通用后端与分布式训练尚未实现。

```text
Review baseline: 2026-09-08 synchronized local repository
Branch: feat/g2.0-d-qa-curriculum
Architecture revision baseline: 3fb0e65
Current repository HEAD: re-check with Git when taking over the project
Tracking: origin/feat/g2.0-d-qa-curriculum
Latest implementation baseline at review time: f4a0089
Authoritative plan: docs/PROJECT_PLAN_v3.md
```

本轮从此基线重新审查，不使用旧 `main` / `2f8401c` 审查结论。远程机器状态和下列历史运行结果未在本轮重跑；当前源码核对与历史门控证据分开陈述。

| 已关闭门控 | 实施 commit | 当时记录的通过测试数 |
|---|---|---|
| v1 diagnostic + LoRA baseline | 历史结果见 §3 | CLOSED |
| G2.0-A Camera Geometry | ddf1594 | 几何兼容验收见 §8.1 |
| G2.0-B Multi-view Renderer | 8ced8de | 24 |
| G2.0-B.1 Camera Sampling | 6d08fcf | 62 |
| G2.0-C Spatial Truth | 19e3526 | 98 |
| G2.0-D QA Curriculum | 6823f29 | 141 |
| G2.0-E1 Data / Rendering | cf2fb52 | 169 |
| G2.0-E2 Training Harness | ada1179 | 188 |
| G2.0-E3.1 Evaluation / Protocol | c494e26, 6553d17 | 201 |
| G2.0-E3.2 Engineering Runner | af77949 | PASS / CLOSED；精确运行总数未从 Git 证据确认，见 §9 |
| G2.0-E3.3A GPU Optimization / Profiles | f4a0089（前序 5ad40fd） | 241 / 241 |

**当前训练事实**：`Qwen/Qwen2.5-VL-3B-Instruct`、BF16、LoRA r=8 / alpha=16 / dropout=0.05；252 LM 模块、0 vision LoRA、14,966,784 可训练参数，视觉塔冻结。不是 QLoRA。研发入口默认 `max_performance`：MB4 × ACC2、GC=False、Frozen Vision Cache=True；历史 E2 MB1 × ACC8 / GC=True 配置保留在 §9。

**当前科学问题未解**：A（south）、C（cardinal）、D（cardinal+jitter）多视角课程是否改善可迁移空间推理，而非仅提升 synthetic 分数？每组 1552 样本、78 个相同训练场景、4 家族各 388、每家族方向 194/194；S1/S2 各 1136，6928/6928 几何复核零差异。详细来源和冻结语义见 §8–§9。

**性能闭环 ≠ 科学矩阵闭环**：P7 Group A seed 42 已记录 1552/1552 样本、194/194 optimizer、194/194 scheduler；182.2 s（3.04 min）、8.519 samples/s、96.2% GPU 平均 / 100% p99、241.1 W、相对历史 E3.2 9.18x。测量边界及 ~32 GB AutoDL 环境限制见 §9，不是所有硬件的性能承诺。

**当前门控状态与近期执行路线（2026-09-08 经 God View 人类 UAT 修订，详见 §14 与 §27）**：

1. **God View / Research Inspector（G1 切片）**：**VALIDATED / HUMAN UAT PASSED（定向对抗性复审 TARGETED ADVERSARIAL REVIEW PENDING，暂不标记 CLOSED）**。已交付完整三维视口交互、双模式分离（Inspect Sample / Explore Scene）、搜索式样本选择器、抽屉式硬件 Profile / Preflight 管理与统一 Inspector 面板。
2. **G2.1 Dynamic Viewpoint & Scene Challenge Foundation（NEXT / COMMITTED）**：自适应相机画幅（Adaptive Camera Framing）、连续进动观测轨迹（Precessing Observation Trajectory）与受控场景复杂度分级（Controlled Scene Complexity V1）。
3. **G2.2 Render-derived Visibility / Occlusion Truth（PLANNED）**：基于渲染实据（全场景实例渲染与参考投影视网膜像素对比）推导权威可见度与遮挡真值层。
4. **G2.3 Temporal / Multi-frame Spatial QA（PLANNED）**：在连续观测轨迹上探索视点变换推理、视点条件时序演进追踪与主动观测前序任务。
5. **Controlled Formal Comparison + Factorized Ablation（DEFERRED / PLANNED）**：正式 A/C/D 多 seed 评测顺延至新动态基石就绪后，作为静态基准与动态轨迹课程开展受控对比与析因消融（Factorized Ablation）。

**边界**：World State ≠ Observation；中心关系 ≠ 像素可见度 / 遮挡真值；God View 特权信息不得进入模型输入；连续进动轨迹是受控观测课程而非 Agent 行动；禁止 silent fallback。

---

# 1. 项目身份

- **项目名**：SpatialForge（空间锻造台）
- **长期定位（PLANNED / ARCHITECTURAL）**：Spatial Intelligence Research Infrastructure；训练只是连接环境、观测、监督、模型、执行、验证和研究检视的一个子系统。
- **当前实现（CURRENT / VALIDATED）**：Blender 受控合成场景、相机几何、静态 QA 课程、Qwen2.5-VL 实验训练与评测管线。
- **代码仓库**：公开 GitHub 仓库，MIT License
- **当前阶段**：v2.0
- **研究对象**：Vision-Language Models / Multimodal Models 的 3D 空间理解、视角转换、主动观察、空间记忆、动作条件世界建模
- **长期目标**：让模型通过多视角、连续行动、可操作环境中的训练经验，形成比静态单图 VQA 更稳定的 3D world representation 与 spatial reasoning 能力

## 1.1 北极星

> 让 VLM 不只是“看一张图猜答案”，而是能够在一个持续存在的 3D 世界中理解“我在哪里、物体在哪里、换一个位置会看到什么、下一步应该去哪里看、动作会如何改变观测或世界状态”。

## 1.2 研究边界

SpatialForge 聚焦 **认知层世界模型**，不把底层机器人电机控制作为研究主体。

允许存在一个确定性的“脑干 / 执行器”层，负责：

- `move_forward(distance)`
- `turn(angle)`
- `look_up/down(angle)`
- `strafe(distance)`
- `open(container)`
- 其他离散或参数化环境动作

但不研究：

- 低层关节控制
- torque control
- motor policy
- locomotion controller 本身

---

# 2. 用户原始愿景（最高优先级）

以下内容是 SpatialForge 最重要的研究愿景。工程降级方案不能替代它，只能作为通往它的脚手架。

## 2.1 AI“附身”进入 3D 世界

最终 Agent 应具有一个受物理/运动约束的第一人称 Camera：

```text
3D Environment
      ↓
  Embodied Agent
      ↓
First-Person Camera
      ↓
 Observation_t
      ↓
 reasoning / memory / world model
      ↓
    Action_t
      ↓
 Environment transition
      ↓
 Observation_t+1
```

模型不能直接读取全局 Scene Graph、隐藏物体坐标、God View 或完整世界状态。

## 2.2 多角度观察与空间课程

同一世界可以从不同视点观察，用于：

- viewpoint transformation
- orientation
- left/right/front/back
- near/far
- occlusion
- depth
- object permanence
- multi-view consistency

## 2.3 “找东西训练法”

代表任务：

> “找玻璃杯。”

Agent 不一定一开始看得到目标，需要：

- 主动换角度
- 移动位置
- 接近目标区域
- 处理遮挡
- 必要时打开柜门/容器
- 利用历史观察
- 最终回答 / 定位目标

对应研究方向：

- Interactive Object Goal Navigation
- Active Visual Search
- Embodied Question Answering
- POMDP / belief-state reasoning

## 2.4 大脑—小脑—脑干分层认知

**RESEARCH OPTION / NOT DECIDED**：以下是研究假说与功能类比，不是固定软件模块划分或神经网络拓扑。SpatialForge 也应支持不包含三个独立神经模块的架构。需要区分的功能是 reasoning/planning、world-state prediction/spatial intuition、environment action execution。

一种可能的概念分解：

```text
┌─────────────────────────┐
│ Brain / System 2        │
│ VLM reasoning/planning  │
│ QA / logic / strategy   │
└───────────┬─────────────┘
            │ latent belief / goal
            ▼
┌─────────────────────────┐
│ Cerebellum / System 1   │
│ Spatial World Model     │
│ z_t + a_t -> z_t+1      │
│ spatial / physical      │
│ intuition               │
└───────────┬─────────────┘
            │ action
            ▼
┌─────────────────────────┐
│ Brainstem / Executor    │
│ deterministic actions   │
└───────────┬─────────────┘
            ▼
       3D Environment
```

核心边界：

> SpatialForge 研究“认知世界模型”，而不是底层机器人控制。

---

# 3. v1 已关闭：实验事实与科学发现

v1 门控已完成，不重做。它的价值是建立空间能力诊断基线并暴露结构性短板。

## 3.1 已验收结果

| 项 | 结果 |
|---|---|
| VSR overall | 3B 80.2 / 3B+LoRA 80.0 / 7B 83.7 |
| VSR orientation | 65.1 / 70.4，持续最弱 |
| synth holdout overall | 3B 前 82.6 / 后 82.9 / 7B 83.5 |
| count | 67.8 / 66.7 / 84.4，容量敏感 |
| near_far | 70.4 / 73.2 / 61.3，随规模出现倒挂 |
| LoRA | r8 / alpha16 / 2718 samples / 1 epoch / loss 0.055 |
| 32B-AWQ | 按预承诺规则放弃 |

## 3.2 v1 核心发现

### D1. 空间能力不是单一标量

Spatial reasoning 应被看成维度剖面，而不是一个 accuracy。

### D2. Orientation 是结构性短板

模型增大或简单 LoRA 并没有消除 orientation weakness。

这给 v2 的 viewpoint-conditioned training 提供了明确动机。

### D3. 尺寸—距离混淆假设

模型可能使用 apparent size 作为 distance shortcut，而不是建立稳定的 3D perspective / geometry representation。

near/far 倒挂随规模增强，是当前需要进一步证伪/验证的现象。

### D4. 静态单视角基元训练跨域迁移弱

v1 静态 synthetic primitives 并没有证明能形成可迁移空间世界模型。

结论：必须走向：

- 多视角
- 视角条件
- 连续观察
- 行动
- 主动观测
- 具身经验

---

# 4. v2 核心科学问题

v2 不再只问：

> “模型在一张图上答对了吗？”

而是逐步回答：

1. 同一个世界只改变 CameraPose，模型是否保持 3D 一致性？
2. 视点变化后，模型能否正确更新 left/right/front/back/orientation？
3. 模型能否区分真实大小、相机距离、camera depth、projected size？
4. 给定 Action，模型能否预测下一视角的 latent spatial state？
5. 看不清时，模型能否主动选择更有信息的下一观察位置？
6. 目标被遮挡时，模型能否利用记忆和交互找到目标？
7. 模型能否形成 persistent 3D belief，而不是每张图重新猜？

---

# 5. 唯一规范架构：从当前实验系统到研究基础设施

## 5.1 核心契约与状态标记

**World State ≠ Observation**：同一个世界可在不同相机条件下产生多个观测。世界真值供环境、监督与 verifier 使用；模型输入仅包含任务授权的观测与问题。监督答案可成为 SFT target，不能因 Inspector 拼接而成为推理 prompt 或隐藏状态输入。

本文状态标签：**CURRENT / VALIDATED** 为已实现或历史验收事实；**NEXT / COMMITTED** 为紧接着要做的门控；**PLANNED / ARCHITECTURAL** 为尚未实现的设计方向；**RESEARCH OPTION / NOT DECIDED** 为需要实验选择的候选。设计维度不是本轮接口实现承诺。

当前代码依赖关系（不是把真值串入模型输入的流程）：

```text
SceneState + SceneObject + CameraPose
  ├─ Blender rendering → image + ObservationMetadata
  └─ compute_spatial_truth → SpatialTruthRecord → QASample
image / QA / manifest → MultimodalSample → training or inference
predictions + evaluation targets → EvaluationPredictionRecord / metrics
training execution → progress.json / NVML telemetry
```

`Observation` 是观测概念；当前实际类为 `ObservationMetadata`，不要虚构已存在的统一 Observation API。当前训练使用单图 QA，多视角体现为课程样本来源；G2.0-D 的 paired QA 支持不等于 E3 已训练双图模型。

## 5.2 六个能力层与一个跨层观测平面

| 层 | CURRENT / VALIDATED | PLANNED / ARCHITECTURAL（增量扩展） |
|---|---|---|
| 1. Canonical World / Experience Contract | 不可变 `SceneState`、`SceneObject`、`CameraPose`；图像 + `ObservationMetadata` | 保留现有 SceneState 语义；增补 `TemporalState/state_t`、`Observation_t`、`Action/Action_t`、`Transition/StateTransition`、`Episode`；Agent-local memory/belief 与世界真值分离 |
| 2. Environment / Simulation | Blender controlled primitives、确定性相机采样与课程渲染 | 未来 `SimulationBackend` 按能力提供 observe / step / reset 等实验语义；把现有 Blender 接入此契约，physics-oriented backend、Isaac Lab / Isaac Sim、MuJoCo-family 则是尚未集成的候选 |
| 3. Task / Supervision / Verifier | `SpatialTruthRecord` 中心关系、`QASample`、单图多模态 SFT、严格 parser 与分维度指标 | classification、regression、structured prediction、latent prediction、action prediction、verifiable reward；可覆盖 depth/distance、pose、trajectory、scene graph、future latent、action、physics-verifier outcome |
| 4. Model Integration | Qwen2.5-VL-3B 专用 processor / chat template、LM 白名单、视觉特征拼接、generation | `ModelBackend / ModelIntegration` 声明 token/位置语义、视觉/LM 模块发现、可训练区域、缓存边界、生成及评估能力；避免以 ModelAdapter 混淆 LoRA adapter |
| 5. Learning Strategy | B zero-shot / inference-only 与 BF16 language-side LoRA | LoRA 扩展、QLoRA、DoRA、selective_ft、full_ft、continual_pretrain、world_model_pretrain、verifier-guided / RL post-training；不构成普适质量阶梯 |
| 6. Execution / Scaling | 单 GPU CUDA 实验路径，三个 performance profiles，缓存、prefetch、遥测 | DDP、FSDP2、可选 ZeRO-3、TP / PP / CP、distributed checkpoint 与可扩展数据执行；按能力组合，未实现 |
| 跨层 Research Inspector / Observability Plane | 已有数据/评测记录、训练进度与 GPU 遥测可供接入；无已交付 3D Inspector | 观察 world truth、observation、sample、model result、evaluation、telemetry，未来观察 episode/action trace 与 verifier feedback；只读检视与重放，不修改科学语义，不向 Agent 泄露特权数据 |

各层是职责边界，不要求今天拆成服务或插件。长期链路是 world/environment → observation → task/supervision → model integration → learning → compute execution → evaluation/verifier，并由跨层 Inspector 关联证据；closed-loop experience 要等 action/transition 契约成立。

## 5.3 训练配置是正交政策维度，不只是“两轴”

| 政策维度 | 当前验证点 | 未来可扩展选择 |
|---|---|---|
| Model Backend | Qwen2.5-VL-3B；概念标识 `qwen2_5_vl`，尚无 registry | 模型家族及其能力声明 |
| Learning Strategy | inference_only / language-side LoRA | QLoRA、DoRA、selective/full FT、CPT、world-model 或 verifier-guided objective 对应学习方法 |
| Execution Backend | single GPU CUDA | DDP / FSDP2 / optional ZeRO-3 / TP / PP / CP 合法组合 |
| Performance / Resource Policy | reference / balanced / max_performance，精确规格见 §9 | 各后端单独验证的 batch、GC、cache、offload、吞吐/显存预算 |
| Precision / Storage Policy | BF16 当前路径 | 支持时的 FP16、量化存储、FP8；存储精度与计算精度分别记录 |
| Task / Objective | directional QA 的 answer+EOS SFT、zero-shot evaluation | 回归、结构预测、latent transition、action prediction、verifier-guided objective |

“正交”指独立声明与追踪，不表示所有组合可运行。QLoRA 保留为未来低资源策略选项，需要相容量化存储；可训练视觉/embedding 区域与缓存策略相互约束；execution backend 影响实际有效 batch。未来 capability resolver 应校验组合并将解析后的全部政策和版本冻结到 run manifest；当前没有通用 resolver，也没有单一万能 training mode。

当前 `FormalTrainingConfig.from_profile()` 与工程 runner 默认 `max_performance`；直接构造 `FormalTrainingConfig()` 仍保留历史 MB1×ACC8 / GC=True / cache=False 默认值，即使其字段名为 `reference`。它不同于现行 `reference` Profile（GC=False / cache=True）。这是兼容性债务，不在本轮修改源码。

## 5.4 表征能力与缓存的有效边界

当前结构是 **frozen foundation model + language-side PEFT adaptation**：252 LM LoRA / 0 vision LoRA、冻结视觉塔。实验主要检验 LM 侧能否更好消费已有视觉表示，不能据此证明学到了根本不同的视觉空间表征；也不能断言 LoRA 不会学习空间推理。

未来 trainability policy 可覆盖 language-side PEFT、projector/merger、selected vision blocks、selected multimodal components、full multimodal FT、multimodal continual pretraining。这些是训练范围选项，不是固定解冻 top N 层的架构；由 ModelBackend 声明模块能力。

`pipeline.py` 的 Frozen Vision Feature Cache 以图像内容、模型 revision、schema、processor 类型与 dtype 等构建 key；`training.py` 还在 no_grad 下预制含文本 embedding 的 `inputs_embeds`。这些路径依赖其上游计算保持冻结且稳定。任何 merger/视觉/文本 embedding 解冻、随机预处理或位置语义改变，都必须重新验证缓存边界，禁用不再合法的缓存或只缓存冻结前缀，并扩充 provenance；不能把现有 P7 路径直接套用到 full FT。当前缓存 API 并不是通用 trainability/capability 验证器。

**Spatial Positional Representation Adaptation（RESEARCH OPTION）** 取代简单的“unfreeze RoPE”处方：候选包括多模态位置编码、camera pose / view embedding、relative geometric bias、temporal position、coordinate/geometric tokens、模型专属 multimodal RoPE。选择依模型家族与消融证据，不要求每个机制都有可解冻参数。当前 cached-input 的 token / positional 处理需在新增模型或任务时单独证明等价，不能从现有 QA 验收推导通用保证。

## 5.5 扩展的进入条件

模型、模拟器、学习策略与计算后端独立是长期目标，当前源码仍有 Qwen 类导入、固定 36 层 LM 路径、CUDA/device 0、AutoDL 默认路径、单图 QA 和本地全量准备等耦合。保留这些为明确结构债务，在相应实验需要第二种实现时提取接口；不为远期设想提前建设完整框架。

---

# 6. 三类 Camera 必须严格区分

这是 v3 计划的关键设计修订。

## 6.1 Diagnostic Camera

用途：

- calibration
- deterministic evaluation
- multi-view ground truth
- counterfactual viewpoint tests
- curriculum generation

特点：

- 可以分布在场景周围
- 可以从上/下/斜向观察
- 可以“瞬移”到 canonical viewpoint
- 不代表 Agent 的物理行为

## 6.2 Embodied Camera

**PLANNED / ARCHITECTURAL**：这是未来 Agent 的受限第一视角传感器，当前没有 action-driven embodied loop。

它依赖：

```text
AgentState
├── position
├── body_heading
├── head_yaw
├── head_pitch
└── camera_height
```

只能通过 Action 改变，而不是任意 teleport。

## 6.3 God View Camera

**NEXT / COMMITTED**：仅供人类研究者 / evaluator 使用的外部检视相机；显示世界几何不代表新增了权威关系语义。当前可用输入为 SceneState、已记录 CameraPose、观测与中心关系。frustum 必须先通过 §14 的投影契约；visibility/occlusion、trajectory、候选动作与 belief 为有相应数据契约后才开放的图层。God View 不进入 Agent 输入。

---

# 7. Camera Sampling System：从 4-view 到连续空间

G2.0-B 的 4 个 cardinal views 只是 smoke test，不是最终训练 camera set。

## 7.1 6-axis canonical views

方向：

```text
(+1, 0, 0) East
(-1, 0, 0) West
(0,+1, 0) North
(0,-1, 0) South
(0, 0,+1) Up
(0, 0,-1) Down
```

用途：最基础轴向诊断。

## 7.2 14-view subset

自然的 14-view 定义：

```text
6 axis
+
8 cube corners
=
14
```

8 个 corner directions：

```text
(±1, ±1, ±1)
```

它覆盖“东北上 / 东南上 / 西北下”等三轴斜视方向。

## 7.3 26-view canonical lattice

若进一步加入所有双轴斜视方向：

```text
6 axis
+
12 edge directions
+
8 corner directions
=
26 viewpoints
```

等价于：

> `3×3×3` 方向格点去掉中心 `(0,0,0)`。

12 个 edge directions 是两个坐标非零、一个坐标为零，例如：

```text
(+1,0,+1) East-Up
(+1,+1,0) East-North
(0,-1,-1) South-Down
...
```

## 7.4 Canonical views 与 Training views 分离

### Evaluation / diagnostics

使用确定性 canonical views：

- 6
- 14
- 26

优点：

- 可重复
- 可比较
- 易做 counterfactual analysis

### Training

以下是长期课程方向；当前冻结 A/C/D 的 fixed/cardinal/jitter 对照按 §8.6 与 §27 保留，不在原实验中追加相机策略。

不允许把 canonical camera 作为永久能力边界。

训练 CameraPose 应引入：

```text
position jitter
radius jitter
yaw / pitch perturbation
target jitter
continuous random sampling
```

目的：防止模型把 viewpoint 当离散类别背下来。

原则：

> Canonical cameras 是坐标骨架，不是训练空间本身。

---

# 8. 当前已完成的 v2 工程门控

## 8.1 G2.0-A — Camera Geometry Core ✅

Commit：

```text
ddf1594 feat(v2.0): add camera geometry core
```

新增：

```text
spatialforge/environment/
├── __init__.py
├── camera.py
└── geometry.py
```

核心能力：

- `CameraPose`
- `camera_basis()`
- `world_to_camera()`
- `metric_distance()`
- `camera_depth()`

坐标约定：

```text
World: Z-up
Camera:
+X = right
+Y = up
+Z = forward
```

验收：

- orthonormal basis
- v1 left/right compatibility
- opposite-view left/right flip
- metric distance orientation invariance
- camera depth orientation dependence
- invalid pose validation
- environment layer 无 `bpy`
- v1 front/back corpus compatibility

兼容性检查：

```text
100 scenes
600 object pairs
0 mismatch
```

## 8.2 G2.0-B — Multi-view Scene Renderer ✅

Commit：

```text
8ced8de feat(g2.0-b): add multiview scene rendering pipeline
```

新增：

```text
spatialforge/environment/
├── scene.py
├── observation.py
└── views.py

scripts/
└── render_multiview.py

tests/
├── test_scene_state.py
└── test_multiview_metadata.py
```

形成：

```text
SceneState
    ↓
4 deterministic CameraPose
    ↓
Blender rendering
    ↓
4 images
+
Observation manifest
```

最终 smoke render：

```text
outputs/synth/scene_000_view_south.png
outputs/synth/scene_000_view_east.png
outputs/synth/scene_000_view_north.png
outputs/synth/scene_000_view_west.png
outputs/synth/manifests/scene_000_manifest.json
```

验收：

```text
24 tests passed
Blender render exit 0
4/4 views generated
manifest valid
repo-relative forward-slash image paths
no scene_scene_000_* artifacts
```

G2.0-B 的定位：

> **Calibration / Diagnostic Observation Layer**，不是最终具身训练方式。

---

## 8.3 G2.0-B.1 — Camera Sampling System ✅

Commit：

```text
6d08fcf feat(g2.0-b.1): add camera sampling system
```

新增：

```text
spatialforge/environment/
└── sampling.py

tests/
└── test_camera_sampling.py
```

形成：

```text
Canonical directions (6-axis / 14-view / 26-view)
    ↓
deterministic CameraPose
    ↓
seeded bounded jitter
    ↓
continuous uniform sphere sampling
```

核心能力：

- canonical 6-axis / 14-view / 26-view direction sets，ID 与顺序完全确定
- camera-offset 方向约定：`position = target + radius * normalize(d)`，`look_at = target`
- 6-axis 顺序：east → west → north → south → up → down
- 14-view = 6 axes + 8 cube corners；26-view = 6 axes + 12 edge directions + 8 corners
- 垂直 / 近垂直视角的稳定 up-vector fallback（`(0,1,0)`），保证 `camera_basis()` 有效
- seeded bounded jitter（angular cone / radius / true radial target，全部本地 RNG）
- 不修改全局 RNG 状态
- 纯 CPU，无 `bpy` 依赖
- 未修改 / 删除既有 4-view API，G2.0-B 回归保持

验收：

```text
62 CPU unittest tests passed
G2.0-B 回归保留（VIEW_IDS 与 generate_cardinal_views 不变）
无既有 source 文件被 G2.0-B.1 修改
Blender 不参与本 gate 的 CPU 测试
```

G2.0-B.1 的定位：

> 把固定 4-view smoke camera 升级为可复用的 diagnostic / curriculum 相机采样系统；
> canonical 视角是可复现的坐标骨架，训练期仍应叠加 jitter / continuous sampling。

---

## 8.4 G2.0-C — View-Conditioned Spatial Truth Engine ✅

Commit：

```text
19e3526 feat(g2.0-c): add view-conditioned spatial truth engine
```

新增：

```text
spatialforge/environment/
└── relation.py

tests/
└── test_spatial_truth_engine.py
```

形成：

```text
SceneState + CameraPose
        ↓
compute_spatial_truth()
        ↓
SpatialTruthRecord
    ├── per-object camera-relative truth
    └── per-pair camera-relative relations
```

输入 `SceneState + CameraPose`，输出确定性 camera-relative spatial truth。
定位：

> Camera-relative 3D geometry truth layer。NOT image-projection / occlusion /
> visibility / QA layer。全部基于 `SceneObject.location` 中心语义，不推断 extent。

已实现语义：

- camera-frame left / right / aligned
- camera-frame above / below / aligned
- signed-depth front / behind / same_depth
- Euclidean camera-distance nearer / farther / equidistant
- metric object-to-object distance 作为 world-invariant truth 保留，不随 CameraPose 改变
- `SceneState.objects` tuple 顺序索引是权威 object identity，不假设 name 唯一，支持重复 name
- behind-camera 对象保留并标记 `in_front_of_camera`
- 确定性 pair ordering（index combinations）
- 确定性 JSON 兼容 `SpatialTruthRecord` 序列化（`to_dict()`）
- finite 且 >= 0 的 eps 校验（拒绝负值 / nan / inf）
- 无 image-plane projection、occlusion、bbox、visibility、QA 逻辑进入本 gate

Counterfactual 验证覆盖：

- opposite-view relation flips
- 纯相机旋转：camera-relative truth 改变，而 metric distance 不变
- camera relocation 可使 near/far flip
- canonical 6/14/26 views
- jittered cameras
- continuous sampled cameras
- behind-camera cases
- epsilon-band behavior
- duplicate-name identity
- deterministic serialization

验收：

```text
98 CPU unittest tests passed
62 pre-existing tests preserved
36 new G2.0-C tests
no existing source files modified
Blender 不参与本 gate 的 CPU 测试
```

---

## 8.5 G2.0-D — Multi-view QA Curriculum ✅

Branch：

```text
feat/g2.0-d-qa-curriculum
```

Implementation commit：

```text
6823f29 feat(g2.0-d): add multiview QA curriculum
```

新增：

```text
spatialforge/environment/
└── qa.py

tests/
└── test_qa_curriculum.py
```

形成：

```text
SpatialTruthRecord (G2.0-C)
        ↓
qa.py 课程层
        ├── CurriculumView / QASample (frozen dataclass)
        ├── generate_single_view_qa()
        ├── generate_paired_view_qa()
        └── generate_multiview_curriculum()
```

定位：

> 把 G2.0-C 的确定性 spatial truth 转成确定性自然语言训练课程。纯 CPU
> deterministic data-generation 层。NOT rendering / model training / LLM /
> projection / FOV / occlusion / visibility / bbox / embodied Agent 层。

已实现语义：

- **Truth consumed, never recomputed**：QA 层只读取 G2.0-C 字段；唯一的几何调用是
  `generate_multiview_curriculum()` 内的 `compute_spatial_truth()`。不做任何
  geometry 重复实现。
- **single-view QA**：固定 family 顺序 horizontal → vertical → depth → near_far；
  问题模板显式含 "From this view"，答案 left/right、above/below、front/behind、
  nearer/farther。
- **paired-view transformation QA**：同 SceneState 双 CameraPose 的 relation
  变化，如 `left -> right` / `front -> behind` / `nearer -> farther`，显式教学
  viewpoint transformation。
- **front-halfspace eligibility policy**：仅当 pair 两个对象在所有相关 view 的
  `in_front_of_camera == True` 才生成 normal visual QA。文档明确
  `eligible_for_visual_qa != guaranteed_visible`；本 gate 不做 FOV/occlusion/
  visibility 声明。
- **neutral filtering policy**：aligned / same_depth / equidistant 默认排除；
  `include_neutral=True` 时以其 G2.0-C 精确标签发射，绝不强制二值化。
- **world-invariant metric-distance controls**：每 eligible pair 附带一条
  "Does the metric distance between ... change ..." → `unchanged`，tag
  `world_invariant`，`is_view_dependent=False`；metric_distance 不一致则抛
  `ValueError`（broken world invariant）。
- **canonical / jitter / hard-angle tags**：全部由调用方 `CurriculumView.tags`
  提供并保留，不根据浮点坐标猜测 provenance。
- **symmetry_flip**：精确逆变换（left↔right / above↔below / front↔behind /
  nearer↔farther）自动附加 `symmetry_flip`。
- **size_distance_conflict**：当物理 size 顺序与 camera-distance 顺序冲突
  （large-far / small-near）时附加 `size_distance_conflict`，只用
  ObjectTruth.size 元数据。
- **identity**：对象引用一律 index + name（`object 0 ("red_cube")`），name 不
  唯一也安全；sample id / ordering 只依赖 index / view id / family，永不依赖
  name，不用 Python `hash()`。
- **deterministic IDs / order / serialization**：样本 id 形如
  `scene:single:view:a-b:family`、
  `scene:paired:a_view->b_view:a-b:family`、
  `scene:paired:a_view->b_view:a-b:metric_invariance`；`to_dict()` JSON 兼容且稳定。
- multiview 输出顺序：input view 顺序的 single-view 块 → 确定性 view 组合
  (i<j) 的 paired 块（每个 block 内 per-pair family 变换样本 + 该 pair 的
  invariant control）。
- 无文件写入核心 API；序列化仅通过 `to_dict()`，JSONL 写入留给未来
  dataset/export 层。
- `_validate_same_scene` 拒绝：scene_id 不一致、object 数量不一致、按
  object_index 的 stable world facts 不一致；multiview 拒绝重复 view_id 与
  空 view_id。

Counterfactual / curriculum 验证覆盖：

- opposite-view（south vs north / west vs east）relation flips → `left -> right`
- rotation-only control：viewpoint-dependent QA 改变，metric ordering 不变
- relocation counterfactual：`nearer -> farther`
- world-invariant control：object-to-object metric distance 恒为 unchanged，
  不兼容 truth 被拒绝
- canonical 6/14/26、seeded jitter、continuous sampled (hard-angle) views
- symmetry traps、size-distance conflict、duplicate names、determinism、
  JSON compatibility、精确 ordering lock

验收：

```text
141 CPU unittest tests passed
98 pre-existing tests preserved
43 new G2.0-D tests
no existing source files modified
Blender 不参与本 gate 的 CPU 测试
```

---

## 8.6 G2.0-E1 — Controlled Multiview Experiment Data & Rendering Foundation ✅

Branch：

```text
feat/g2.0-d-qa-curriculum
```

Implementation commit：

```text
cf2fb52 feat(g2.0-e1): add controlled multiview experiment data foundation
```

门控结论：

```text
PASS / CLOSED
Final independent review verdict: APPROVE E1 COMMIT
Test suite: 169 / 169 PASS (0 regressions)
Closed gates G2.0-A / B / B.1 / C / D files remain completely unchanged.
```

新增文件：

```text
scripts/
├── export_v1_scenes.py
└── render_curriculum.py

spatialforge/
└── experiment/
    ├── __init__.py
    └── dataset.py

tests/
└── test_experiment_dataset.py
```

### 交付成果与科学事实记录

#### 1. 历史 v1 场景生成器精确复原（Historical v1 Scene Recovery）
- 历史 v1 程序化场景生成逻辑被完整复原并实现为纯 Python 脚本：`scripts/export_v1_scenes.py`。
- 可严格重构全部 100 个历史场景（`scene_000` ... `scene_099`）。
- 生成结果经严格比对完全吻合：
  - `examples/scenes/scene_000.json`（语义与浮点坐标严格一致）
  - `tests/fixtures/v1_front_back_reference.json`（100 个场景全部物体坐标对齐至数值精度）
- 场景划分保持历史兼容：
  - `train`: `scene_000` ... `scene_079`（80 场景）
  - `holdout`: `scene_080` ... `scene_099`（20 场景）

#### 2. 独立 G2.0-E 课程渲染器（Dedicated Curriculum Renderer）
- 新增专用渲染入口：`scripts/render_curriculum.py`。
- 已关闭的 G2.0-B 渲染器 `scripts/render_multiview.py` 保持完全不变（作为回归基准）。
- G2.0-E 渲染图像存放在独立隔离命名空间：`outputs/experiments/g2.0-e/rendered/`。
- 渲染器特性：
  - Blender 4.2.0（安装于 `/root/autodl-tmp/tools/blender/blender`）
  - 渲染引擎：`BLENDER_WORKBENCH`
  - 着色模式：`OBJECT` 颜色着色
  - 分辨率：512 × 512
  - 与 G2.0-B 保持完全一致的基础几何体与地面网格视觉域，不引入新 Eevee/Cycles/材质光照域混淆。
- 全量数据集渲染完成：**100 场景 × 8 PRIMARY_D 视角 = 800 张图像**。
- `PRIMARY_D` 视角组合：
  `south`, `east`, `north`, `west`, `south_jitter`, `east_jitter`, `north_jitter`, `west_jitter`。

#### 3. 冻结的主实验组设计（Primary Experiment Groups）
- **Group B**：Pretrained Baseline（Qwen2.5-VL-3B-Instruct 无微调基线）。
- **Group A**：固定单视点控制组（仅 `south` 视角，单图 Single-view QA）。
- **Group C**：四个基准方位视角处理组（four cardinal viewpoints: `south`, `east`, `north`, `west`，单图 Single-view QA，透视投影）。
- **Group D**：四个基准方位视角 + 有界扰动处理组（4 个基准方位视角 + 4 个确定性扰动视角，单图 Single-view QA，透视投影）。
- **实验边界**：主实验 A/C/D **仅使用单视图 QA（Single-view QA）**。配对双图变换 QA（Paired multi-image transformation QA）属于后续次级消融，明确不进入主实验 G2.0-E 训练。

#### 4. 相机策略约束（Camera Policy）
- 首次视觉训练实验明确仅使用**水平 Cardinal 视角及扰动**，绝不直接使用全部 6 个规范轴。
- **物理原因**：规范 "down" 相机位于地面以下，且当前 G2.0-D 的前半空间适格性（front-halfspace eligibility）并不等同于 FOV 内投影或像素无遮挡。
- 规范 6/14/26 相机框架保留作为诊断基准。

#### 5. 视觉指称表达策略（Visual Reference Policy）
- 内部权威对象标识严格沿用 G2.0-D 的 `object_index`。
- 内部底层变量名（如 `obj0`, `obj1`）绝不暴露给 VLM。
- 实验层面向模型的自然语言指称统一采用 `"the {color} {shape}"`。
- 歧义过滤：若场景内存在相同颜色+形状的重复描述对象，相关配对样本严格排除。在不渲染人工文本标签的前提下保障视觉 Grounding 的无歧义性。

#### 6. 适格训练场景（78-Scene Eligible Universe）
- 历史 80 个训练场景中，在视觉指称无歧义策略下，**正好 78 个场景可用**。
- **排除场景**：
  - `scene_007`：含 3 个 red cylinder、1 个 red sphere；任何物体对均含有非唯一的 red cylinder，可用无歧义配对数为 0。
  - `scene_056`：含 3 个 cyan sphere、1 个 blue cylinder；任何物体对均含有非唯一的 cyan sphere，可用无歧义配对数为 0。
- 其余 78 个训练场景可用无歧义配对数在 2 至 12 对之间。
- **结论**：适格训练场景宇宙严格为 78 场景；A/C/D 三组全部使用且仅使用这 78 个场景。这是歧义策略的确定性必然结果，绝非意外场景丢失。

#### 7. 最终统一训练预算（Common Training Budget）
- 冻结的 A/C/D 统一训练预算：**$N = 1552$**（各组完全相等）。
- 关系家族分配完全均衡（各 388 条）：
  - `horizontal` = 388
  - `vertical` = 388
  - `depth` = 388
  - `near_far` = 388

#### 8. 操作数顺序归一化（Operand-Order Normalization）
- **科学阻断点排查（F-02）**：初始评审发现 Group A 因固定 south 相机、固定插槽几何及 $i < j$ 配对顺序，存在严重的先验方向答案偏置。
- **修复层级**：严格在实验数据呈现层（Presentation Layer）解决，**绝不修改 G2.0-D 内部真值与 QA 定义**。
- **严格数学双射反转表**：
  - `left` $\leftrightarrow$ `right`
  - `above` $\leftrightarrow$ `below`
  - `front` $\leftrightarrow$ `behind`
  - `nearer` $\leftrightarrow$ `farther`
- **核心不变量**：
  - `QASample` 与 `spatialforge/environment/qa.py` 保持完全不变。
  - `source_sample_id` 完整保留 G2.0-D 血统。
  - `presentation_order` 显式记录（`0` 规范，`1` 反转）。
  - 每个数据集内任意 `source_sample_id` 最多出现一次（严禁通过同时包含规范与反转样本虚增 $N$）。
  - 反转决策纯确定性；跨组共享的同一底层样本在不同组中呈现决策 100% 一致（0 冲突）。

#### 9. 最终标签平衡（Final Label Balance）
- A/C/D 三组全部 4 个关系家族达到**绝对 50.0% / 50.0% 完美对称平衡**（各 194 / 194）：
  - `horizontal`: `left` 194, `right` 194
  - `vertical`: `above` 194, `below` 194
  - `depth`: `front` 194, `behind` 194
  - `near_far`: `nearer` 194, `farther` 194
- 消除了已发现的全局方向标签不平衡；不据此排除所有条件性捷径。

#### 10. 场景感知分层采样（Scene-Aware Stratified Sampling）
- **科学阻断点排查（F-01）**：初始 E1 采样器因贪心截断，导致 A 覆盖 78 场景，C 仅 18 场景，D 仅 9 场景，造成视点多样性与场景多样性的严重因果混淆。
- **修复**：重构为分视点、分家族跨场景确定性轮询分层采样算法。
- **最终状态**：
  - A/C/D 覆盖场景数完全一致：**均为 78 场景**。
  - 平均场景暴露度：各组均为 **19.8974 样本 / 场景**。
  - 场景暴露向量余弦相似度极高（A vs C: 0.974, A vs D: 0.970, C vs D: 0.980），彻底清除了场景截断混淆。

#### 11. 语义分布一致性控制（Semantic Distribution Control）
- 独立复核确认 A/C/D 语义构成高度一致，无实质性语义偏差：
  - 形状比例最大差异 < 1.6%（cube ~34%, cylinder ~33%, sphere ~33%）
  - 颜色比例最大差异 < 1.2%
  - 物体配对共现频率最大差异 < 1.6%
  - 物体尺寸均值基本完全一致（A: 0.7848, C: 0.7897, D: 0.7871）
  - 困难负例 `size_distance_conflict` 比例在 30.7% ~ 33.8% 之间稳定分布
  - 原门控结论为上述已审计分布未发现实质偏差；这些数字不能证明排除了所有潜在因果混淆。

#### 12. 冻结的评测 Holdout 集（Frozen Holdout Sets）
- **Holdout S1 (`holdout_s1_cardinal.jsonl`, $N = 1136$)**：未见场景（`scene_080`–`scene_099`）+ 四个基准方位视角（four cardinal viewpoints，透视投影）。用于测试熟悉视角家族下的未见场景泛化。
- **Holdout S2 (`holdout_s2_jitter.jsonl`, $N = 1136$)**：未见场景（`scene_080`–`scene_099`）+ 全新实例化有界扰动视点。用于测试同一扰动体系下的视点扰动泛化。
- **重要边界**：S2 不代表真正的连续球面上强 OOD 视点泛化基准。未来 E2/E3 评估可补充连续球面上任意视角的独立评估。

#### 13. 已知非阻断限制（Known Non-Blocking Limitations）
- **A. 相机 FOV 元数据**：`CameraPose.fov_deg` 标称为 60°，而 Blender 历史 parity 采用 `lens = 35mm`（对应默认传感器水平 FOV 约 54.4°）。当前 center-based 3D 几何真值不受影响，但在后续涉及像素反投影/精细可见性阶段需统一两处元数据。
- **B. 图像格式**：当前渲染输出为 RGBA PNG，E2 数据加载器在送入 Qwen2.5-VL 预处理器前需显式 `Image.open(...).convert("RGB")`。
- **C. 可见性语义**：前半空间适格性（`in_front_of_camera`）不等于保证在视野内或无遮挡。
- **D. S2 范围**：S2 属于有界扰动验证，非连续自由度 OOD。

#### 14. E2 依托的复原模型与训练基线（Model / Training Baseline for E2）
- **主选模型**：`Qwen/Qwen2.5-VL-3B-Instruct`（备选 7B 参照：`Qwen/Qwen2.5-VL-7B-Instruct`）。
- **历史 v1 LoRA 规范**：$r = 8$, $\alpha = 16$, $\text{dropout} = 0.05$；目标模块为全部 7 个线性投影层：`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`。
- **训练超参数**：`bf16`, per-device batch size = 1, gradient accumulation = 8, 学习率 $1\times 10^{-4}$，1 epoch，启用 gradient checkpointing，prompt token 屏蔽，仅监督 answer + EOS。
- **样本数说明**：历史 v1 为 2718 条，G2.0-E 主训练集采用 1552 条（因果实验科学严谨性优先于单纯数字复刻）。
- **纪律**：**不额外引入未经测试的 cosine scheduler 或 warmup**，严格保留历史 v1 优化语义以保障对照实验纯洁性。

#### 15. 外部评估指标定义（External Evaluation Semantics）
- 严格区分 VSR 外部迁移指标体系：
  - **直接迁移（Direct Transfer）**：与 G2.0-D 监督信号直接对齐的子集（`VSR left_right` 和 `VSR front_back`）。
  - **历史宽泛 Orientation**：涵盖更广泛的真实空间关系（`facing`, `facing away from`, `toward`, `opposite to`, `parallel to`, `perpendicular to`, `across from`, `across`, `along` 等）。
  - 后续 E2/E3 报告必须区分这两类指标，不得将 left/right/front/back 简单混为宽泛 orientation。

#### 16. 最终门控验收（Final Verification）
- 单元测试状态：**169 / 169 PASS**（141 pre-existing + 28 new G2.0-E1 tests；0 regressions）。
- **独立几何复核**：6928 / 6928 records 与原始 3D camera truth 一致，0 mismatches。
- **构建确定性**：重复生成 bit-for-bit 一致，SHA-256 相同；数据集内 source_sample_id 零重复，跨组共享源样本 presentation_order 零分歧。
- 提交 Commit：`cf2fb52`。
- 终审结果：**APPROVE E1 COMMIT**，正式关闭 G2.0-E1。

---

# 9. 已关闭训练 / 评估 / 性能门控证据（历史档案）

G2.0-B.1 / C / D / E1 的验收与事实统一见 §8，后续路线统一见 §27。以下 E2 配置是历史记录；当前执行规格以 E3.3A profiles 为准。

## G2.0-E2 — Controlled Training Harness & Pipeline Verification ✅ PASS / CLOSED

> 已完成并通过门控（commit ada1179），12 步分步验证全通，188/188 单元测试 PASS。

### E2 冻结事实与关键结论（E2 Frozen Facts）
- **基模与精度**：`Qwen/Qwen2.5-VL-3B-Instruct`，`bf16`。AutoDL RTX 4080 SUPER 显存适配安全无溢出。
- **历史 LoRA 规范严格保留**：$r=8, \alpha=16, \text{dropout}=0.05$。
- **严格语言模型专属 LoRA（Strict Language-Only LoRA）**：
  - 显式 LM 白名单：252 个语言模型投影模块（36 层 × 7 projections: `q/k/v/o/gate/up/down_proj`）。
  - 视觉塔（Vision Tower）0 个 LoRA 模块，视觉编码器保持 100% 冻结。
  - 可训练参数量：**14,966,784**（占总参数量 0.3970%）。
- **正式训练语义与梯度累积（Formal Training Semantics）**：
  - `per_device_train_batch_size = 1`
  - `gradient_accumulation_steps = 8`
  - `learning_rate = 1e-4`
  - `num_train_epochs = 1`
  - `bf16 = True`
  - `gradient_checkpointing = True`
  - `scheduler = linear`
  - `warmup_steps = 0`
  - 无 QLoRA / bitsandbytes。
- **训练规模与累积验证**：
  - 冻结 E1 单组样本规模 $N = 1552$。
  - 1552 microbatches / 8 = 194 optimizer steps / epoch，无累积余数。
  - 精确累积语义验证：microbatch 1~7 仅累积梯度，优化器步数为 0，LoRA 参数值不改变；microbatch 8 触发 1 次 `optimizer.step()` 与 1 次 `scheduler.step()`，参数完成更新。
- **多模态张量构建与损失屏蔽**：
  - RGBA 输入显式转换为 RGB。
  - Prompt 标记（362 tokens）严格屏蔽为 `-100`；答案与 EOS（`<|im_end|>`, 2 tokens）受监督。
  - `attention_mask` 全长为 1，完全覆盖答案 token，无截断 bug。
- **Adapter 保存与重新加载**：
  - 纯语言模型 LoRA adapter 保存大小约 59.95 MB（~60 MB）。
  - 重新加载至干净基模验证通过，加载后仍保持 0 视觉参数、504 个 LM 权重张量，推理正常。
- **资源与吞吐**：
  - 烟测峰值显存 ~13.5 GB（RTX 4080 SUPER 31.5 GB 显存裕量 >18 GB）。
  - 8-microbatch 累积烟测单步耗时 ~0.61s，吞吐 ~1.63 样本/秒。
- **代码与测试基线**：
  - 新增/修改代码：`scripts/train_smoke.py`, `spatialforge/experiment/training.py`, `tests/test_training_harness.py`。
  - 测试套件：188 / 188 PASS（181 pre-existing + 7 F1/F2 regression tests）。
  - 实施 Commit：`ada1179`。

### 关键 Blocker 复盘与修复记录（Recovered Blocker & Fix）
1. **F1 视觉塔 LoRA 误绑定**：初版使用未限定层级的后缀匹配 `target_modules = ["q_proj", ...]`，意外命中了 `model.visual.blocks.*.mlp` 中的 96 个视觉模块（3,609,600 参数）。在审查中被拦截，改用显式语言模型目标白名单（`get_language_model_target_modules()`），最终状态严格对齐为 252 LM / 0 视觉。
2. **F2 梯度累积语义缺失**：初版烟测直接单步 backward 即 step，缺乏 formal 累积语义。在审查中被拦截，重构为微批次梯度缩放（`loss / 8`）、每 8 步执行 1 次 optimizer+scheduler step 的可复用训练循环，并补充启用梯度检查点与线性调度器。

### E2 后续事项处置记录（历史状态）
- 正式评估前加固 directional / yes-no 解析器（已在 E3.1 完成）。
- 消除训练循环中的冗余计算与瓶颈（已在 E3.3A 完成，达成 9.18x 加速）。
- 明确认知：S2 为同扰动过程泛化，而非连续球面上强连续 OOD。
- 相机 FOV 元数据与 Blender 35 mm lens 的差异，现提升为 God View 0 前置契约（§14）；约 54.4° 依赖传感器假设。

---

## G2.0-E3.1 — Evaluation Hardening & Experiment Protocol Freeze ✅ PASS / CLOSED

- **交付成果**：
  - 严格词边界定向/判定答案解析器（`spatialforge/experiment/evaluation.py`），消除子串误判。
  - 不可变协议常量、SHA-256 数据集哈希校验与前置检查器（`spatialforge/experiment/protocol.py`）。
  - 10 项严格前置检查防线（数据集完整性、基模版本锁定、场景划分隔离、LoRA 目标白名单、确定性配置冻结）。
- **实施 Commit**：`c494e26`, `6553d17`。
- **测试状态**：201 / 201 PASS（188 pre-existing + 13 evaluation/protocol tests）。

---

## G2.0-E3.2 — Controlled Engineering Experiment Runner ✅ PASS / CLOSED

- **交付成果**：
  - 端到端受控工程实验执行脚本：`scripts/run_engineering_experiment.py`。
  - 全流程 4 条件（Baseline B 零样本、Control A42、Treatment C42、Treatment D42）配置预先冻结与 SHA-256 哈希固化。
  - Baseline B 零样本评测与训练-评估流水线验证。
  - 确定性对齐与因果增量计算（`A_minus_B`, `C_minus_B`, `D_minus_B`, `C_minus_A`, `D_minus_A`, `D_minus_C`）。
- **实施 Commit**：`af77949`。
- **历史测试证据**：`af77949` 新增 `tests/test_engineering_runner.py`，其中静态可核对 3 个 `test_*` 方法；后续 `5ad40fd` 才新增 `tests/test_training_throughput.py`。不能把后续测试增长归入 E3.2。已检查的 Git 提交信息与当时计划未提供该门控独立运行总数，因此删除原“227 / 227、201 + 26”的错误归属，不以静态方法数推算 runtime PASS 总数；门控与提交记录保留。

---

## G2.0-E3.3A — GPU Training Optimization & Performance Profiles ✅ PASS / CLOSED

> 已完成并通过终验，单条件完整训练耗时由 **27.88 分钟压缩至 3.04 分钟（9.18x 实测加速）**，平均 GPU 利用率由 **~24.9% 提升至 96.2%**。241 / 241 单元测试全通。

### 1. 最终验证的 8 项核心工程成果（Final Implementation Artifacts）
1. **Frozen Vision Feature Cache**：针对不可变图像预先提取并缓存视觉嵌入，彻底消除重复的冻结视觉编码器前向耗时。
2. **禁用梯度检查点（GC Disabled for Performance Profiles）**：在显存裕量充分的单卡环境中关闭梯度检查点，消除重算开销。
3. **数学受控的 Cached-Vision 批处理（Mathematically Controlled Batching）**：在 post-vision 特征阶段执行微批次右填充（mask=0, label=-100），实现完全一致的等样本加权损失（per-example loss），在数学上精确对齐单样本参考损失。
4. **显式训练执行 Profile（Explicit Training Performance Profiles）**：产品化规范三级第一类 Profile（`reference`, `balanced`, `max_performance`）。
5. **实时 NVML / PyTorch 遥测系统（Real-time Telemetry Sampler）**：后台独立线程按 0.2s 采样 GPU 利用率、功耗、显存使用及 SM 时钟频率（avg, min, max, p99）。
6. **样本基准进度核算（Sample-based Progress Accounting）**：进度条、ETA 与心跳严格基于物理样本数（1552）而非微批次数（388）。
7. **Profile 算力前置校验（Capability Preflight Check）**：在训练启动前严格比对当前硬件 VRAM 与 Profile 需求。
8. **严禁静默降级策略（No-Silent-Fallback Policy）**：硬件不足时强制抛出 `ProfileCapabilityError` 阻断执行，提供适配建议，杜绝破坏实验一致性的静默退化。

- **实施 Commit**：`f4a0089`（基准 commit：`5ad40fd`）；241 = 227 pre-existing + 14 profiling/telemetry tests。

### 2. 性能演进与全量浸泡压测事实（Exact Performance Progression & P7 Full Soak）

| 执行阶段 / Profile | 微批次 (MB) | 累积步数 (ACC) | 有效批次 | 视觉重算 | 梯度检查点 | 1552 样本耗时 | 吞吐率 (sps) | 相对基准加速比 | 平均 GPU 利用率 | 峰值 Alloc | 峰值 Reserved | 状态 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **历史 E3.2 参考基准** | 1 | 8 | 8 | 是 (每次重算) | 开启 | 27.88 min (1673s) | 0.928 | 1.00x | ~24.9% | 8.31 GB | 8.31 GB | 历史对比基线 |
| **Validated P5 / Reference** | 1 | 8 | 8 | 否 (特征缓存) | 关闭 | ~8.01 min (480s) | 3.230 | 3.48x | ~39.6% | 9.78 GB | 10.05 GB | 复现/调试保留 |
| **Validated P6 / Balanced** | 2 | 4 | 8 | 否 (特征缓存) | 关闭 | ~4.75 min (285s) | 5.440 | 5.86x | ~56.0% | 14.35 GB | 15.20 GB | 均衡安全规格 |
| **Validated P7 / Max Performance** | **4** | **2** | **8** | **否 (特征缓存)** | **关闭** | **3.04 min (182.2s)** | **8.519** | **9.18x** | **96.2%** | **20.96 GB** | **22.58 GB** | **全量浸泡验收** |

> **注**：加速比 9.18x 为全量 1552 样本训练阶段的历史实测对比，严禁四舍五入为模糊的“~9x”。

#### P7 全量 1552 样本单 Epoch 浸泡测试数据（Group A, Seed 42）
- **样本处理数**：1552 / 1552
- **优化器步数**：194 / 194
- **调度器步数**：194 / 194
- **全量训练循环耗时**：182.2 s = **3.04 min**（优于 $\le 3.8\text{ min}$ 目标，达成 $\le 3.4\text{ min}$ 优秀线）
- **实测吞吐率**：**8.519 samples/s**
- **GPU 平均利用率**：**96.2%**（峰值 p99: **100.0%**）
- **平均板卡功耗**：**241.1 W**
- **平均 SM 时钟**：**~2767 MHz**
- **峰值 Torch Allocated 显存**：**20.96 GB**
- **峰值 Torch Reserved 显存**：**22.58 GB**
- **训练收敛轨迹**：初始 loss 4.2417 $\to$ 最终 loss 0.0269（前 10 步均值 2.1902 $\to$ 后 10 步均值 0.0408，无 NaN/Inf）
- **单元测试状态**：**241 / 241 PASS**。

测量边界核对：`benchmark_training_throughput.py::run_p7_full_soak` 的 timer 位于模型加载与特征准备之后，adapter 保存之前。因此保留 182.2 s / 9.18x 原始记录，但不将它扩张为包含模型加载、冷缓存构建、保存及评估的完整任务 wall time。

### 3. 显存稳定性与规范术语说明（Memory Stability & Terminology）
- **规范用语**：严禁将 `reserved - allocated` 描述为确定性的 CUDA 显存“碎片（fragmentation）”。其准确术语为 **PyTorch 缓存分配器保留容量 / 常驻复用内存池（PyTorch caching-allocator reserved capacity / retained allocation pool）**。
- **7 点显存检查点实测证明**：
  - `model_loaded`: Alloc 7,218.5 MB | Res 7,294.0 MB | Host RSS 960.1 MB
  - `cache_ready`: Alloc 7,219.8 MB | Res 7,294.0 MB | Host RSS 3,530.8 MB
  - `first_optimizer_update` (Step 1): Alloc 7,834.3 MB | Res 23,084.0 MB | Host RSS 4,248.8 MB
  - `progress_25_pct` (Step 48.5): Alloc 7,891.4 MB | Res 23,120.0 MB | Host RSS 4,317.1 MB
  - `progress_50_pct` (Step 97): Alloc 7,834.3 MB | Res 23,120.0 MB | Host RSS 4,317.2 MB
  - `progress_75_pct` (Step 145.5): Alloc 7,891.4 MB | Res 23,120.0 MB | Host RSS 4,317.4 MB
  - `progress_100_pct` (Step 194): Alloc 7,834.3 MB | Res 23,120.0 MB | Host RSS 4,317.5 MB
- **稳定性结论**：
  - 显存分配进入稳定平台期（Plateau）：25% 到 100% 阶段 Allocated 仅在 7,834 MB 与 7,891 MB 间随序列长度 padding 波动（漂移仅 57.09 MB）；Reserved 容量**严格锁定为 23,120.0 MB（漂移 0.0 MB）**。
  - 宿主机 Host RSS 在训练循环中漂移仅 0.39 MB（4317.13 $\to$ 4317.52 MB）。
  - **无单调显存泄漏（No monotonic GPU leak），无宿主机泄漏（No host-memory leak），无运行期 OOM**。

### 4. 三级第一类执行 Profile 规范（Frozen Training Profiles）
1. **REFERENCE (`reference`)**：
   - 执行参数：MB1 $\times$ ACC8，有效批次 8，GC=False，VC=True。
   - 估算显存需求：~19,900 MB。
   - 定位：严格对照复现、代码调试、数值回归验证、本 profile 家族中较低显存预算（仍估算需 ~19,900 MB）。
2. **BALANCED (`balanced`)**：
   - 执行参数：MB2 $\times$ ACC4，有效批次 8，GC=False，VC=True。
   - 估算显存需求：~23,550 MB。
   - 定位：吞吐与显存裕量折中，适用于显存较紧张或共享计算环境（~9.2 GB 裕量）。
3. **MAX PERFORMANCE (`max_performance`)**：
   - 执行参数：MB4 $\times$ ACC2，有效批次 8，GC=False，VC=True。
   - 估算显存需求：~30,900 MB。
   - 定位：最大化释放 GPU 算力，达成极速迭代（8.519 sps, 96.2% util）。已在当前 32 GB AutoDL 环境完全验证。
   - **开发默认项（Development Default）**：`max_performance` 为 SpatialForge 显式指定的研发默认 Profile。

### 5. Profile 语义学边界（Profile Semantics & Scientific Boundaries）
- 三个 Profile 均为 **语义等价执行 Profile（Semantically Equivalent Execution Profiles）**：拥有完全一致的数据集切分、模型架构、LoRA 结构、学习率 ($1\times 10^{-4}$)、调度器、warmup、训练轮数、优化器窗口语义（有效批次恒等于 8）、等样本权重目标及视觉缓存语义。
- **不宣称严格 Bitwise 等价**：不同微批次结构会改变 CUDA reduction 浮点加法顺序及 Dropout RNG 随机数流的消耗顺序，可能在参数轨迹微观数值上产生极小差异。
- **因果对照纪律**：正式 A/C/D 科学对照实验必须在全矩阵范围内**统一固定使用单个执行 Profile**（推荐 `max_performance`）。

### 6. 严禁静默降级策略（No-Silent-Fallback Policy）
- 当用户显式指定 Profile（例如 `--profile max_performance`），若当前检查报告的显存容量低于其估算预算：
  1. 立即中断 preflight；
  2. 报告请求的 Profile 及其估算显存需求；
  3. 报告当前检查获得的显存容量；
  4. 给出向下兼容的推荐 Profile（例如 `balanced` 或 `reference`）；
  5. 严禁自动/静默降级 Profile，必须将选择权保留给用户。

当前实现默认查询的是 NVML / PyTorch **总显存容量**，不是实时空闲显存；无 CUDA 时此 helper 可直接返回，后续 runner 仍要求 CUDA。它实现了容量不匹配时拒绝静默降档，不是跨设备、并发占用或全部策略组合的安全证明。未来扩展需要更完整 capability checks。

### 7. 科学与工程结论（Engineering & Scientific Conclusion）
- 原先 ~24.9% 的低 GPU 利用率**绝非由于硬件算力不足**（单机独立大算力 BF16 GEMM 压测轻松达到 100% 利用率与 282 W 满功耗）。
- 真正的工程瓶颈在于**执行形态不匹配（MB=1 带来的并行度不足）**与**重复前向冻结视觉编码器**造成的管道空转。
- 将有用批次扩大至 MB4 并剥离重复视觉编码，成功将闲置 GPU 算力转化为真实吞吐：
  - 平均利用率：**~24.9% $\to$ 96.2%**
  - 单组训练耗时：**27.88 min $\to$ 3.04 min**
  - 样本吞吐率：**0.928 $\to$ 8.519 samples/s**
- 提升利用率本身不是目的，**压缩迭代延迟与提升真实算力产出**才是目标。

### 8. 运行环境事实与研发能效策略（Execution Environment & Efficiency Policy）
- **当前验证环境事实**：AutoDL 远程 Linux / CUDA, NVIDIA GeForce RTX 4080 SUPER vGPU 暴露, 32,760 MiB (32 GB) 可见显存, 320 W 标称功耗上限。P7 全量测试数据均基于此环境实测。不保证所有其他 32 GB 硬件展现完全一致的吞吐。
- **研发能效策略**：在已通过验证的环境中，SpatialForge 研发与实验应默认采用 `max_performance` Profile，以缩短研究迭代延迟、节省计算开销。质量门控与因果设计标准保持不变。

---

# 10. Track A 研究规格 — Embodied Explorer（PLANNED）

历史 v2.1 名称表示能力主题，不是版本排期；此项是从“系统选择 Camera”到“Agent 通过行动改变 Camera”的关键跃迁。

## 10.1 AgentState

计划结构：

```text
AgentState
├── position
├── body_heading
├── head_yaw
├── head_pitch
├── camera_height
└── optional velocity / action history
```

## 10.2 Action Space

第一版尽量简单、认知优先：

```text
move_forward(distance)
move_backward(distance)
strafe_left/right(distance)
turn_left/right(angle)
look_up/down(angle)
```

不进入底层 motor control。

## 10.3 Trajectory

```text
Trajectory
├── Observation_0
├── Action_0
├── Observation_1
├── Action_1
└── ...
```

## 10.4 关键研究问题

- 模型是否理解自运动导致的视图变化？
- 能否维持跨时刻 object identity？
- 被遮挡物是否能保持 object permanence？
- 能否建立持续的空间 belief？

---

# 11. Track A 研究规格 — Action-Conditioned World Model（PLANNED）

当前静态任务是 `WorldState + Observation → Prediction` 的受控空间推理 / experience infrastructure，尚不是完成的 world model；WorldState 是实验条件，实际模型输入仍受观测权限限制。

动态世界建模引入 `state_t + action_t → state_t+1` 或 latent 等价 `z_t + a_t → z_t+1`。状态转移可有随机性，不必限定为单点确定预测；模型内部 latent/belief 不等于可直接读取的全局真值。

候选 objective 包括 camera-motion-conditioned prediction、object permanence、occlusion persistence、future observation prediction、geometry/state transition、trajectory、latent transition、multi-step rollout、uncertainty/belief update。像素预测、结构状态预测与 latent prediction 均可评估；不预先指定 JEPA、DiT 或 autoregressive latent 为必选架构。

原“小脑 / System 1”只是 §2.4 的研究类比。推进条件是 action/transition 数据契约、可验证监督和相应时间外推评测成立；不要求先构建三个独立神经模块。

---

# 12. Track A 研究规格 — Active Observation（PLANNED）

模型开始自己选择下一观察动作，而不是系统喂固定 view。

基本回路：

```text
Question / Goal
      ↓
Observation_t
      ↓
Belief / uncertainty
      ↓
Choose Action_t
      ↓
Observation_t+1
      ↓
uncertainty decreases
```

研究指标：

- answer accuracy
- steps to answer
- path length
- information gain
- entropy reduction
- redundant observation rate
- failure type

核心思想：

> 下一视角选择应服务于减少答案不确定性，而不是随机探索。

---

# 13. Track A 研究规格 — Interactive Object Search / Embodied QA（PLANNED）

开始正式落地“找东西训练法”。

## 13.1 目标任务

```text
Find the glass cup.
```

Agent 可能需要：

- 搜索空间
- 改变观察角度
- 记住已经看过的位置
- 接近物体
- 绕开遮挡
- 打开容器
- 重新观察
- 最终定位并回答

## 13.2 状态变化

此阶段动作不仅改变 CameraPose，还可能改变 World State：

```text
State_t + open(cabinet) -> State_t+1
```

这是从 viewpoint dynamics 走向 interaction dynamics 的关键一步。

---

# 14. 3D God View / Research Inspector — VALIDATED / HUMAN UAT PASSED (TARGETED ADVERSARIAL REVIEW PENDING)

> **当前门控状态**：**VALIDATED / HUMAN UAT PASSED（定向对抗性复审 TARGETED ADVERSARIAL REVIEW PENDING，暂不标记 CLOSED）**。首个可用三维研究检视器切片已完成工程实现并通过人类研究员可用性验收；在定向对抗性复审与全链路边界加固最终闭环前，暂不标记为 CLOSED。目标是可用的研究检视器；当前已接入场景几何、视角、QA 样本、预测记录、Profile 与 telemetry。

## 14.1 God View 0 — Observation / Projection Contract

先解决已知语义前提：`CameraPose.fov_deg=60°`，而课程 renderer 固定 `cam_data.lens=35`，未将 fov_deg 应用到相机内参。历史默认传感器假设下水平 FOV 约 54.4°，并非所有 sensor/aspect 下的固定值。当前 center-based G2.0-C truth 不依赖投影，因此不受此差异影响；精确 frustum 与像素投影则受影响。

本子门控必须冻结：camera intrinsics、FOV 是水平/垂直/对角的定义、sensor size / sensor fit / lens 解释、image resolution/aspect（含 pixel aspect 和实际输出比例）、render metadata、projection 与 clipping/pixel coordinate 语义，以及坐标变换约定。当前 world Z-up、camera +X right/+Y up/+Z forward，Blender camera 看向 local -Z；应给出明确变换，不能只画一个标称 60° 视锥就声称与旧 RGB 一致。

验收应以代表性几何点与渲染投影对齐为证据，包含非方形 aspect 和边界点；缺失历史内参必须标记 unknown / reconstructed with provenance，不能假装 manifest 已提供完整 K 矩阵。优先明确历史图像的实际投影，保留原始 manifest、图像、哈希与已关闭真值；若需要改 renderer/FOV，建立新版本数据和独立回归，不替换冻结 E1 实验输入。投影/可见性作为新增 truth layer，绝不重定义 G2.0-C。

## 14.2 God View 1 — Inspector Data Contract

定义未来轻量 **Inspector Event / Replay Stream**：关联 run/config identity、`scene_id`、`view_id`、`sample_id`、`training_profile_id`、step（明确 microbatch/sample/optimizer/scheduler）、loss、camera pose、observation reference、target、prediction when available、telemetry reference。事件需带来源/版本及缺失状态；精确 schema 留给实施门控，不假定当前 `progress.json` 已有 sample-level 完整 replay。

特权 truth、监督 target、合法模型输入分字段/通道，Inspector 在 evaluator 侧关联。模型 prompt 保持 image + question，不能包含隐藏坐标、全局 scene graph、ground truth overlay、God View 图像或答案。SFT target 是合法监督，不能回流成推理输入。未来本体感知/受限位姿必须有任务授权的传感器契约，不能借相机调试数据开放全局真值。

训练热路径不得承担 Blender/3D 渲染或阻塞 UI 通信。优先消费现有不可变产物与 progress/telemetry 快照，未来事件使用有界队列、限频、异步或离线重放；UI 断开不能暂停训练或改变 RNG/样本顺序/梯度。可丢弃或合并显示用 telemetry，但必须显式报告缺口，不能丢科学结果后伪称完整 replay。开启/关闭 Inspector 应比较样本与更新计数、数值容差及耗时，探针成本单独计量。

## 14.3 三种检视模式

| 模式 | 展示内容 | 能力边界 |
|---|---|---|
| Training Inspector（NEXT） | scene、camera、已有 RGB、sample/question、supervision target、loss、samples/optimizer/scheduler 进度、profile、GPU telemetry | 普通 SFT microbatch 不是 Agent action；当前训练不进行具身环境行动。完整 sample replay 还需数据契约 |
| Evaluation / Inference Inspector（NEXT） | scene truth → camera → observation → question → raw/parsed prediction → ground truth → error classification | 预测比真值分析优先消费 `EvaluationPredictionRecord`；没有预测时显示 unavailable，不能编造 confidence 或 reasoning |
| Embodied Episode Inspector（FUTURE） | Observation_t → available belief/decision trace → Action_t → transition → Observation_t+1 | 仅在具身基础设施存在后开放；belief 若不可观测就不显示为已知内部状态 |

禁止为了 UI 在每个训练 microbatch 强制 greedy generation。训练中的 prediction 可来自明确标记的 optional periodic probe、独立异步评估或历史 replay；探针不得扰动正式训练状态或被误标为当前权重的即时输出。VSR test-only 约束同样适用于 Inspector，不能用于在线挑选 checkpoint、超参数或 seed。

## 14.4 God View 2 — Minimal 3D Vertical Slice

首个可用切片范围固定为：

1. 从权威 SceneState 按 renderer 的 primitive/size 规则绘制实际几何，保留 object_index 身份；画几何不等于拥有 bbox/surface 关系真值。
2. 展示 camera pose 与通过 God View 0 校准的 frustum/projection；无法重建内参时显式标记，不作精确投影声明。
3. 并列展示选中相机已有渲染 observation，区别于研究员自由漫游相机；当前称 observation view，不伪称已存在 embodied Agent。
4. 选择 QA sample，显示相应中心关系 truth、neutral/front-halfspace 标志与监督 target。
5. 有匹配评估记录时展示 raw/parsed prediction、判卷与错误类别。
6. 读取训练 telemetry / profile 状态，并可按样本导航或重放；UI 切换选择不能改变正在运行的冻结 profile。

验收：跨视角切换保持 scene/sample/prediction 对齐；清晰区分可用、缺失、未来图层；合法模型输入中不存在 Inspector 特权字段；离线数据可检视；训练吞吐不被同步渲染或逐批生成拖慢。动态物理、规划、active exploration、world-model rollout 不进入此切片。

## 14.5 真值与判卷边界

当前自动判卷限于已定义中心关系：camera-frame left/right、above/below、signed-depth front/behind、Euclidean camera-distance near/far、world metric distance 和 front-halfspace flag。`in_front_of_camera` 仅为 depth > eps，不能标成 visibility。

FOV、像素投影、occlusion、bbox relation、surface/contact、containment/support、target reached、trajectory success 都需要各自新增语义和验证；现有 relation.py 不提供它们。人工标注与诊断应单独记录来源，不覆写冻结 ground truth；可提出捷径/幻觉假说，但不能从正确答案或 UI 观感直接证明模型内部理解或因果机制。

## 14.6 人类可用性测试（Human UAT）科学发现与路线图重构动因

在完成 God View G1（三维视口交互、Inspect Sample / Explore Scene 双模式解耦、搜索式样本选择器、抽屉式 Profile / Preflight 管理与统一 Inspector 表面）工程实现并由人类研究员完成可用性验收（Human UAT）的过程中，God View 发挥了其作为“特权研究者与真值验证基础设施”的核心科学职能：**通过允许人类研究员直接在三维几何空间中漫游视锥，并与渲染 RGB 观测、空间真值和模型预测进行多模态同步比对，直观暴露出当前受控数据集的四大深层科研局限**：

1. **历史固定画幅在稀疏场景中过松（Loose Framing）**：
   历史固定的相机距离（$r = 6.0\text{ m}$）对于由 3–4 个基元构成的稀疏场景而言过于空旷。物体在 512×512 渲染图像中仅占中央狭小面积，边缘背景无效空白过大，像素有效承载率偏低，未能充分利用 VLM 的视觉分辨率。
2. **当前受控训练视点过于稀疏（Sparse Viewpoints）**：
   正式受控实验目前仅使用 4 个基准方位视角（four cardinal viewpoints: south, east, north, west）以及同分布下的 15° 有界扰动视角。对于空间智能下一阶段所需的连续视点变换与立体空间表征学习，视点采样密度和空间覆盖维度严重不足。
3. **场景复杂度与视觉推理难度过低（Trivial & Sparse Scenes）**：
   场景仅包含 3–4 个规则摆放的几何基元，缺乏明显的纵深层叠（depth layering）、投影视线重叠候选（projected overlap candidates）与同类视觉干扰物（distractors），模型仅凭粗粒度单图视觉线索即可答对，不足以强迫模型建立真正的三维空间结构认知。
4. **静态多视角本身不足以作为通向具身主动观测的唯一桥梁（Static Multiview Insufficiency）**：
   离散、静态的多视角切片缺乏相机运动的连续几何流（continuous motion parallax）、时序连贯性及渐进式遮挡揭示（progressive disocclusion），无法独立承担向最终具身第一人称智能体（Embodied Agent）跃迁的全部训练职责。

**科研决策与路线图重构**：
- **已关闭门控科学效力完全保留**：上述发现属于 God View 带来的更深层实证洞察，**绝不代表 G2.0 已关闭门控是错误或无效的**。G2.0-A 至 G2.0-E3.3A 的数据、代码与 GPU 训练优化闭环是坚实的基础设施。
- **A/C/D 实验设计作为受控静态基线保留**：保留冻结的 A/C/D 样本集、LoRA 模块定义与严格对齐协议。
- **顺延正式多 seed 训练**：若在画幅过松、视点过稀、场景过简的现有旧数据上盲目推进多 seed 训练，将浪费大量 GPU 计算并得出受限于玩具数据的平庸结论。因此，正式多 seed 实验执行顺序后移，优先在 G2.1–G2.3 构建动态视点、进动轨迹、场景挑战分级与渲染遮挡真值。
- **God View 基础设施持续演进**：God View 维持特权外部验证平面角色，后续将扩展支持：连续进动轨迹路径（Trajectory Path）显示、当前轨迹帧（Trajectory Frame）步进播放、场景复杂度级别切换展示、像素级遮挡图层（Visibility Masks）叠加以及多帧时序 QA 检视。God View 特权真值绝不泄漏至模型输入。

---

# 15. 尺寸—距离混淆实验（I5；RESEARCH OPTION）

这是 v1 发现延伸出的重要可证伪假设。涉及 projected size / FOV 的实验须先通过 §14.1 投影契约，尚非已完成结果。

必须明确区分：

```text
physical_size
metric_camera_distance
camera_depth
projected_apparent_size
```

构造 counterfactual pairs：

```text
Large + Far
Small + Near
```

以及控制：

- 相同投影大小但真实距离不同
- 相同真实大小但距离不同
- 相同 depth 但 lateral offset 不同
- camera FOV 改变导致 apparent size 改变

如果模型仍用 apparent size shortcut，其错误模式应被系统性放大并可测量。

---

# 16. 数据契约与 Data-Plane Scaling（PLANNED）

当前已有 `CameraPose`、`SceneObject`、`SceneState`、`ObservationMetadata`、`SpatialTruthRecord`、`QASample`、`MultimodalSample`、`EvaluationPredictionRecord`。不要把已实现真值/评测类再次列为待建的 SpatialRelationRecord / EvaluationRecord，也不为新名称重命名旧契约。

未来按任务增补 Action、Transition、Episode、TemporalState、TrajectoryStep、agent-local BeliefState 与 Inspector events。保持单一权威 pose，不同时维护相互冲突的 Euler/yaw/pitch/target 真值；版本化 JSON 契约保留 seed、scene/view/sample 身份、split、来源与合法输入权限。

高上限不只是模型显存问题。当前 JSONL manifests、本地 PNG、内存预制 tensors、图像特征缓存适合当前规模，不是永久数据架构。未来按证据引入 sharded datasets、streaming、parallel rendering/simulation、distributed generation、checkpointable data iteration、remote/object storage adapters。

不可变 dataset provenance 必须跨存储布局保留：逻辑样本身份、内容哈希、生成配置、renderer/模型/processor/truth 版本、split 与采样策略独立于物理 URI。当前 E1 content hash 仅涵盖协议指定字段及 image_path，不能当成完整图像字节 provenance；扩展时新增图像/分片校验，不能重定义或覆盖旧哈希。恢复迭代需记录 shard/样本位置、shuffle/RNG 与 worker 划分，避免重复/遗漏或跨 split 泄漏。这些均未实现，本轮不迁移数据。

---

# 17. SimulationBackend 的采用边界

**CURRENT**：Blender 提供 controlled rendering、synthetic scene generation 与 camera observation；它也是未来 God View 几何重建的重要参照，当前尚无 God View UI。不要仅为 realism 替换已验证 renderer。

**PLANNED**：按 §5 的 SimulationBackend 契约统一实验语义，而非重写外部模拟器。物理/机器人后端由 interaction/dynamics 的具体需求选择；真实采集输入可通过观测数据导入接入，不伪装成支持 action/step 的模拟器。后端必须声明控制、传感器、坐标、单位、真值与确定性能力；不兼容能力显式拒绝。没有第二个实际用例前不建设庞大插件框架。

---

# 18. 环境逼真度：Track C 的课程维度（PLANNED）

受控 primitives 持续保留，支持因果实验与确定性 truth。现实感是额外课程维度，不取代可控性；图像更逼真也不自动得到正确物理真值。

在已识别 domain gap 后，分别评估资产、材质、光照与布局变化，再考虑 interaction/dynamics。Objaverse、AI2-THOR assets、Habitat-compatible scenes、procedural rooms 保留为研究候选，未作当前集成声明。每次扩展记录 renderer/数据版本并控制变量，避免把多个域变化混为一个实验。

---

# 19. 计算资源策略

## 19.1 本地 Windows

角色：**用户侧控制 / 协调环境**。

负责：

- 用户控制与协调
- Git 决策（commit / push / merge 节奏）
- 本地 review
- 需要时可运行轻量级检查（例如快速 CPU 验证 / 代码审查）

说明：当前 OpenCode CLI 直接运行在 AutoDL 实例上。本地 Windows 仍是用户侧控制 / 协调端；当有用时，本地仍可运行轻量级检查，并不被禁止运行测试或 Blender。

## 19.2 AMD Radeon Developer Cloud — 当前开发工作流中停用

**已在当前开发工作流中停用。** 原因：其 One-click 池无法分配最低 GPU 资源请求，无法支撑当前开发。

说明：

- 当前执行后端为 AutoDL / CUDA。
- SpatialForge 长期目标是 **计算后端无关**；当前验证路径是 CUDA，ROCm 仍需独立兼容验证；这**不代表永久禁止** ROCm / AMD 支持。
- 若未来 AMD 节点可用并满足资源需求，可重新评估其作为可选计算节点。

## 19.3 AutoDL（CUDA Cloud）— 当前默认远程执行环境

- AutoDL 是当前主要 / 默认的远程执行环境（Linux / CUDA）。
- OpenCode CLI 直接运行在 AutoDL 实例上。
- AutoDL 当前承担 CPU / code 工作（CPU tests、代码开发等）。
- AutoDL 已用于历史 CUDA LoRA / evaluation / inference 验证；本轮未连接或复验该实例。
- Blender work：Blender 4.2.0 已在 AutoDL 实例成功安装并验证（`/root/autodl-tmp/tools/blender/blender`），已完成 G2.0-E1 全量 100 场景 800 张图像渲染。
- 本地 Windows 是用户侧控制 / 协调环境；需要时可运行轻量级检查。

说明：更早的 G2.0-B Blender 4-view smoke render 已在上一环境中通过（历史事实保留，见 §8.2）；G2.0-E1 800 张课程渲染已在当前 AutoDL 实例完整完成（见 §8.6）。

## 19.4 GPU 时间纪律

不要用昂贵 GPU 时间做：

- 文本编辑
- CPU geometry tests
- Git 操作
- 普通单元测试

只有 pipeline 已经稳定时才启动大规模计算。

## 19.5 高上限执行方向（PLANNED / ARCHITECTURAL）

学习策略与硬件政策分开，统一政策维度见 §5.3。可建议低资源策略，但当用户有足够资源并明确请求已支持的高容量策略时，不得强制使用 PEFT。反之，当前尚未支持 full FT 的事实也不能被界面隐藏。能力不匹配应报告所选组合、估算需求和备选项，保留用户选择，禁止 silent fallback。

PyTorch-native **FSDP2** 是未来 fully-sharded training 的首选评估候选，可将 parameters、gradients、optimizer states 分片到多个 worker，以通信换显存；这不保证所有模型/PEFT/缓存组合可用。[PyTorch fully_shard 官方说明](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)

当模型可复制到每卡时可考虑 DDP；按模型大小、序列长度、网络拓扑与实测瓶颈评估 FSDP2、Tensor Parallel、Pipeline Parallel、Context Parallel 与 Distributed Checkpoint 的组合。DeepSpeed ZeRO-3 保留为可选互操作后端。TorchTitan 仅作为 PyTorch composable parallelism 的设计参考，不是必需依赖。[TorchTitan 官方仓库](https://github.com/pytorch/torchtitan)

引入条件：先建立相应 ModelBackend/trainability 与数据契约，证明单卡参考语义，再验分布式 loss normalization、global batch、采样/seed、checkpoint-resume、失败恢复和资源成本。不宣称上述后端已实现，不因某个 27B 模型固定一种拓扑。

SpatialForge 应在后端能力内 model-scale agnostic：3B、7B、27B、72B+ 或未来架构只是可能实验变量，不承诺最大参数量。扩容服务于空间/世界智能的可测科学问题，不以参数规模定义项目 ambition；数据面的扩容同样必需（§16）。

---

# 20. ROCm 纪律

> 当前开发工作流未使用 AMD 节点（见 §19.2）。如未来重新启用 AMD / ROCm 节点，遵循以下纪律。

AMD 节点：

- PyTorch 使用 HIP compatibility
- `torch.cuda.*` 可保持 PyTorch API
- `torch.version.hip` 应非空
- LoRA 优先 bf16

禁用/谨慎：

- bitsandbytes
- autoawq
- flash-attn（除非后续明确验证 ROCm 兼容）

原则：

> SpatialForge 不能变成 AMD-specific project；CUDA / ROCm 都只是计算后端。

---

# 21. Git 与存储纪律

## 21.1 GitHub

角色：

> 代码、文档、可复现实验结果的唯一长期真源。

## 21.2 云服务器

原则：

- pull/fetch 为主
- 不把云节点当唯一代码副本
- 大模型权重/可再生缓存不进入 Git

## 21.3 大型输出

默认不提交：

- rendered PNG bulk datasets
- model weights
- temporary manifests generated only for smoke tests
- caches

提交：

- source code
- tests
- compact evaluation result
- experiment configs
- small goldens / fixtures when necessary

---

# 22. AI 协作与开发流程

原“一切用户手动执行”策略已更新。

以下 CLI、模型与版本是历史执行 provenance（本轮未作远程复验），不是项目架构依赖；实际工具可按 gate 更换，并随该次执行记录：

- OpenCode CLI 直接运行在 AutoDL 实例（远程 Linux / CUDA）上
- 历史 OpenCode 版本：1.18.29
- 历史执行模型：DeepSeek V4 Flash · low
- 本地机器是用户侧控制 / 协调端
- GitHub 仍是代码 / 文档长期真源
- 用户保留最终 commit / push 决策

## 22.1 角色划分

### 架构 / 协调角色

负责：

- 研究路线
- 架构设计
- 门控
- Prompt 设计
- 代码结果复审
- 实验解释
- 项目记忆维护

### 实现 Agent

负责：

- 实际代码修改
- 测试
- smoke checks
- 小范围修复
- git diff / status 检查

### 独立复审角色

负责核对实现证据、科学边界与验收条件；重要 gate 独立于实现过程复审，工具与模型按次记录。

### 用户（最终 Git 决策权）

负责：

- 最终决策
- 亲自 Git commit
- push / merge 节奏
- 实验方向裁决

## 22.2 历史工具选择示例

以下保留当时 OpenCode / DeepSeek 的使用经验，不作为所有 gate 的固定默认或能力要求：

```text
Default: DeepSeek V4 Flash · low
        ↓ first failure
improve prompt / reduce scope
        ↓ still fails
DeepSeek V4 Flash · medium
        ↓ rare complex case
high temporarily
```

免费模型可用于低风险任务，但如果出现：

- context explosion
- compaction drift
- connection instability

立即退出，不反复消耗时间。

## 22.3 Prompt 原则

低成本执行模型的 Prompt 应：

- 指定允许读取的文件
- 指定修改范围
- 明确禁止 broad refactor
- 明确 acceptance criteria
- 明确“不 commit”
- 输出尽量短
- 一个会话只做一个小任务

长期记忆放在：

- Git commits
- 本文
- 项目级 AI memory

而不是依赖 OpenCode 单个超长 session。

本轮由 Codex 在同步后的本地分支执行 docs-only 架构审查，不改变上述远程工程工作流记录。

---

# 23. 门控协议

每个门控都必须提前定义 PASS 条件。

格式：

```text
Goal
Scope
Files
Acceptance criteria
Tests
Smoke validation
Git state
Commit
```

门控完成后：

1. 测试通过
2. diff check
3. 独立复审（重要阶段）
4. 用户亲自 commit
5. 更新本文 Current State / Roadmap

禁止：

> “代码看起来差不多”就宣布完成。

---

# 24. 唯一长期设计原则与科学纪律

**LOW FLOOR, HIGH CEILING — “降低的是使用门槛，而不是工具能力上限。”**

算力可用性决定 execution strategy，不定义 scientific capability ceiling。低资源研究者可以选择合适 PEFT/精度/执行政策，高算力研究者可在支持范围内选择更大训练范围；这不等于当前已支持所有硬件与策略。

| 长期原则 | 具体约束 |
|---|---|
| Scientific semantics before optimization | 有用吞吐与迭代成本优先于利用率数字；不以性能改变 objective、split、标签或冻结协议 |
| No silent fallback | 能力不匹配显式报告、估算需求并建议替代，用户决定策略；不强迫高算力用户 PEFT |
| Controlled worlds before realism | 保留可控合成基准，现实感/模拟器作为新增课程能力 |
| Truth extensions are additive | G2.0-C 中心坐标、signed depth、Euclidean near/far、world metric、neutral、front-halfspace 语义保持；新投影/FOV/visibility/occlusion/bounding/surface/contact/containment/support/dynamic/trajectory truth 版本化、尽可能确定性 |
| God View privilege isolation | 世界监督、合法输入、研究检视分离；target 不回流到推理输入 |
| Model / simulation / learning / compute independence | 长期按能力组合；当前专用实现的耦合如实列债务，不伪装成已完成后端抽象 |
| Scale follows scientific evidence | 参数量/资源是实验变量；外部迁移证据先于广泛能力声明 |
| Architecture now, implementation when demanded | 现在声明边界，等实验需求出现再落地；不提前建设 speculative infrastructure |

## 24.1 控制变量

优先做可以解释因果的实验：

```text
same scene
same objects
only CameraPose changes
```

或：

```text
same model
same eval
only training curriculum changes
```

## 24.2 Scene-level split

训练/验证/测试按 Scene 切分，避免同世界不同 camera 泄漏到不同 split。

这是 multi-view 数据尤其重要的红线。

## 24.3 不能把 synthetic 提升直接称为通用提升

必须区分：

- in-domain synthetic accuracy
- cross-view generalization
- cross-scene generalization
- external benchmark transfer
- real-world transfer

VSR 是当前重要的外部真实图像基准，并按 protocol.py 保持 final test-only：不得用于 checkpoint selection、early stopping、超参数或 seed 选择。长期评估按任务涵盖 controlled synthetic、real-image spatial reasoning、video/temporal、embodied interaction、physical prediction；当前没有后三类的新 benchmark 结果。

## 24.4 失败结果也保留

例如：

- multi-view improves synthetic orientation but not VSR
- jitter hurts stability
- larger model worsens near/far

都属于科学证据，不应“优化掉”。

---

# 25. 研究假说与 Verifier / Reward 方向

以下 I1/I3/I4/I5/I6/I7 是研究方向或待交付架构，不能作为已实现创新成果；I2 的历史诊断证据见 §3。

## I1. Action-conditioned next-view / latent-state prediction

`z_t + a_t -> z_t+1`

## I2. Orientation 结构性短板诊断

v1 已提供证据链，v2 直接针对其设计 curriculum。

## I3. Active Observation as uncertainty reduction

下一观察动作以减少答案不确定性为目标。

## I4. Counterfactual physical / spatial engine

用于：

- support
- gravity
- occlusion
- interaction traps
- viewpoint counterfactuals

## I5. Size-distance confusion hypothesis

构建可证伪的数据与分析工具。

## I6. Brain / Cerebellum / Brainstem cognitive decomposition

明确认知世界模型与机器人控制边界。

## I7. Privileged God View + restricted Agent View evaluation architecture

人类拥有全局监督能力，但 Agent 严格受传感器与动作接口约束。

## Verifier / Reward interface（PLANNED；算法未定）

当前确定性 geometry truth + parser/evaluation 不等于通用物理 verifier。未来接口应定义输入、输出类型、有效域、误差/容差、版本、失败状态与 reward 映射；可验证 geometry consistency、collision/penetration、visibility consistency、state transition error、trajectory error、goal success、physical constraint violation。

先验证 verifier 再选择学习算法。PPO、GRPO、SAC、DPO-style preference optimization、offline RL 或其他方法仅为依 action/output 与监督形式决定的选项；偏好优化也不自动等于在线环境 RL。禁止从 static QA 直接跳到某个普适 RL 算法或把 GRPO 写死。

---

# 26. 产品化界面（PLANNED；God View 首门控见 §14）

统一 CLI 是未来产品入口，可覆盖 scene generate、render multiview、curriculum build、train、eval、explorer run；当前各 scripts 不等于已实现此统一命令组。

God View Workbench 的模式、数据与首切片只在 §14 定义。未来 UI 应把 **Learning Strategy** 与 **Performance / Compute Profile** 分开，再展示 model、execution、precision、task 的已解析组合和能力限制；profile 精确参数统一引用 §9。运行开始后选择器只读，改变配置必须产生新 run/config identity。

可接入已有 GPU utilization、Torch allocated/reserved、NVML memory、power、SM clock、samples/s、processed samples、optimizer/scheduler progress 与 ETA；这些 telemetry 字段有实现，并不代表 Dashboard 已完成。指标显示实际来源与采样时刻，不拿 P7 历史值当实时读数。

后续 Experiment Dashboard 可显示 dimension-wise scores、orientation/near-far errors、viewpoint-conditioned failures；trajectory success/efficiency 等待具身任务定义。人工 annotation 独立于权威 truth，不在此重复定义 NEXT。

---

# 27. 唯一路线图：近期门控 + 协同研究 Tracks

## 已关闭近期门控（HISTORICAL RECORD）

- **G2.0-A ～ G2.0-E3.3A**：已闭环完成几何核心、渲染管线、采样系统、真值引擎、QA 课程、工程 Runner 及 GPU 极速训练配置（P7 浸泡压测达 8.52 samples/s，9.18x 加速，全套 CPU 测试 261 项通过）。
- **3D God View / Research Inspector（G1 切片）**：**VALIDATED / HUMAN UAT PASSED（定向对抗性复审 TARGETED ADVERSARIAL REVIEW PENDING，暂不标记 CLOSED）**。已交付完整三维视口交互、双模式分离（Inspect Sample / Explore Scene）、搜索式样本选择器、抽屉式硬件 Profile / Preflight 管理与统一 Inspector 面板。

---

## 唯一执行路线图演进（NEW IMMEDIATE ROADMAP）

```text
God View G1
VALIDATED / HUMAN UAT PASSED
(targeted adversarial review pending)
       ↓
G2.1 Dynamic Viewpoint & Scene Challenge Foundation
  ├── G2.1-A: Adaptive Camera Framing
  ├── G2.1-B: Precessing Observation Trajectory
  └── G2.1-C: Controlled Scene Complexity V1
       ↓
G2.2 Render-derived Visibility / Occlusion Truth
       ↓
G2.3 Temporal / Multi-frame Spatial QA
       ↓
Controlled Formal Comparison + Factorized Ablation
(Static A/C/D Baselines vs. Dynamic Curricula)
       ↓
Action-conditioned Camera
       ↓
Embodied First-person Agent
       ↓
World Model
       ↓
Active Observation
       ↓
Interactive Object Search
```

---

### NEXT 1 — G2.1 Dynamic Viewpoint & Scene Challenge Foundation（NEXT / COMMITTED）

本门控针对 God View 人类 UAT 暴露的画幅过松、视点过少、场景过简问题，建立从静态离散多视角向连续空间观测跨越的技术底座。

#### G2.1-A Adaptive Camera Framing（自适应相机画幅）
- **核心逻辑**：严禁粗暴地将 `radius = 6.0` 替换为另一个静态魔法常数。引入基于场景三维包围体（Bounding Extent）的自适应求解器：
  $$\text{Scene Geometry Extent} \longrightarrow \text{Framing Solver} \longrightarrow \text{Desired Projected Occupancy} \longrightarrow \text{Camera Distance } r$$
- **科研约束**：
  - 历史渲染数据（E1 数据集，固定 $r = 6.0\text{ m}$）保持不可变只读参考。
  - 初期探索性目标设为物体在图像中投影占比约 65%–85%，但**该数值范围与具体占满度度量指标仅为探索区间，严禁在实验验证前冻结为先验科学真理**。

#### G2.1-B Precessing Observation Trajectory（连续进动观测轨迹系统）
- **定位**：在离散多视角与具身自主行动之间构建确定性的连续相机观测轨迹，属于**受控连续观测课程（Controlled Continuous Observation Curriculum），而非 Agent 自主行为（Not Agent Action）**。
- **概念参数化形式**：
  - 方位角进动：$\theta(t) = \omega_\theta \cdot t + \theta_0$
  - 仰角振荡：$\phi(t) = \phi_0 + A_\phi \sin(\omega_\phi \cdot t + \delta_\phi)$
  - 视距脉动：$r(t) = r_0 \cdot [1 + A_r \sin(\omega_r \cdot t + \delta_r)]$
  - 有界视线中心扰动：$\mathbf{c}(t) = \mathbf{c}_0 + \Delta \mathbf{c}(t)$
- **设计标准**：
  - 明确声明：**具体的波形与频率策略尚未冻结**。核心要求是在配置的 episode 周期内实现平滑、广阔的三维空间覆盖，绝非简单水平圆周绕轨，亦非短周期平凡闭环。
  - 保持场景为核心观测目标，视线中心扰动严格有界。
  - 轨迹在指定 seed 和配置下完全确定性可复现。
  - 引入概念数据契约：`CameraTrajectory` 与 `TrajectoryFrame`（包含 `trajectory_id`、`frame_index`、`normalized_time` / timestamp、`CameraPose`、配置溯源），且在 God View 中具备轨迹路径绘制与时序回放能力。

#### G2.1-C Controlled Scene Complexity V1（受控场景复杂度分级 V1）
- **基元约束**：第一阶段继续严格采用可控几何基元（cube, sphere, cylinder），暂不引入复杂物理模拟或非刚体/复杂网格资产。
- **概念难度阶梯（C0–C5 设计包络）**：
  - **C0（Calibration）**：3–4 物体，基础空间关系校准。
  - **C1（Mild Clutter）**：5–7 物体，轻度视线遮挡与干扰。
  - **C2（Depth Layering）**：7–10 物体，明显纵深层次（前景、中景、背景）。
  - **C3（Projected Overlap）**：8–12 物体，受控视线投影重叠候选。
  - **C4（Strong Clutter & Distractors）**：10–14 物体，高密度杂物与多重干扰物。
  - **C5（Dense Spatial-Relation Reasoning）**：12–18 物体，高密度空间拓扑与复杂关系推理。
- **重要说明**：
  - **C0–C5 的物体数量范围为临时设计包络（Provisional Design Envelopes）**，最终阈值将在 G2.1-C 实施中根据描述符容量（descriptor capacity）、视觉指称唯一性（visual-reference uniqueness）、碰撞放置可行性（collision/placement feasibility）、有效 QA 产出率（useful QA yield）、渲染质量与计算成本综合确定。
  - **本门控不声称拥有“遮挡真值（Occlusion Truth）”**，相关表述使用“投影视线重叠候选（projected-overlap candidates）”、“杂乱度（clutter）”与“纵深层叠（depth layering）”，因为权威遮挡语义需待 G2.2 确立。
  - 避免使用未有契约定义的“复杂空间拓扑”，统一采用“高密度空间关系推理（dense spatial-relation reasoning）”。
- **多维挑战轴**：不仅考量物体数量，重点构建：
  1. *尺寸—距离混淆挑战（Size-Distance Confounds）*：小物体在近处 vs 大物体在远处，形成相似视网膜投影尺寸。
  2. *视觉描述符相近干扰（Descriptor Similarity）*：例如同一场景中并存蓝色球体、青色球体、蓝色圆柱与绿色球体。
  3. *视点依赖重叠（Viewpoint-Dependent Overlap）*：物体在某一视点产生严重投影重叠，而在另一视点完全分离。

---

### NEXT 2 — G2.2 Render-derived Visibility / Occlusion Truth（PLANNED）

- **保留 G2.0-C 几何语义**：严禁修改 G2.0-C `SpatialTruth` 语义。`in_front_of_camera` 必须严格保持为相机前半空间几何真值（depth > 0），**绝不得从相机深度或物体中心点空间关系推导能见度/遮挡**。
- **能见度必须源自渲染实据（Render-derived Visibility）**：
  - **严禁声称单次 RGB + Instance Segmentation 即可充分推导出完整的 `visibility_ratio`**。
  - 严格区分两类像素度量：
    - `visible_pixel_count`：可在常规全场景实例/ID 渲染中直接获得（实际未被遮挡的像素数）。
    - `reference_projected_pixel_count`（无遮挡投影参考像素数）：需要经过显式验证的参考渲染方案（例如逐物体剪影/单独无遮挡投影渲染，或其他无遮挡投影方法）。
  - 概念可见度比率形式化为：
    $$\text{visibility\_ratio} = \frac{\text{visible\_pixel\_count}}{\text{reference\_projected\_pixel\_count}}$$
  - **具体的参考渲染实现流程留待 G2.2 门控正式冻结**。
- **真值字段与分类约束**：
  - 预期概念字段：`object_index`, `projected_pixel_count`, `visible_pixel_count`, `visibility_ratio`, `visibility_class`。
  - `fully_visible`, `partially_occluded`, `strongly_occluded`, `not_visible` 等分类阈值**严禁在 G2.2 获得实证验证前被冻结为权威定义**。

---

### NEXT 3 — G2.3 Temporal / Multi-frame Spatial QA（PLANNED）

基于 G2.1-B 提供的连续进动轨迹帧序列，设计多帧时序空间认知任务。

- **关于时序一致性的关键纠偏（Critical Correction on Temporal Consistency）**：
  - **相机相对空间关系绝非普适的时序不变量**。随着相机移动，`left_right`、`front_behind` 等视点条件关系发生变化是完全合法且自然的物理规律。
  - **严禁任何要求模型在帧间保持所有空间标签不变的错误表述**。
  - 将**视点条件时序一致性（Viewpoint-Conditioned Temporal Consistency）**严谨定义为：
    > 给定已知的相机运动轨迹，模型能否正确追踪相机相对空间关系随时间的动力学演进，同时保持世界级不变量（如物体身份连续性与静态世界几何结构）？
- **候选时序科研任务**：
  1. **视点变换推理（Viewpoint Transformation）**：相机平滑移动到新视角后，物体 A 相对物体 B 的方位朝向如何变化？
  2. **相机相对关系演进（Camera-Relative Relation Evolution）**：沿着连续观测轨迹追踪方位标签的动态反转与过渡。
  3. **静态世界自身运动一致性（Static-World Ego-Motion Consistency）**：在假定世界静态的前提下，追踪观测变化全由观察者自身运动引起。
  4. **客体恒常性（Object Permanence）**：仅在 G2.2 遮挡真值就绪后引入，追踪物体被短暂遮挡后再度出现的连续身份。
  5. **主动观测前序（Active-Observation Precursor）**：在给定存在歧义的当前视角下，模型预测并选择“下一步向哪个视点移动能最有效消除歧义”（预测优势视点，但不执行具身闭环）。
- **关于自身运动 vs 物体运动辨识的严格界定**：
  - **不得将自身运动 vs 物体运动辨识作为当前阶段的二元判别任务**，因为 G2.1 中所有物体保持静态。
  - 明确记录：*有意义的观察者运动 vs 物体运动分类任务，必须依赖未来扩展的受控物体运动与反事实动力学门控*。

---

### NEXT 4 — Controlled Formal Comparison + Factorized Ablation（DEFERRED / PLANNED）

- **定位与命名**：严禁将未来的静态 vs 动态对比自动称为“因果对比（Causal Comparison）”；严格规范表述为**受控对比与析因消融（Controlled Comparison and Factorized Ablation）**。
- **保留既有 A/C/D 协议作为静态基线**：
  - 完整保留已关闭的 A/C/D 实验设计、9 条件矩阵（B zero-shot 对照，A42/123/456，C42/123/456，D42/123/456）、配对比较（Paired Comparison）、McNemar 检验、Paired Bootstrap 95% CI 及 VSR 迁移评测规程。
  - 将其作为衡量动态升级收益的受控静态基线（Static Baselines）。
- **析因消融矩阵维度**：
  未来的正式实验应当解耦并隔离各维度的独立贡献：
  1. 静态基线（A/C/D 历史配置）
  2. 仅自适应画幅（Adaptive Framing Only）
  3. 仅场景复杂度提升（Scene Complexity Only）
  4. 仅动态轨迹观测（Dynamic Trajectory Only）
  5. 动态轨迹 + 场景复杂度组合
  6. 时序多帧课程（Temporal / Multiframe Curriculum）
  具体的全矩阵正交设计留待该门控实施前正式冻结。

**统计分析契约（正式执行前冻结，完整保留）**：逐一报告每个 seed，并汇总跨 seed 的 mean ± SD；sample IDs 可对齐时按同一 seed / 同一评测样本作 paired comparison，报告 accuracy delta 的 paired bootstrap 95% CI、paired binary correctness 的 McNemar test，以及 discordant-pair odds ratio / paired effect diagnostic。报告不一致对计数，并预先约定零计数处理；不以 Cohen’s d 作为配对二元正确性的主要效应量。

执行前冻结 success/regression criteria、主要比较与分析单位、bootstrap 重采样单位及相关性处理（同场景多样本不能默认独立）、统计检验规则；未有冻结数值阈值时不补造阈值。区分评测样本不确定性与 seed 间训练变异，不把同一样本的多个 seed 预测当独立样本，也不将 n=3 宣称为高统计功效的总体推断。VSR 继续仅作 final test，不用于 checkpoint selection、hyperparameter tuning 或 seed selection。

## 长期 Tracks（PLANNED；不虚构总顺序或日历日期）

以下箭头表示能力依赖方向，细部次序由实验决定，均不插队到 NEXT 1/2 之前。

| Track | 能力演进 | 跨 Track 进入条件 |
|---|---|---|
| A — Spatial / Embodied Research | embodied observation loop → active observation + memory/belief → interactive object search → dynamic scene transitions → world-model objectives → multi-step rollout → planning | camera-only motion 可先研究 next-view；对象交互依赖 C 的 dynamics 和 D 的验证；§10–§13 是规格，不是发布承诺 |
| B — Training Capability / Scale | training generalization → model-independent trainability → PEFT variants / selective/full FT → continual pretraining；按需多 GPU / 大模型 | 方法不是质量阶梯；由科学问题决定哪些分支必需。缓存兼容、数据面、单卡参考与恢复验证先于规模扩张 |
| C — Environment / Simulation Fidelity | controlled primitives → validated projection/FOV → visibility/occlusion truth → richer assets/materials → interaction/dynamics → optional robotics/physics backends | God View 0 是此 Track 的近期起点；新增 truth version 不覆盖旧 E1 数据或中心关系 |
| D — Verification / Closed Loop | deterministic geometry verifier → richer geometry → temporal/dynamics verifier → trajectory verifier → verifiable reward → RL/policy optimization where justified | 与 A/C 的输出和动作契约对齐；可靠 verifier 与收益证据先于算法选择 |

当前 A/C/D 课程实验结果决定下一项投入；即使结果为负，仍是路线选择依据。FSDP2、full FT、27B/72B+、机器人模拟器均为按需扩展，不是下一天的实现工作。

---

# 28. 架构审查记录与显式结构债务（2026-09-08）

架构修订的审查基线为 §0 的 `3fb0e65`，审查时最新实现基线为 `f4a0089`；当前仓库 HEAD 在接手时通过 Git 重新核对；没有复用旧 main 审查。架构裁决为 **major revision**：原计划混合了当前静态 QA、未来具身能力和固定模型/硬件假设，需重建职责与能力边界，而非小幅补文案。只编辑本文件，不运行模型/训练/Blender、不安装依赖、不 stage/commit/push。§8–§9 保存门控与性能证据，§27 是唯一执行路线。

| 当前限制 / 文档纠偏 | 核对来源与处置 |
|---|---|
| 旧 §28 与 §8 冲突：C 写 104/42，D 写 104 pre-existing + 37 | 已以 §8 和门控提交内 test 方法静态计数校正：`19e3526:tests/test_spatial_truth_engine.py` 36，`6823f29:tests/test_qa_curriculum.py` 43；保留 C 总 98、D 总 141。静态计数不是本轮测试执行 |
| 旧 §28 把 B.1 的 38 新测试称为 intrinsic/frustum 覆盖 | `6d08fcf:tests/test_camera_sampling.py` 确为 38 个 test 方法，主题是采样；不据此声称已验证图像 frustum/projection。B.1 总 62、历史 Blender 坐标 parity 与 §8 的几何回归记录保留 |
| Model / learning / execution 耦合 | 当前 Qwen 专属类/LM 路径、冻结视觉与文本 embedding 预制、single CUDA/device 0、AutoDL 路径；未来接口见 §5，未在本轮实现 |
| 历史配置与现行 Profile 同名风险 | `FormalTrainingConfig()` 的历史默认与 `.from_profile()` 不同；精确字段必须随 run 记录，不能只比较字符串 reference |
| Capability preflight 范围有限 | 当前主要按总显存容量阈值判断；非完整实时空闲显存/全后端能力证明。保留 No-Silent-Fallback 行为，未来扩充检查 |
| 投影与 truth 缺口 | camera.py 的 60° 与 renderer 35 mm 未对齐，ObservationMetadata 未保存完整内参；God View 0 优先补契约，不重定义 G2.0-C |
| 任务、数据面、动态闭环未泛化 | 当前 QA/JSONL/本地 PNG/全量内存准备；无通用 objective、SimulationBackend、Episode、verifier/reward 或分布式恢复实现 |
| Inspector 交付与 UAT | G1 切片交付完整三维视口、双模式与统一面板并通过人类 UAT；暴露出画幅过松、视角过少、场景过简等 4 项实据；处于对抗性复审阶段，暂不标记 CLOSED |
| 科学结果边界 | Formal 3-seed 尚待执行；S2 仅同 jitter 过程，synthetic 成功不足以证明外部迁移或新视觉表征 |
| P7 计时边界 | 数值原样保留，benchmark timer 为训练循环；不是含模型加载/冷缓存/保存/评测的总任务时间 |

历史补充事实统一保留：E2 train 与 held-out S1 greedy inference smoke 无 ground-truth 输入泄漏；adapter reload 保持 0 visual LoRA；E3.3A 25%→100% Host RSS 漂移 0.39 MB、reserved plateau ~23.1 GB、该次运行无 OOM/无单调泄漏，详见 §9。环境的 OpenCode/AutoDL/Blender 状态属于原计划记录，本轮未作远程实测。

针对性自审：闭合 truth 含 neutral/identity/depth/distance 含义未改变；未来能力有状态标记；无 QLoRA/DoRA 普适优劣断言；Qwen 是当前 backend；world model 是未来动态目标；特权通道与 hot-path 成本隔离；学习与执行/精度/任务独立声明；NEXT 2 科学矩阵保留。当前与历史状态不再在这里复制整块快照。

---

# 29. 新 AI 接手时的决策检查表

在提出改动前先回答：

1. 这个改动是在服务最终 embodied world-model 愿景，还是只是让固定-camera pipeline 更复杂？
2. 是否区分 Diagnostic Camera / Embodied Camera / God View？
3. 是否保持 World State 与 Observation 分离？
4. 是否会造成 Agent 读取 privileged state？
5. 是否能通过受控实验验证科学价值？
6. 是否会破坏 v1 / G2.0-A / G2.0-B reproducibility？
7. 是否值得现在实现，还是属于过早抽象？
8. 是否能用 CPU test 验证核心逻辑？
9. 是否需要 GPU，还是可以等 pipeline 稳定再烧算力？
10. 完成后是否更新本文？

---

# 30. 一句话项目定义

> **SpatialForge 是面向空间智能的研究基础设施：当前以受控合成世界、多视角 QA、训练与外部评测建立证据，长期连接具身观测、动作条件世界建模、多种学习与计算后端，并通过不向 Agent 泄露特权真值的 Research Inspector 支持研究。**

---

# 31. 文档维护规则

本文件是活文档。

每完成一个关键门控后更新：

- §0 快速拉起
- §8 已完成门控
- §27 唯一路线图
- §28 架构审查记录 / 债务
- 相关架构/决策章节

如果路线发生变化：

> 允许修改，不需要为了“保持旧计划正确”而硬撑。

但已经完成的 commit / 实验事实必须保留历史记录，不能事后改写。

---

# 32. Embodied Vertical Slice Record（2026-09-09）

> 本轮交付对应文件全部属于 **未提交工作树**（用户控制最终 Git 操作）。以下记录
> 仅描述实现事实；完整逐项报告见交付说明（Final Report）。

## 32.1 依赖 pin（本轮新增，仅 Embodied 主线）
| package | version | reason |
|---|---|---|
| `ai2thor` | `5.0.0` | 官方 AI2-THOR 控制器；当前 ProcTHOR-10K 需 ai2thor 5.0+；需 Python ≤ 3.11（独立 Python 3.10 venv） |
| `prior` | `1.0.3` | 官方 AllenAI ProcTHOR-10K 数据集加载器 |
| `procthor` | `0.0.1.dev2` | 官方 ProcTHOR house / asset metadata（generation，备用） |
| `numpy`/`Pillow`/`opencv-python-headless`/`flask`/`msgpack` 等 | 见 venv | embodied runtime / ai2thor 运行期依赖（用 headless opencv 避免 73MB 全量版） |
| AI2-THOR Unity Linux build | ai2thor 自动下载（约 769MB） | headless Linux 渲染与 physics；Xvfb (`:99`, GL 4.5 llvmpipe) |
| 网络代理 | `http://127.0.0.1:17898` | Windows Clash 反向代理（仅下载期使用，未写入仓库） |
未自行下载来路不明的资产包；兼容版本显式 pin，不静默赌最新。

## 32.4 运行事实（本环境）— Real ProcTHOR Runtime: VALIDATED / PASS
- **全量 CPU suite**：`pytest -q` → **356 passed + 21 subtests passed**（新增
  `test_procthor_backend.py` 5 例；含既有 camera/projection/trajectory/embodied/i18n/server）。
- **真实 ProcTHOR-10K runtime（真实 Unity）→ PASS**：
  - 官方数据：`prior`/`allenai/procthor-10k` `val.jsonl.gz`（git-lfs，5,094,681 B），取 house index 0。
  - 数据：`ai2thor==5.0.0` Controller(`scene="Procedural"`) → `CreateHouse(house)` 成功
    （`lastActionSuccess=True`，场景实例化 236 objects，house rooms=7）。
  - `TeleportFull` 到官方 agent pose → 真实 first-person RGB `(256,256,3)`。
  - 初始 agent：position `(11.5, 0.901, 6.5)`，yaw `90`，cameraHorizon `30`。
  - `MoveAhead`：position `(11.5→11.75)`，`lastActionSuccess=True`，RGB md5 改变。
  - `RotateLeft`：yaw `90→0`，RGB 改变。`LookDown`：horizon `30→60`，RGB 改变。
  - 证据 PNG：`outputs/smoke/procthor/{00_reset_firstperson,01_after_move,02_after_rotate,03_after_look,backend_reset,backend_after_rotate}.png`（gitignored，未提交）。
- **SpatialForge `ProcTHORBackend` 真实接入 → PASS**：
  - `EmbodiedEnvironment` 包装真实 event → `AgentObservation`/`AgentState`/`AgentAction`。
  - model-input 泄漏检查：**CLEAN**（不含 target_positions / reachable / teacher_path / position / segmentation）。
  - 真实 target（privileged/debug 可见）：`mug` → `['Mug|surface|6|56','Mug|surface|7|71']`（来自真实 Thor metadata）。
  - Done verifier：目标不可见 → `success=False`（行为正确，基于引擎 authoritative `visible`）。

## 32.5 Known limitations
- Inspector 主训练 tab 为执行配置 UI（此轮未做大规模训练）。
- ProcTHOR-10K val house index 0 为本次真实验证载体；headless 渲染为 Xvfb software GL
  （llvmpipe），单帧偏慢，仅用于 smoke / 少量 step。
- Debug/God map 为 researcher 顶层投影表示，非像素级遮挡真值。


## 32.2 新增模块
- `spatialforge/embodied/`（主训练运行时）
  - `contracts.py` — AgentObservation / AgentAction / AgentState / EpisodeState /
    ObjectSearchTask / EpisodeHistory / HouseInfo（agent 可见 vs researcher/debug 分层）
  - `actions.py`(常量并入 contracts) / `model_input.py`（严格 allowlist + 泄漏防护）
  - `environment.py`（EmbodiedEnvironment 闭环 + Done verifier + 单主泄漏边界）
  - `teacher.py`（GridTeacher 最短路径，privileged）
  - `backends/deterministic.py`（CPU 测试 harness，**非训练源**）
  - `backends/procthor.py`（真实 AI2-THOR/ProcTHOR 后端）
  - `i18n.py`（zh-CN / en / bilingual 统一字典）
- `spatialforge/inspector/embodied_session.py`、`embodied_server.py`、`embodied_web/`
  （index.html / styles.css / app.js）
- `scripts/run_embodied_smoke.py`
- tests：`test_embodied_contracts.py`、`test_i18n.py`、`test_embodied_transition.py`、
  `test_embodied_teacher.py`、`test_embodied_server.py`、`test_procthor_backend.py`

## 32.3 数据隔离（严格）
`model_input` 只允许：RGB、goal/instruction、camera horizon、permitted action/observation
history。禁止进入模型输入：target_object_ids / target_positions / reachable /
teacher_path / world position / segmentation / God View / scene graph。
泄漏防护以 allowlist + forbidden 语义双保险，并有单元测试覆盖。

## 32.4 运行事实（本环境）
- **全量 CPU suite**：`pytest -q` → **351 passed + 21 subtests passed**（含既有稳定
  camera/projection/trajectory/test 与新 embodied/i18n/server 测试）。
- **确定性纵切 smoke**（CPU，无 Unity）→ PASS：house load → episode → MoveAhead/Rotate
  真实改变 first-person RGB 与 agent state → teacher 最短路 → replay → Done → verifier success。
- **真实 ProcTHOR runtime**：**外部 blocker，未能在此会话完成**。
  依据：本 AutoDL 镜像 pip 源实测 ~165 KB/s；`ai2thor==4.3.0` 依赖（含 73.8 MB
  opencv-python）在多分钟内未能完成下载，ai2thor 未装入 Python 3.10 venv；AI2-THOR
  Unity build（>1 GB）与 ProcTHOR-10K 资产同源网络下载在可用带宽下不可行。真实后端代码、
  清晰 blocker 上报（`--procthor`）均已提供，未以 mock 顶替该 smoke。

## 32.5 Known limitations
- Inspector 主训练 tab 为执行配置 UI（此轮未做大规模训练）。
- 真实 ProcTHOR Unity 运行待网络/环境就绪后在 Python 3.10 venv 内用
  `AI2THOR_EXECUTABLE_PATH` 指向 ProcTHOR-capable build 验证。
- Debug/God map 为 researcher 顶层投影表示，非像素级遮挡真值。

---

# 33. REAL ProcTHOR Object Search Teacher Rollout（2026-09-10）

> 承接 §32 vertical slice；本记录属于未提交工作树（用户控制 Git）。真实 runtime 证据全部在本环境复现（非 replay / 非 mock）。

## 33.1 交付
- **真实 Teacher**（`spatialforge/embodied/procthor_teacher.py`）：真实驱动 AI2-THOR
  `Observation_t -> AgentAction -> controller.step() -> Observation_{t+1}`，无预生成 playback。
  用权威 `GetReachablePositions`(0.25m grid) 建导航图 + BFS；privileged `TeleportFull` 预览选
  **suitable visible observation pose**（`visible` engine metadata）；navigate 到达后只在对齐
  heading/camera 且 authoritative `target visible` 成立时才 `Done`。成功与否由 verifier 判定，
  Teacher 不自宣成功。
- **Episode Generation**（`rollout.py` + `records.py`）：house/目标类别/可到达 spawn 采样、
  initial-visible 重采样、`max_steps`/timeout/失败处理、terminal 状态。两个逻辑数据域严格分离：
  `student_training_record`（仅 RGB 引用/goal/permitted history/teacher next action，重新过递归泄漏闸）
  vs `privileged_research_record`（world pose/target id+pos/reachable/teacher plan/authoritative
  visibility/executed trajectory）。model_input strict allowlist 未放宽。
- **后端**：`backends/procthor.py` 增加 `_fetch_reachable`（GetReachablePositions）与 privileged
  `set_agent_pose`（非 student action）。
- **Inspector Episode Inspection**（`embodied_session/server/web`）：新增 Episodes tab —
  列已存 teacher 回合、加载 timeline、点击 step 显示该步帧 + pose/action/visible/target(priv)/
  terminal，researcher-only 标识；`/api/episodes/{list,load}`、`/api/episodes/state`、
  `/api/episodes/frame`。
- 修复 `EmbodiedEnvironment._handle_done` 未更新 `last_raw_action`（Done transition 的 action
  恒为上一动作）的 bug。

## 33.2 真实运行证据（Xvfb :99, GL llvmpipe, ai2thor 5.0.0, val_house_0）
```
[case] mug        success steps=6  spawn_visible=False resample=1  Done while target visible
[case] winebottle success steps=30 spawn_visible=False resample=1  (30-step real nav, non-Mug)
[case] mug(demo)  success steps=7  spawn_visible=False resample=2  (initial-visible resample)
[case] teddybear  failure steps=5  step budget exhausted before Done (timeout)
[case] nonexistent failure steps=0  category has no real instances (invalid)
[case] apple      failure steps=0  no reachable/observable view pose (unreachable)
```
- 每步为真实 `controller.step()`；success 回合末段 `authoritative_target_visible` 由 0→1 后才
  `Done`，verifier 判定 success=True。
- Student Training Record 6/6 `_leak_checked=clean`，JSON 全串扫描不含
  position/target_object_ids/reachable/teacher_path/world_position/segmentation/god_view。
- Inspector 成功 HTTP 加载真实回合（timeline 含 context+Done，末步 Done/visible/terminal=True，
  PNG frame 正常返回）。

## 33.3 全量 CPU suite
`pytest -q` → **368 passed + 21 subtests passed**（新增 records / nav / episode-inspection 测试；
既有 server 测试需在本环境以 `NO_PROXY=127.0.0.1,localhost` 运行以绕过 shell 级 http_proxy）。

---

# 34. GPU-backed ProcTHOR rollout runtime（2026-09-10）

> 承接 §33。将 headless llvmpipe blocker 关闭：AI2-THOR 真实渲染改走 NVIDIA GPU。

## 34.1 ROOT CAUSE（llvmpipe）
- `ai2thor 5.0.0` 的 Linux64 build 经 **X 的 GLX** 渲染（`launch_env` 只设 `DISPLAY`）。
- 此前用 `Xvfb :99` —— Xvfb 是纯软件 X server，不与 NVIDIA 通信 → Mesa GLX 只能回退
  **llvmpipe** 软件渲染。`nvidia-smi`：util 0% / VRAM ~1 MiB / thor 进程高 CPU。
- 结论：不是缺少 GPU lib，而是**缺一个由 NVIDIA driver 承载 GLX 的 X display**。

## 34.2 采用路径（官方、可复现）
`Xorg(nvidia 驱动) + 虚拟 framebuffer` 的虚拟 X display，再让 Unity 用该 DISPLAY 渲染。
官方提示即 `ai2thor-xorg`；但它在多 GPU 宿主上会枚举所有 PCI NVIDIA（含不可访问），故改为
**只针对 `nvidia-smi` 可见 GPU**：
- 需要先 `apt install xserver-xorg-core pciutils`（本实例缺 `Xorg`）。
- `scripts/start_nvidia_xorg.py start --display :0`：从 `nvidia-smi` 解析真实 BusID，
  生成仅含该 GPU 的 xorg.conf（`AllowEmptyInitialConfiguration/Interactive False`），
  后台启动 Xorg，轮询 `glxinfo` 直到渲染器为 NVIDIA；失败则退出非 0（**NO SILENT FALLBACK**）。
- `scripts/bench_gpu_thor.py`：同一 house/workload 的软件 vs GPU 可重复对比。
- `--require-nvidia`（`ProcTHORBackend(require_nvidia=True)` / `ProcthorEpisodeGenerator(...)`）：
  加载 house 前校验 DISPLAY 渲染器为 NVIDIA，否则 `ProcTHORError` 明确失败。

## 34.3 硬件证据
- renderer：`NVIDIA GeForce RTX 4080 SUPER`（OpenGL 4.6.0 NVIDIA 595.58.03，direct=Yes）。
- nvidia-smi 采样：VRAM 23 → **384 MiB**；rollout 期 peak GPU util **~57–60%**；power 上浮。
  thor-Linux64 / Unity 渲染负载真实命中 NVIDIA GPU（非仅"不显示 llvmpipe"）。

## 34.4 性能 before / after（同一 val_house_0 / 同一脚本）
| 指标 | Software (Xvfb llvmpipe) | GPU (NVIDIA :0) |
|---|---|---|
| renderer | llvmpipe | RTX 4080 SUPER |
| house load | ~14.2 s | ~11.7 s |
| mean step (256²) | ~646 ms | ~328 ms |
| steps/sec (256²) | ~1.55 | ~3.07 |
| mean step (512²) | ~714 ms | ~531 ms |
| VRAM / util | 23 MiB / 0% | ~325–384 MiB / ~60% |

单步约 **~2x（256²）/ ~1.3x（512²）** wall-clock 加速；低分辨率下 Unity CPU/IO 开销占比高，
硬件收益主要体现在 util/VRAM 与可扩展性（更高分辨率/并行）。已测量速度提升真实存在。

## 34.5 正确性复验（GPU）
真实 ProcTHOR house → CreateHouse → TeleportFull → GetReachablePositions(1083) →
真实 first-person RGB（每动作 rgb 改变）→ MoveAhead/RotateLeft/RotateRight/LookDown →
authoritative visibility → Done verifier；Teacher Object Search episode（mug）success，
最终 target visible=True，verifier 判定 success=True，7 步。student record `_leak_checked` 不变，
world pose/target id 只在 privileged record。语义/契约/隔离/成功定义未改。

## 34.6 全量 CPU suite
`pytest -q` → **368 passed + 21 subtests passed**（历史 baseline 未破坏）。

---

# 35. Embodied BC Vertical Slice（2026-09-10）— REAL Qwen-controlled Object Search

> 承接 §33/§34。本 Gate：Real ProcTHOR teacher BC dataset → Qwen2.5-VL-3B LoRA BC
> fine-tune → offline action eval → **真实模型闭环** → God View 实时观察。全部为未提交
> 工作树（用户控制 Git）。本记录不抹除/不贬低历史 controlled baseline；G2.0-* 与
> Teacher/Inspector/NVIDIA runtime Gate 保持历史事实。

## 35.1 Maximum Useful Throughput policy（本次起执行）
- 优化目标 = MAX(有效 wall-clock 产出)：environment steps / episodes / samples /
  optimizer updates / model-controlled episodes，不是 MAX(GPU utilization %)。
- worker 数选择以 aggregate useful throughput 平台/下降为停止条件；GPU 高 util 但不是
  唯一目标（见 35.4 sweep 表：w6=8.33 steps/s 峰值；GPU avg util 仅 ~7.6%，
  bottleneck 为 Unity 每步 CPU/IO 与 privileged 规划渲染，非 GPU 像素吞吐）。

## 35.2 数据与隔离
- 官方 ProcTHOR-10K **val.jsonl.gz**（HF allenai/procthor-10k 现 gated/401、AWS S3 403，
  无法获取官方 train split）→ 改用官方 val houses 严格 **house-level disjoint split**：
  train=40 houses / val=3 houses（house_0005..house_0044 等 vs val_house_0/1/2），
  provenance 显式记录；train/val 无 house 交集。
- renderer=NVIDIA GLX(Xorg :0) quality Low；resolution 256×256；Teacher=§33 真实
  ThorObjectSearchTeacher；action vocabulary=8（MoveAhead/RotateLeft/RotateRight/
  LookUp/LookDown/Crouch/Stand/Done，未扩 interaction）。
- Student record schema 不变（`student_training_record.v1`）：
  - 新增 `Done` target 语义过滤：teacher 偶发在 step-budget 边缘发出“Done 但目标不可见”
    （verifier 判 false）；此类行的 Done target 不进入 BC 数据（防止教坏 Done 语义），
    其余 teacher truth 未改动、未伪造 action。
  - model-input strict allowlist / recursive forbidden-term 保持；FORBIDDEN_TERMS 增补
    plain-English 短语（target position / object id / world position / coordinates 等）。
- dataset：train **6400** rows / 123 episodes / 35 houses（house-stratified cap）；
  val **1500** rows / 36 episodes / 3 houses。leakage scan = **0**。

## 35.3 BC 训练（新 objective，非历史 static-QA 对照）
- 模型 Qwen/Qwen2.5-VL-3B-Instruct（bf16，本地 /root/autodl-tmp/models/…）。
- LoRA r=8 alpha=16 dropout=0.05，语言侧 252 modules / 14,966,784 params，vision frozen。
- profile：historical `max_performance` MB4×ACC2（effective 8）作为起点实跑；
  Frozen Vision Feature Cache 在本任务**合法复用**（cache key=图像内容 hash；
  Qwen2.5-VL vision tower 输出与文本无关；单帧 per-sample 契约不变；无 stale 复用）。
- 6400 samples / 800 optimizer steps / wall 约 19-20 min（含 ~9 min 串行 cache prep）
  / samples/sec 波动 3.7→6.2（cache prep 段低；训练段 GPU ~84-85%）。

## 35.4 实测数据与结论（真实 NVIDIA GPU）
- **concurrency sweep**（12 fixed jobs×256²，n=1,2,4,6,8，全程 0 crash）：
  n=1 0.43 steps/s（受 pathological 规划负载+watchdog 约束）→ n=2 2.50 → n=4 5.79 →
  **n=6 8.33 steps/s（选型）** → n=8 8.02（平台）。GPU util avg 0.3%→7.6%；
  VRAM avg ~1.0→3.1 GB；CPU avg 5-12%（Unity 单步 CPU/IO 为主瓶颈）。
- **dataset generation**（w6 实跑）：train 89+11+40 jobs ≈ 30 min；val 12+24 jobs ≈ 4 min；
  val GPU util avg ~9.9%、p95 51%、VRAM avg 2.5-5.2 GB。
- **closed-loop**：9/9 model-controlled 真实 episodes（val_house_0/1/2，不在训练集），
  720 env steps，0 crash，0 invalid，**success=0**（全部 80-step budget exhausted，
  模型从不输出 Done、几乎只 MoveAhead（719/720）→ 与离线 eval 一致：类不均衡导致
  Done 学习信号过弱；无 false-Done；诚实报告，不改数据/不放宽）。
- decision latency 实测量级：单次 inference ~0.5-1.0 s 级（服务器端 greedy decode +
  ViT）；含 env step 的闭环步骤 ~1.5-2.5 s。

## 35.5 God View（Live Embodied Inspector，researcher-only）
- episodes_root 指向 model_rollout 目录；Episodes tab 列出 model-controlled 回合
  （header：model_controlled / terminal_reason / decision/invalid counts），REPLAY
  timeline（frames+pose+visible+Done+terminal，privileged），decisions 表
  （raw output / parsed action / executed / invalid / latency / frame ref）。
- **LIVE**：episode 运行中原子写 `live_state.json`+逐帧 PNG；list/load 返回
  in_progress payload（当前 step、model action、raw、latency、visible、live frame）；
  前端 2s 轮询。实测轮询 3 次 step 17→23→30→36 增长。全程 read-only、非阻塞、
  不触碰模型输入（student 侧隔离契约未变）。

## 35.6 工程事实 / 已知问题
- ai2thor 每次 CreateHouse 的对象实例集**存在逐次变动**（同 house 文件不同 load 出现
  不同 mug/pen 实例），导致 privileged 规划（view-pose/spawn teleport renders）
  耗时剧烈波动（病理 episode 可 >600 s）→ 引入：worker per-job watchdog（monotonic
  clock，免疫 NTP 跳变）、instance 规划 probe cap、setup timeout（150 s）与确定性
  失败上报；这些是吞吐工程，不改变 verifier/teacher truth/student 语义。
- 本机 http_proxy 会劫持 urllib 到 127.0.0.1 → localhost 流量显式 ProxyHandler({})。
- 403 passed + 21 subtests（CPU 全量，NO_PROXY=127.0.0.1,localhost）。
