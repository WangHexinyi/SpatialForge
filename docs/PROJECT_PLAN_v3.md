# SpatialForge 项目计划书 v3.0

> **文档定位**：跨 AI 项目记忆、当前状态快照、研究路线与工程执行指南。  
> **不是不可修改的纲领**：其中任何架构、门控、实验和实现细节都可以基于新证据讨论、调整、替换。  
> **真正需要长期保持一致的内容**：用户原始研究愿景、已经完成并可复现的实验事实、当前 Git 状态、已通过门控、核心边界条件。  
> **最近更新**：2026-09-04  
> **建议仓库路径**：`docs/PROJECT_PLAN_v3.md`  
> **公开仓库注意**：本文不记录任何私有 SSH 地址、密钥、令牌、API Key、账号凭据或云实例敏感信息。

---

# 0. 新会话 / 新 AI 快速拉起

任何新 AI 接手 SpatialForge 时，先读取本文件，并以以下状态为准：

```text
Project: SpatialForge
Stage: v2.0 multi-view / spatial experience foundation
Current development branch: feat/v2.0-camera-geometry
Current HEAD after G2.0-B commit: 8ced8de

Completed:
- v1 diagnostic + LoRA baseline: CLOSED
- G2.0-A Camera Geometry Core: PASS
  commit: ddf1594
- G2.0-B Multi-view Scene Renderer: PASS
  commit: 8ced8de

Verified current test state after G2.0-B:
- 24 tests passed
- Blender 4-view smoke render passed
- manifest schema passed
- broken `scene_scene_000_*` naming eliminated

NEXT:
- G2.0-B.1 Camera Sampling System
  canonical 6 / 14 / 26 viewpoints + training jitter / continuous sampling
- then G2.0-C View-Conditioned Spatial Truth Engine
```

新 AI **不要重做** G2.0-A / G2.0-B，不要把固定多视角误解为最终产品形态。固定相机只属于 calibration / diagnostics / curriculum generation 层，最终研究目标仍是具身第一视角 Agent + world model + active observation。

---

# 1. 项目身份

- **项目名**：SpatialForge（空间锻造台）
- **形态**：VLM 空间推理诊断、训练经验生成、具身观察与认知世界模型训练工作台
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

概念架构：

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

# 5. v2 总体架构

## 5.1 核心原则：World State ≠ Observation

```text
                 World State
                     │
                     │ Observe(camera/agent state)
                     ▼
                 Observation
```

同一个 World State 可以产生多个 Observation。

## 5.2 当前已建立架构

```text
SceneState
  │
  ├── SceneObject
  │
  └──────────────┐
                 │
             CameraPose
                 │
                 ▼
             Renderer
                 │
                 ▼
          ObservationMetadata
                 │
                 ▼
               Image
```

## 5.3 后续完整架构

```text
                      SpatialForge
                           │
         ┌─────────────────┴─────────────────┐
         │                                   │
   Environment State                   Human God View
   (authoritative world)                (privileged)
         │                                   │
         │ Sensor / action API               │
         ▼                                   │
    Embodied Agent                           │
         │                                   │
 First-Person Camera                         │
         │                                   │
         ▼                                   │
    Observation_t                            │
         │                                   │
         ├── Brain / VLM                     │
         ├── Spatial memory                  │
         └── World model                     │
                 │                           │
                 ▼                           │
              Action_t                       │
                 │                           │
                 └──────────> Environment ───┘
```

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

这是最终 Agent 的真实第一视角传感器。

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

只供人类监督者 / evaluator 使用。

可显示：

- 完整 3D 世界
- Agent 位置与朝向
- Camera frustum
- 运动轨迹
- visible / occluded objects
- AI 选择的下一观察点
- hidden ground truth
- observation history
- confidence / entropy

**God View 永远不能暴露给 Agent。**

原则：

> Privileged evaluator state ≠ agent observation state.

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

不允许永远固定在 canonical camera 上。

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

# 9. v2.0 后续路线图

## G2.0-B.1 — Camera Sampling System（NEXT）

目标：把当前 4-view smoke camera 升级成可复用采样系统。

必须支持：

- 4 cardinal（保留作为回归）
- 6-axis
- 14-view subset
- 26-view canonical lattice
- seeded jitter
- continuous CameraPose sampling

验收核心：

1. canonical view IDs / order deterministic
2. 6/14/26 数量正确
3. 方向不重复
4. 所有相机目标与 radius 规则明确
5. seeded jitter 可复现
6. jitter 后不退化成固定离散 viewpoint classification
7. CPU tests 不依赖 Blender
8. 4-view API 不删除，保护 G2.0-B 回归

---

## G2.0-C — View-Conditioned Spatial Truth Engine

这是 v2.0 最关键的“空间真值”层。

输入：

```text
SceneState + CameraPose
```

输出：

```text
camera-relative spatial truth
```

初始关系：

- left / right
- above / below
- front / behind
- near / far
- metric distance
- camera depth
- 可扩展 visibility / occlusion

关键原则：

> 不能再用 world X/Y hard-code 直接代替视觉关系。

必须由 CameraPose + geometry 推导。

核心反事实验收：

```text
same SceneState
same objects
different CameraPose
→ viewpoint-dependent relation changes correctly
```

同时：

```text
physical size
metric object-object distance
object identity
```

等世界真值不能错误随视角改变。

---

## G2.0-D — Multi-view QA Curriculum

在真值引擎稳定后，再生成自然语言任务。

### 单视图视角条件 QA

例如：

```text
From this view, is the red cube left of the blue sphere?
```

### 配对视角 QA

```text
View A → relation X
View B → relation Y
```

训练模型显式学习 viewpoint transformation。

### 课程类型

- canonical views
- jitter views
- hard-angle views
- symmetry traps
- near/far counterfactuals
- large-far vs small-near
- occlusion-aware questions（后续）

---

## G2.0-E — Controlled Training + Transfer Evaluation

目标：回答第一个 v2 科学问题：

> 多视角 / 视角条件训练是否真正改善 orientation，并跨域迁移？

对照：

```text
Baseline A: v1 single-view synthetic
Baseline B: no new LoRA
Treatment C: canonical multi-view
Treatment D: canonical + jitter
```

评测：

- VSR overall
- VSR orientation
- synthetic orientation
- near_far
- count
- 其他维度稳定性

优先观察：

```text
orientation ↑
其他维度不显著退化
```

如果仅 synthetic orientation 上升而 VSR 无迁移，应明确记录为 domain-specific fitting，而不能宣称获得通用 3D reasoning。

---

# 10. v2.1 — Embodied Explorer

v2.1 是从“系统选择 Camera”到“Agent 通过行动改变 Camera”的关键跃迁。

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

# 11. v2.1+ — Action-Conditioned World Model

对应核心创新 I1。

不要求像素级复原作为唯一目标。

优先考虑 latent / representation prediction：

```text
z_t + action_t -> z_t+1
```

其中 `z_t` 应表达：

- object layout
- relative geometry
- visibility belief
- spatial memory
- camera/agent state

目标是让模型形成：

> “如果我这样移动/转头，下一刻空间观测应如何变化”的隐空间直觉。

这对应“认知小脑 / System 1”方向。

---

# 12. v2.2 — Active Observation

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

# 13. v2.3 — Interactive Object Search / Embodied QA

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

# 14. God View / Human Evaluation Workbench

SpatialForge 最终应该提供一个真正可视化的“上帝视角工作台”。

## 14.1 目标界面

```text
┌──────────────────────────────────────────────────────┐
│                     GOD VIEW                         │
│                                                      │
│  objects / rooms / containers                        │
│  Agent ● → trajectory                                │
│  camera frustum                                      │
│  selected next viewpoint                             │
│                                                      │
├───────────────────────┬──────────────────────────────┤
│   AGENT FIRST PERSON  │       REASONING STATE        │
│                       │                              │
│   current RGB/view    │ Question / Goal             │
│                       │ Confidence / Entropy         │
│                       │ Next action                  │
├───────────────────────┴──────────────────────────────┤
│ timeline: O0 → A0 → O1 → A1 → O2 ...                │
└──────────────────────────────────────────────────────┘
```

## 14.2 God View 显示内容

- 完整场景布局
- Agent position / heading
- current camera frustum
- trajectory
- next selected action / viewpoint
- visible objects
- occluded objects
- hidden target location
- observation history
- model answer
- confidence / uncertainty
- optional ground truth overlay

## 14.3 判卷策略

### 自动判卷

适合可由引擎精确确定的任务：

- left/right
- depth
- distance
- visibility
- occlusion
- collision
- object identity
- target reached

### 人工判卷

适合高层行为：

- 搜索策略是否合理
- 是否真正使用历史记忆
- 是否存在碰巧猜中
- 是否选择了高信息量视角
- 是否绕路过多
- 是否形成稳定的空间解释

因此最终评估不是“全人工”也不是“全自动”，而是：

> **自动 Ground Truth + Human Supervisor**。

---

# 15. 尺寸—距离混淆实验（I5）

这是 v1 发现延伸出的重要可证伪假设。

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

# 16. 数据与 Schema 方向

当前已有：

```text
CameraPose
SceneObject
SceneState
ObservationMetadata
```

后续建议逐步加入：

```text
SpatialRelationRecord
CameraSampleSet
AgentState
Action
TrajectoryStep
Trajectory
BeliefState (research-facing, optional)
EvaluationRecord
```

## 16.1 原则

- world truth 与 observation 分离
- privileged state 与 agent-visible state 分离
- authoritative pose 只有一个
- 不同时维护互相可能冲突的 yaw/pitch/Euler/target 多套真值
- JSON 可序列化
- deterministic seed
- scene-level split 防泄漏

---

# 17. 渲染与引擎策略

当前 v2 底层使用 Blender 做受控 synthetic scene 渲染。

长期不把 SpatialForge 锁死在 Blender。

概念上保持：

```text
Environment / Scene semantics
        ↓
Renderer Adapter
        ├── Blender
        ├── future simulator
        └── real-world capture adapter (future)
```

但不要过早建设庞大的插件框架。

先完成科学闭环，再抽象。

---

# 18. 真实感与资产路线

真实感不是当前第一优先级。

顺序：

1. controlled primitives
2. multi-view / embodied consistency
3. scientific signal confirmed
4. then asset diversity / textures / real-looking scenes

后续候选：

- Objaverse
- AI2-THOR assets
- Habitat-compatible scenes
- procedural rooms

目的不是“更漂亮”，而是降低 synthetic-to-real / synthetic-to-benchmark domain gap。

---

# 19. 计算资源策略

## 19.1 本地 Windows

角色：**主开发环境**。

负责：

- VS Code / OpenCode CLI
- Git
- CPU tests
- small Blender smoke renders
- schema / geometry / QA development
- local review

## 19.2 AMD Radeon Cloud

当前结论：不适合作为日常开发工作站。

原因：

- 实例启动摩擦
- Jupyter connection instability
- GitHub HTTPS proxy / CA friction

后续仅作为可选计算节点：

- large rendering
- ROCm training
- batch inference

使用原则：

> 云服务器是计算节点，不是工作站。

## 19.3 CUDA Cloud / AutoDL 类节点

作为稳定 fallback：

- LoRA
- evaluation
- model inference

## 19.4 GPU 时间纪律

不要用昂贵 GPU 时间做：

- 文本编辑
- CPU geometry tests
- Git 操作
- 普通单元测试

只有 pipeline 已经稳定时才启动大规模计算。

---

# 20. ROCm 纪律

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

# 22. AI 协作与开发流程（当前实际工作流）

原“一切用户手动执行”策略已更新。

## 22.1 角色划分

### ChatGPT

负责：

- 研究路线
- 架构设计
- 门控
- Prompt 设计
- 代码结果复审
- 实验解释
- 项目记忆维护

### OpenCode CLI

负责：

- 实际代码修改
- 测试
- smoke checks
- 小范围修复
- git diff / status 检查

### 用户

负责：

- 最终决策
- 亲自 Git commit
- push / merge 节奏
- 实验方向裁决

## 22.2 OpenCode 默认模型策略

当前基于实际体验：

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

# 24. 科学实验纪律

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

## 24.4 失败结果也保留

例如：

- multi-view improves synthetic orientation but not VSR
- jitter hurts stability
- larger model worsens near/far

都属于科学证据，不应“优化掉”。

---

# 25. 主要创新点（当前版本）

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

---

# 26. 产品化方向

SpatialForge 不应最终只是脚本集合。

未来产品壳可包括：

## CLI

统一入口：

```text
spatialforge scene generate
spatialforge render multiview
spatialforge curriculum build
spatialforge train
spatialforge eval
spatialforge explorer run
```

## God View Workbench

建议最终优先级高于单纯 Gradio 指标面板。

功能：

- real-time 3D scene
- Agent first-person feed
- trajectory
- camera frustum
- selected next action
- ground truth toggle
- evaluation timeline
- manual annotation / judgement

## Experiment Dashboard

显示：

- dimension-wise scores
- orientation curves
- near/far confusion
- viewpoint-conditioned failures
- trajectory success / efficiency

---

# 27. 近期路线（执行顺序）

## NEXT 1 — G2.0-B.1 Camera Sampling System

不要直接上 Agent。

先把 diagnostic / curriculum camera sampling 做成正确的连续空间工具：

```text
4 cardinal
6 axis
14 subset
26 lattice
jitter
continuous sample
```

## NEXT 2 — G2.0-C View-Conditioned Spatial Truth

让任意 CameraPose 都能得到正确 relation labels。

## NEXT 3 — G2.0-D QA Curriculum

把 relation truth 转成视角条件训练数据。

## NEXT 4 — G2.0-E First Multi-view Training Experiment

回答：orientation 能不能移动？

## NEXT 5 — v2.1 Embodied Explorer

把 CameraPose 从系统指定转为 Agent action transition。

## NEXT 6 — World Model Objective

动作条件 latent prediction。

## NEXT 7 — v2.2 Active Observation

信息增益驱动的下一步观察。

## NEXT 8 — v2.3 Interactive Object Search

真正实现“找东西 + 遮挡 + 容器交互”。

---

# 28. 当前暂停点（2026-09-04）

今天工作在**项目规划完成**处暂停。

正式状态：

```text
Branch:
feat/v2.0-camera-geometry

Latest commits:
8ced8de feat(g2.0-b): add multiview scene rendering pipeline
  parent:
ddf1594 feat(v2.0): add camera geometry core

G2.0-A: PASS
G2.0-B: PASS

Verified after G2.0-B:
- 24 tests passed
- Blender 4-view smoke render passed
- 4 corrected images generated
- manifest generated under outputs/synth/manifests/
- complete CameraPose metadata
- relative portable image paths
- no broken scene_scene_000 artifacts

NEXT IMPLEMENTATION:
G2.0-B.1 Camera Sampling System
```

当前不要做：

- 不要重做 v1
- 不要重写 G2.0-A / B
- 不要把 4 fixed cameras 当最终训练方案
- 不要直接引入复杂 Agent before camera/truth layers are stable
- 不要把 God View 暴露给 Agent

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

> **SpatialForge 是一个面向 VLM 空间推理与认知世界模型研究的 3D 训练与诊断工作台：它从可控多视角几何出发，逐步走向具身第一视角 Agent、动作条件世界预测、主动观察、空间记忆与交互式目标搜索，同时为人类研究者提供不泄露给 Agent 的 God View 监督与评估界面。**

---

# 31. 文档维护规则

本文件是活文档。

每完成一个关键门控后更新：

- §0 快速拉起
- §8 已完成门控
- §27 近期路线
- §28 当前暂停点
- 相关架构/决策章节

如果路线发生变化：

> 允许修改，不需要为了“保持旧计划正确”而硬撑。

但已经完成的 commit / 实验事实必须保留历史记录，不能事后改写。