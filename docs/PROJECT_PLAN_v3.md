# SpatialForge 项目计划书 v3.0

> **文档定位**：跨 AI 项目记忆、当前状态快照、研究路线与工程执行指南。  
> **不是不可修改的纲领**：其中任何架构、门控、实验和实现细节都可以基于新证据讨论、调整、替换。  
> **真正需要长期保持一致的内容**：用户原始研究愿景、已经完成并可复现的实验事实、当前 Git 状态、已通过门控、核心边界条件。  
> **最近更新**：2026-09-07
> **建议仓库路径**：`docs/PROJECT_PLAN_v3.md`  
> **公开仓库注意**：本文不记录任何私有 SSH 地址、密钥、令牌、API Key、账号凭据或云实例敏感信息。

---

# 0. 新会话 / 新 AI 快速拉起

任何新 AI 接手 SpatialForge 时，先读取本文件，并以以下状态为准：

```text
Project: SpatialForge
Stage: v2.0 multi-view / spatial experience foundation
Current development branch: feat/g2.0-d-qa-curriculum
Latest implementation commit: cf2fb52 feat(g2.0-e1): add controlled multiview experiment data foundation

Completed:
- v1 diagnostic + LoRA baseline: CLOSED
- G2.0-A Camera Geometry Core: PASS
  commit: ddf1594
- G2.0-B Multi-view Scene Renderer: PASS
  commit: 8ced8de
- G2.0-B.1 Camera Sampling System: PASS
  commit: 6d08fcf
- G2.0-C View-Conditioned Spatial Truth Engine: PASS
  commit: 19e3526
- G2.0-D Multi-view QA Curriculum: PASS
  commit: 6823f29
- G2.0-E1 Controlled Multiview Experiment Data & Rendering Foundation: PASS / CLOSED
  commit: cf2fb52

Verified after G2.0-B (historical, preserved):
- 24 tests passed at that gate
- Blender 4-view smoke render passed
- 4 corrected images generated
- manifest schema passed
- broken `scene_scene_000_*` naming eliminated

Verified after G2.0-B.1:
- total CPU unittest suite: 62 tests passed
- G2.0-B 4-view regression preserved (API unchanged, no existing source files modified)
- canonical 6-axis / 14-view / 26-view sampling added
- seeded jitter + continuous uniform-sphere sampling added
- G2.0-B.1 itself does not require Blender

Verified after G2.0-C:
- total CPU unittest suite: 98 tests passed
- 62 pre-existing tests preserved
- 36 new G2.0-C tests
- no existing source files modified
- new files:
  - spatialforge/environment/relation.py
  - tests/test_spatial_truth_engine.py
- G2.0-C itself does not require Blender

Verified after G2.0-D:
- total CPU unittest suite: 141 tests passed
- 98 pre-existing tests preserved
- 43 new G2.0-D tests
- no existing source files modified
- new files:
  - spatialforge/environment/qa.py
  - tests/test_qa_curriculum.py
- G2.0-D itself does not require Blender

Verified after G2.0-E1:
- total CPU unittest suite: 169 tests passed
- 141 pre-existing tests preserved
- 28 new G2.0-E1 tests
- no existing source files modified
- new files:
  - scripts/export_v1_scenes.py
  - scripts/render_curriculum.py
  - spatialforge/experiment/__init__.py
  - spatialforge/experiment/dataset.py
  - tests/test_experiment_dataset.py
- Blender 4.2.0 Workbench OBJECT shading installed and verified on AutoDL
- 100 historical v1 scenes recovered (train: scene_000-079, holdout: scene_080-099)
- 800 images rendered (100 scenes x 8 PRIMARY_D views) into outputs/experiments/g2.0-e/rendered/
- 78 eligible training scenes covered identically across Groups A, C, D (scene_007 and scene_056 excluded due to visual ambiguity policy)
- common training budget N = 1552 per group, 388 per family
- exact 194 / 194 (50.0% / 50.0%) directional label balance across all families in A, C, D
- operand-order normalization eliminates south-view directional priors without altering G2.0-D truth
- deterministic scene-aware stratified sampling eliminates early-scene truncation confound
- zero source_sample_id duplicates within any dataset
- zero cross-group presentation-order disagreements on shared source samples
- 6928 / 6928 dataset records independently verified against raw 3D geometric camera truth (0 mismatches)
- bit-for-bit build determinism verified (identical SHA-256 hashes across repeated generation)
- holdout S1 (cardinal, N=1136) and S2 (jitter, N=1136) materialized and disjoint from train

NEXT:
- G2.0-E3 Formal Controlled Training & Transfer Evaluation (NOT started)
```

新 AI **不要重做** G2.0-A / G2.0-B / G2.0-B.1 / G2.0-C / G2.0-D / G2.0-E1，不要把固定多视角误解为最终产品形态。固定相机只属于 calibration / diagnostics / curriculum generation 层，最终研究目标仍是具身第一视角 Agent + world model + active observation。

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
- 历史 v1 程序化场景生成逻辑被完整复原并实现为纯 Python 脚本：[`scripts/export_v1_scenes.py`](file:///root/autodl-tmp/SpatialForge/scripts/export_v1_scenes.py)。
- 可严格重构全部 100 个历史场景（`scene_000` ... `scene_099`）。
- 生成结果经严格比对完全吻合：
  - `examples/scenes/scene_000.json`（语义与浮点坐标严格一致）
  - `tests/fixtures/v1_front_back_reference.json`（100 个场景全部物体坐标对齐至数值精度）
- 场景划分保持历史兼容：
  - `train`: `scene_000` ... `scene_079`（80 场景）
  - `holdout`: `scene_080` ... `scene_099`（20 场景）

#### 2. 独立 G2.0-E 课程渲染器（Dedicated Curriculum Renderer）
- 新增专用渲染入口：[`scripts/render_curriculum.py`](file:///root/autodl-tmp/SpatialForge/scripts/render_curriculum.py)。
- 已关闭的 G2.0-B 渲染器 [`scripts/render_multiview.py`](file:///root/autodl-tmp/SpatialForge/scripts/render_multiview.py) 保持完全不变（作为回归基准）。
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
- **Group C**：水平四向正交多视点处理组（`south`, `east`, `north`, `west`，单图 Single-view QA）。
- **Group D**：水平正交 + 有界扰动多视点处理组（4 正交视角 + 4 确定性扰动视角，单图 Single-view QA）。
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
- 彻底杜绝了模型利用标签先验捷径作答的因果混淆。

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
  - 结论：不存在剩余因果混淆。

#### 12. 冻结的评测 Holdout 集（Frozen Holdout Sets）
- **Holdout S1 (`holdout_s1_cardinal.jsonl`, $N = 1136$)**：未见场景（`scene_080`–`scene_099`）+ 水平正交视点。用于测试熟悉视角家族下的未见场景泛化。
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
- 单元测试状态：**169 / 169 PASS**（0 regressions）。
- 提交 Commit：`cf2fb52`。
- 终审结果：**APPROVE E1 COMMIT**，正式关闭 G2.0-E1。

---

# 9. v2.0 后续路线图

## G2.0-B.1 — Camera Sampling System ✅ DONE

> 已完成并通过门控，见 §8.3。以下保留为历史计划记录。

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

## G2.0-C — View-Conditioned Spatial Truth Engine ✅ DONE

> 已完成并通过门控，见 §8.4。以下保留为历史计划记录。

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

## G2.0-D — Multi-view QA Curriculum ✅ DONE

> 已完成并通过门控（commit 6823f29），见 §8.5。以下保留为历史计划记录。

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
- occlusion-aware questions（后续，不在本 gate 范围）

---

## G2.0-E1 — Controlled Multiview Experiment Data & Rendering Foundation ✅ DONE

> 已完成并通过门控（commit cf2fb52），详细实现与科学事实见 §8.6。

交付成果：
- 100 场景历史程序化生成器复原（[`scripts/export_v1_scenes.py`](file:///root/autodl-tmp/SpatialForge/scripts/export_v1_scenes.py)）
- Blender 4.2.0 Workbench 800 张图像隔离渲染（[`scripts/render_curriculum.py`](file:///root/autodl-tmp/SpatialForge/scripts/render_curriculum.py)）
- 78 场景 A/C/D 统一平衡数据集构建（$N = 1552$，194/194 标签对称平衡，操作数顺序归一化，消除场景截断混淆）
- 冻结 Holdout S1（正交，$N=1136$）与 S2（扰动，$N=1136$）评测集
- 169 单元测试全绿，底层 3D 几何真值 0 误差

---

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
  - 新增/修改代码：[`scripts/train_smoke.py`](file:///root/autodl-tmp/SpatialForge/scripts/train_smoke.py), [`spatialforge/experiment/training.py`](file:///root/autodl-tmp/SpatialForge/spatialforge/experiment/training.py), [`tests/test_training_harness.py`](file:///root/autodl-tmp/SpatialForge/tests/test_training_harness.py)。
  - 测试套件：188 / 188 PASS。
  - 实施 Commit：`ada1179`。

### 关键 Blocker 复盘与修复记录（Recovered Blocker & Fix）
1. **F1 视觉塔 LoRA 误绑定**：初版使用未限定层级的后缀匹配 `target_modules = ["q_proj", ...]`，意外命中了 `model.visual.blocks.*.mlp` 中的 96 个视觉模块（3,609,600 参数）。在审查中被拦截，改用显式语言模型目标白名单（`get_language_model_target_modules()`），最终状态严格对齐为 252 LM / 0 视觉。
2. **F2 梯度累积语义缺失**：初版烟测直接单步 backward 即 step，缺乏 formal 累积语义。在审查中被拦截，重构为微批次梯度缩放（`loss / 8`）、每 8 步执行 1 次 optimizer+scheduler step 的可复用训练循环，并补充启用梯度检查点与线性调度器。

### E3 待办与非阻塞积压事项（Non-blocking Backlog for E3）
- 正式评估前加固 directional / yes-no 解析器（避免 `"not left"` 等被子串逻辑误判）。
- 将单元测试中的局部张量切片测试重构为直接覆盖生产路径 `build_training_tensors`。
- 明确认知：S2 为同扰动过程泛化，而非连续球面上强连续 OOD。
- 相机 FOV 元数据 vs Blender 实际 ~54.4° 差异留待未来投影/可见性工作统一对齐。

---

## G2.0-E3 — Formal Controlled Training & Transfer Evaluation 🎯 NEXT

目标：回答第一个 v2 科学问题：

> 多视角 / 视角条件训练是否真正改善 orientation，并跨域迁移？

### 经济高效的执行顺序（E3 启动原则）
1. **加固正式评估器与解析器**：加固 directional / yes-no 答案抽取与词边界保护。
2. **冻结实验配置**：严格遵循 E2 冻结语义（lr=1e-4, accum=8, 1 epoch, linear scheduler, warmup=0）。
3. **运行 Baseline B 零样本评测**：评估未经微调的基模在 S1, S2, VSR 上的基线能力。
4. **单 seed 工程验证（seed=42）**：先跑 Control A, Treatment C, Treatment D 的单个工程 seed，验证训练与评测流水线耗时与产出。
5. **指标与资源审查**：验证训练平稳收敛、checkpoint 保存与评估指标产出。
6. **多 seed 完整扩展**：在流水线完全确认可靠后，方可扩展执行 seed=123 与 seed=456，完成统计显著性矩阵。

对照设计：

```text
Baseline B: pretrained baseline (no fine-tuning)
Control A: fixed south view (single-image QA)
Treatment C: 4 canonical horizontal views (south, east, north, west)
Treatment D: 4 canonical + 4 bounded jitter views
```

评测矩阵：

- 合成 Holdout S1（未见场景正交视角）
- 合成 Holdout S2（未见场景扰动视角）
- VSR Direct Transfer (`left_right`, `front_back`)
- VSR Broad Orientation（历史宽泛定向关系）
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
- SpatialForge 本身保持 **计算后端无关**（CUDA / ROCm 都只是执行后端）；这**不代表永久禁止** ROCm / AMD 支持。
- 若未来 AMD 节点可用并满足资源需求，可重新评估其作为可选计算节点。

## 19.3 AutoDL（CUDA Cloud）— 当前默认远程执行环境

- AutoDL 是当前主要 / 默认的远程执行环境（Linux / CUDA）。
- OpenCode CLI 直接运行在 AutoDL 实例上。
- AutoDL 当前承担 CPU / code 工作（CPU tests、代码开发等）。
- AutoDL 计划用于 CUDA / GPU workloads（LoRA / evaluation / inference）。
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

# 22. AI 协作与开发流程（当前实际工作流）

原“一切用户手动执行”策略已更新。

当前执行环境：

- OpenCode CLI 直接运行在 AutoDL 实例（远程 Linux / CUDA）上
- 当前 OpenCode 版本：1.18.29
- 当前执行模型：DeepSeek V4 Flash · low
- 本地机器是用户侧控制 / 协调端
- GitHub 仍是代码 / 文档长期真源
- 用户保留最终 commit / push 决策

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

> G2.0-A / G2.0-B / G2.0-B.1 / G2.0-C / G2.0-D / G2.0-E1 / G2.0-E2 均已通过门控（G2.0-E2 见 §28 章节）。以下是当前执行顺序。

## NEXT 1 — G2.0-E3 Formal Controlled Training & Transfer Evaluation

> G2.0-E2 Controlled Training Harness & Pipeline Verification 已完成并通过门控（commit ada1179，见 §28 章节）。
> G2.0-E3 尚未开始。

回答第一个 v2 科学问题：多视角 / 视角条件训练是否真正改善 orientation，并跨域迁移？
对照：Baseline B (zero-shot) vs Control A (south) vs Treatment C (cardinal) vs Treatment D (cardinal+jitter)。
评测：S1 / S2 合成泛化 + VSR Direct (`left_right`, `front_back`) + VSR Broad Orientation。
执行顺序遵循经济原则：1. 加固评估器/解析器 -> 2. 冻结配置 -> 3. Baseline B -> 4. 单 seed=42 工程验证 -> 5. 审查指标/显存 -> 6. 扩展至 seeds 123/456。

## NEXT 2 — v2.1 Embodied Explorer

把 CameraPose 从系统指定转为 Agent action transition。

## NEXT 3 — World Model Objective

动作条件 latent prediction。

## NEXT 4 — v2.2 Active Observation

信息增益驱动的下一步观察。

## NEXT 5 — v2.3 Interactive Object Search

真正实现“找东西 + 遮挡 + 容器交互”。

---

# 28. 当前暂停点（2026-09-07）

今天工作在 **G2.0-E2 门控完成（commit ada1179）** 处暂停。

正式状态：

```text
Branch:
feat/g2.0-d-qa-curriculum

Latest implementation commit:
ada1179 feat(g2.0-e2): add controlled training harness

G2.0-A: PASS
G2.0-B: PASS
G2.0-B.1: PASS
G2.0-C: PASS
G2.0-D: PASS
G2.0-E1: PASS / CLOSED
G2.0-E2: PASS / CLOSED

Verified after G2.0-B (historical, preserved):
- 24 tests passed at that gate
- Blender 4-view smoke render passed
- 4 corrected images generated
- manifest schema passed
- broken `scene_scene_000_*` naming eliminated

Verified after G2.0-B.1:
- total CPU unittest suite: 62 tests passed
- G2.0-B 4-view regression preserved
- 38 new tests covering camera intrinsic/extrinsic geometry, frustum, 3D math
- complete clean camera geometry layer implemented
- numerical parity with Blender Camera coordinates verified

Verified after G2.0-C:
- total CPU unittest suite: 104 tests passed
- 62 pre-existing tests preserved
- 42 new tests covering spatial relations, frame transforms, and edge cases
- single source of spatial truth operational
- zero coordinate frame confusion: all relations computed in camera space
- strictly deterministic: no randomness, no heuristics, no Python hash()

Verified after G2.0-D:
- total CPU unittest suite: 141 tests passed
- 104 pre-existing tests preserved
- 37 new tests covering curriculum generation, balance, and invariants
- zero cross-view label leakage: queries use view-invariant object descriptions
- strictly deterministic QA generation across all 4 canonical views
- clean decoupling: question generation operates on geometric truth objects,
  no Python hash(), no geometry recomputation in QA layer
- G2.0-D itself does not require Blender

Verified after G2.0-E1:
- total CPU unittest suite: 169 tests passed
- 141 pre-existing tests preserved
- 28 new G2.0-E1 tests
- no existing source files modified
- new files:
  - scripts/export_v1_scenes.py
  - scripts/render_curriculum.py
  - spatialforge/experiment/__init__.py
  - spatialforge/experiment/dataset.py
  - tests/test_experiment_dataset.py
- 100 historical v1 scenes recovered (train: scene_000-079, holdout: scene_080-099)
- Blender 4.2.0 Workbench OBJECT shading verified on AutoDL; 800 images rendered
- A/C/D common training budget N = 1552 per group, 388 per family
- 78 eligible training scenes covered by all A/C/D groups (007 & 056 excluded by visual ambiguity policy)
- exact 194 / 194 directional label balance across all families in A, C, D
- operand-order normalization with exact inverse bijection
- deterministic scene-aware stratified sampling eliminates early-scene truncation confound
- zero source_sample_id duplicates within any dataset
- zero cross-group presentation-order disagreements on shared source samples
- 6928 / 6928 dataset records independently verified against raw 3D geometric camera truth (0 mismatches)
- bit-for-bit build determinism verified (identical SHA-256 hashes across repeated generation)
- holdout S1 (cardinal, N=1136) and S2 (jitter, N=1136) materialized and disjoint from train

Verified after G2.0-E2:
- total CPU unittest suite: 188 tests passed (181 pre-existing + 7 new F1/F2 regression tests)
- Qwen2.5-VL-3B-Instruct (bf16) LoRA pipeline end-to-end verified on AutoDL RTX 4080 SUPER
- historical LoRA preserved: r=8, alpha=16, dropout=0.05
- strict language-model-only LoRA: 252 LM projection modules, 0 vision modules, 14,966,784 trainable params
- vision tower remains 100% frozen (0 trainable visual params)
- formal training semantics: batch=1, grad accumulation=8, lr=1e-4, epochs=1, gradient checkpointing=true, linear scheduler, warmup=0, no QLoRA
- frozen E1 group size N=1552 -> 1552 microbatches / 8 = 194 optimizer steps per epoch (0 remainder)
- verified accumulation: microbatches 1-7 -> no optimizer step, weights stable; microbatch 8 -> exactly 1 optimizer + 1 scheduler step, weights updated
- RGBA inputs explicitly converted to RGB
- prompt masked with -100 (362 tokens); answer + EOS (<|im_end|>, 2 tokens) supervised; attention_mask all 1s
- language-only adapter save/reload verified (~60 MB); reloaded model preserves 0 visual LoRA params
- train and held-out S1 greedy inference smoke verified without ground-truth leakage
- peak smoke VRAM ~13.5 GB (headroom >18 GB)
- F1 blocker fixed: eliminated 96 visual LoRA modules via explicit LM allowlist
- F2 blocker fixed: implemented formal gradient accumulation loop, linear scheduler, and enabled gradient checkpointing

NEXT IMPLEMENTATION:
G2.0-E3 Formal Controlled Training & Transfer Evaluation (NOT started)
```

环境说明：

- AMD Radeon Developer Cloud 已从当前开发工作流停用（其 One-click 池无法满足最低 GPU 资源请求）
- 当前开发 / 计算环境为 AutoDL 实例（远程 Linux / CUDA）
- OpenCode CLI 直接运行在 AutoDL 实例上；本地机器为控制 / 协调端
- AutoDL 当前承担 CPU / code 工作；计划用于 CUDA / GPU workloads
- Blender 4.2.0 已在 AutoDL 实例就绪并完成 G2.0-E1 800 张图像渲染
- GitHub 仍为长期真源；用户保留 commit / push 最终决策

当前不要做：

- 不要重做 v1
- 不要重写 G2.0-A / B / B.1 / C / D / E1 / E2
- 不要把 4 fixed cameras 当最终训练方案
- 不要直接引入复杂 Agent before camera/truth/curriculum layers are stable
- 不要把 God View 暴露给 Agent
- 不要声称 G2.0-E3 已经开始
- 不要跳过单 seed=42 验证直接启动全量 A/C/D 多 seed 训练
- 不要修改历史 v1 优化语义（如擅自添加 cosine scheduler 或 warmup）
- 不要将 S2 宣传为连续球面上强 OOD 视点泛化
- 不要将 front-halfspace 等同于像素级可见/无遮挡

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