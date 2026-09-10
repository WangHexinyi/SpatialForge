"""Central i18n layer for SpatialForge (Inspector + labels).

Supported languages:
* ``zh-CN`` / ``zh``   -- Chinese
* ``en``               -- English
* ``bilingual``        -- short labels composed as "智能体 Agent"; long prose is
                          taken from the current language setting.

UI strings are never hardcoded in HTML/JS. The frontend fetches this dictionary
over ``/api/i18n``; Python callers use :func:`t` / :func:`ls`.
"""

from __future__ import annotations

from typing import Dict

SUPPORTED_LANGS = ("zh-CN", "zh", "en", "bilingual")
DEFAULT_LANG = "bilingual"

# ---------------------------------------------------------------------------
# Dictionary. Each key maps to per-language text (or {short,long} tuple set).
# "long" used for paragraphs/help; "short" for compact labels.
# ---------------------------------------------------------------------------

_zh = {
    "app.title": {"short": "空间熔炉", "long": "SpatialForge 具身研究检查器"},
    "top.env": {"short": "当前环境", "long": "当前环境"},
    "top.task": {"short": "当前任务", "long": "当前任务"},
    "top.agent": {"short": "智能体状态", "long": "智能体状态"},
    "top.lang": {"short": "语言", "long": "语言"},
    "privacy": {
        "short": "研究者特权真值 · 绝不作为模型输入",
        "long": "右侧 Debug / 环境面板中的特权数据仅供研究者与判卷，绝不进入模型输入。",
    },
    # tabs
    "tab.observation": {"short": "观察", "long": "智能体第一人称观察"},
    "tab.agent": {"short": "智能体", "long": "研究者可见的智能体状态"},
    "tab.task": {"short": "任务", "long": "具身物体搜索任务"},
    "tab.environment": {"short": "环境", "long": "ProcTHOR 环境"},
    "tab.training": {"short": "训练", "long": "训练执行配置"},
    "tab.debug": {"short": "调试", "long": "研究者特权真值 / 判卷"},
    "tab.episodes": {"short": "回合回放", "long": "教师 Rollout 回合回放（研究者专用）"},
    "tab.settings": {"short": "设置", "long": "设置"},
    "ep.researcher": {
        "short": "研究者专用回合 · 特权真值 · 绝不作为模型输入",
        "long": "Researcher-only teacher-rollout inspection · privileged truth · never model input",
    },
    "ep.list": {"short": "已保存回合", "long": "已保存教师回合"},
    "ep.timeline": {"short": "时间线", "long": "点击步骤查看对应帧与特权状态"},
    "ep.target": {"short": "目标(特权)", "long": "Privileged target id / position"},
    "ep.action": {"short": "动作", "long": "教师动作"},
    "ep.visible": {"short": "目标可见", "long": "authoritative target visible"},
    # observation tab
    "obs.firstperson": {"short": "第一人称 RGB", "long": "Agent 第一人称视角 RGB"},
    "obs.step": {"short": "步数 / 时间戳", "long": "当前 observation 步数与时间戳"},
    "obs.horizon": {"short": "Camera 俯仰", "long": "当前 camera horizon（俯仰角）"},
    "obs.last_action": {"short": "上一步动作", "long": "上一步动作及其成败"},
    "obs.history": {"short": "历史", "long": "模型观察历史入口"},
    # agent tab
    "agent.pose": {"short": "位姿", "long": "研究者可见位姿"},
    "agent.stance": {"short": "站/蹲", "long": "站立或蹲下"},
    "agent.room": {"short": "房间", "long": "当前房间（如可获取）"},
    # task tab
    "task.goal": {"short": "目标", "long": "寻找目标物体"},
    "task.category": {"short": "目标类别", "long": "目标物体类别"},
    "task.episode": {"short": "回合状态", "long": "Episode 状态 / 步数预算"},
    # environment tab
    "env.house": {"short": "房屋 ID", "long": "ProcTHOR house ID"},
    "env.split": {"short": "数据划分", "long": "dataset split"},
    "env.rooms": {"short": "房间数", "long": "房间数量"},
    "env.backend": {"short": "渲染后端", "long": "渲染后端 / 物理运行时"},
    # debug tab
    "debug.warn": {
        "short": "特权调试数据 · 绝不作为模型输入",
        "long": "PRIVILEGED DEBUG DATA · NEVER MODEL INPUT",
    },
    "common.start": {"short": "开始回合", "long": "开始新的物体搜索回合"},
    "common.teacher": {"short": "生成教师路径", "long": "生成教师参考路径（特权）"},
    "agent.pose.long": {
        "short": "研究者/调试层位姿",
        "long": "该位姿元数据属于研究者/调试层，不自动加入模型输入。",
    },
    "note.training": {
        "short": "目标形式 π(a_t | o≤t, goal)",
        "long": "训练目标形式：根据观察/历史与目标预测下一动作 π(a_t | o_≤t, goal, history)。旧 toy-camera curriculum 已移出主训练入口。",
    },
    # common action-ish labels
    "common.running": {"short": "运行中", "long": "运行中"},
    "common.success": {"short": "成功", "long": "成功"},
    "common.failure": {"short": "失败", "long": "失败"},
    "common.step": {"short": "步", "long": "步"},
    # workbench layout (research inspector)
    "wb.subtitle": {"short": "具身研究工作台", "long": "具身研究工作台"},
    "mode.live": {"short": "LIVE", "long": "实时模式"},
    "mode.replay": {"short": "REPLAY", "long": "回放模式"},
    "wb.godview": {"short": "God View", "long": "God View 研究者特权视图"},
    "wb.decision": {"short": "决策追踪", "long": "模型决策与判卷追踪"},
    "wb.timeline": {"short": "回合时间线", "long": "回合时间线（点击步骤查看）"},
    "wb.episodes": {"short": "回合列表", "long": "已保存回合列表"},
    "wb.metrics": {"short": "指标", "long": "回合指标"},
    "wb.debug": {"short": "调试详情", "long": "研究者调试详情"},
    "wb.manual": {"short": "手动控制", "long": "手动控制（次要面板）"},
    "wb.settings": {"short": "设置", "long": "设置"},
    "wb.privileged": {"short": "特权元数据", "long": "研究者特权元数据（绝不作为模型输入）"},
    "privacy.short": {
        "short": "研究者特权真值 · 绝不作为模型输入",
        "long": "研究者特权真值 · 绝不作为模型输入",
    },
    "dec.raw": {"short": "模型原始输出", "long": "模型原始输出（raw）"},
    "dec.parsed": {"short": "解析动作", "long": "严格解析后的动作"},
    "dec.executed": {"short": "执行动作", "long": "环境中实际执行的动作"},
    "dec.teacher": {"short": "教师参考动作", "long": "教师参考动作（如记录）"},
    "dec.visible": {"short": "目标可见", "long": "authoritative target visible"},
    "dec.verifier": {"short": "判卷", "long": "verifier 判定"},
    "dec.terminal": {"short": "终止 / 失败原因", "long": "终止或失败原因"},
    "dec.latency": {"short": "延迟", "long": "决策延迟"},
    "dec.teacher_na": {"short": "模型控制回合不存逐步教师参考", "long": "模型控制回合不存储逐步教师参考动作"},
    "dec.manual": {"short": "手动回合（无模型决策）", "long": "手动回合，无模型决策"},
    "tl.start": {"short": "起始", "long": "回合起始"},
    "tl.empty": {"short": "未加载回合", "long": "尚未加载回放回合"},
    "tl.click": {"short": "点击步骤查看帧 / 位姿 / 判卷", "long": "点击步骤查看对应帧、位姿与判卷"},
    "overlay.trajectory": {"short": "轨迹", "long": "智能体轨迹"},
    "overlay.fov": {"short": "视野", "long": "当前朝向视野（近似）"},
    "overlay.target": {"short": "目标真值", "long": "目标特权真值"},
    "overlay.teacher": {"short": "教师路径", "long": "教师参考路径"},
    "god.no_data": {"short": "无可用世界数据", "long": "当前没有可用的世界数据"},
    "god.approx": {"short": "近似朝向", "long": "视野楔形为近似朝向，非精确投影"},
    "god.plane": {"short": "平面", "long": "俯视平面"},
    "god.step": {"short": "步", "long": "步"},
    "fpv.none": {"short": "无画面", "long": "暂无第一人称画面"},
    "fpv.visible": {"short": "目标可见", "long": "目标可见"},
    "fpv.not_visible": {"short": "目标不可见", "long": "目标不可见"},
    "live.poll": {"short": "LIVE 轮询", "long": "实时轮询运行中回合"},
    "live.running": {"short": "实时回合运行中", "long": "实时回合运行中"},
    "live.waiting": {"short": "等待实时状态", "long": "等待实时状态"},
    "live.target_na": {"short": "实时回合暂不提供目标坐标", "long": "实时回合运行期间暂不提供目标坐标"},
    "metrics.none": {"short": "无决策数据", "long": "无决策数据"},
    "metrics.outcome": {"short": "结果", "long": "回合结果"},
    "metrics.steps": {"short": "执行步数", "long": "执行步数"},
    "metrics.decisions": {"short": "决策数", "long": "决策数"},
    "metrics.invalid": {"short": "无效决策", "long": "无效决策数"},
    "metrics.visible": {"short": "可见决策", "long": "决策时目标可见次数"},
    "metrics.latency": {"short": "平均延迟", "long": "平均决策延迟"},
    "metrics.actions": {"short": "动作分布", "long": "模型动作分布"},
    "metrics.terminal": {"short": "终止原因", "long": "终止原因"},
    "metrics.spawn": {"short": "出生可见", "long": "出生时目标可见"},
    "ep.refresh": {"short": "刷新", "long": "刷新回合列表"},
    "ep.load": {"short": "加载", "long": "加载选中回合"},
    "ep.none": {"short": "无已保存回合", "long": "没有已保存回合"},
    "ep.one_valid": {"short": "仅一个 VALID post-fix 回合；其余为 legacy/raw（带标签）", "long": "当前只有一个 VALID post-fix 回合；下拉中其余回合为 legacy/raw，已带标签，可显式选择查看"},
    "common.view_json": {"short": "查看 JSON", "long": "查看原始 JSON"},
    "common.yes": {"short": "是", "long": "是"},
    "common.no": {"short": "否", "long": "否"},
    "manual.note": {
        "short": "回放模式下手动控制为次要面板",
        "long": "回放模式以 God View / RGB / 时间线 / 决策追踪为主，手动控制为次要面板。",
    },
    "manual.house": {"short": "房屋", "long": "选择用于真实 ProcTHOR 手动控制的房屋"},
    "manual.live": {"short": "手动 LIVE 会话", "long": "手动控制为真实环境 transition，非特权 teleport"},
    "manual.replay": {"short": "回放会话", "long": "当前为回放会话，手动控制需先 Start 一个真实回合"},
    "settings.backend": {"short": "后端", "long": "渲染后端"},
    "settings.maxsteps": {"short": "最大步数", "long": "最大回合步数"},
    "debug.privileged_json": {"short": "特权 JSON", "long": "特权真值 JSON"},
    "debug.model_input": {"short": "模型输入（仅允许字段）", "long": "模型输入（仅允许字段）"},
    "debug.teacher": {"short": "教师路径", "long": "教师路径"},
    "drawer.hint": {"short": "次级面板", "long": "次级面板：回合 / 指标 / 调试 / 手动控制 / 设置"},
    "player.play": {"short": "播放", "long": "播放回合"},
    "player.pause": {"short": "暂停", "long": "暂停播放"},
    "player.prev": {"short": "上一步", "long": "上一步"},
    "player.next": {"short": "下一步", "long": "下一步"},
    "player.speed": {"short": "倍速", "long": "播放倍速"},
    "player.step": {"short": "步", "long": "步"},
    "player.hint": {"short": "空格 播放/暂停 · ←/→ 上一步/下一步", "long": "空格播放/暂停，左右方向键切换步骤"},
    "player.end": {"short": "已到终止步", "long": "已播放至终止步"},
    # 3D God View
    "god.view3d": {"short": "三维语义", "long": "三维语义研究视图（Three.js/WebGL）"},
    "god.view2d": {"short": "二维回退", "long": "二维俯视回退视图"},
    "god.unity": {"short": "Unity 相机", "long": "真实 AI2-THOR Unity 检视相机"},
    "god.unity_main": {"short": "Unity 主视图", "long": "真实 AI2-THOR Unity 渲染的主 God View（非模型输入）"},
    "god.unity_follow": {"short": "跟随", "long": "跟随智能体视角的真实 Unity 相机"},
    "god.unity_overview": {"short": "全景", "long": "房屋全景真实 Unity 相机"},
    "god.unity_rendering": {"short": "正在渲染真实 Unity 场景…", "long": "正在请求 AI2-THOR 渲染真实场景（首次加载引擎较慢）"},
    "god.unity_unavailable": {"short": "真实 Unity God View 不可用", "long": "当前无法启动 AI2-THOR 渲染（缺少引擎或显示环境）"},
    "god.unity_replay_only": {"short": "真实 Unity God View 仅用于回放回合", "long": "真实 Unity 主视图针对已保存回放回合；实时/手动模式无特权渲染引擎"},
    "god.diagnostic": {"short": "辅助诊断", "long": "Three.js 语义图仅为辅助诊断视图，非最终 God View"},
    "ep.condensed": {"short": "压缩", "long": "压缩展示无进展段落"},
    "ep.collapsed_run": {"short": "{action} × {count} 无进展 (Δ{disp}m)", "long": "折叠的无进展段落：{action} 重复 {count} 次，位移 {disp}m"},
    "warn.legacy": {"short": "历史低质量回放（修复前）", "long": "该回合生成于渲染质量/出生语义修复之前，FPV 可能过曝、轨迹含 setup 位移"},
    "warn.sensor": {"short": "传感器质量不可用", "long": "渲染质量缺失或为已知过曝档位，FPV 可能不可辨"},
    "warn.trajectory": {"short": "轨迹含 setup/无法解释位移（已断线）", "long": "检测到 setup/出生/瞬移位移，轨迹已断线而非直连穿墙"},
    "warn.collapse": {"short": "策略坍缩（单一动作主导）", "long": "检测到策略坍缩：单一动作占比过高，回合科研价值低"},
    "warn.stuck": {"short": "卡死/无进展", "long": "长时间无空间进展，建议使用压缩回放"},
    "god.cam_persp": {"short": "透视", "long": "透视视角"},
    "god.cam_top": {"short": "俯视", "long": "俯视视角"},
    "god.cam_follow": {"short": "跟随", "long": "跟随智能体"},
    "god.cam_reset": {"short": "重置", "long": "重置相机"},
    "god.model": {"short": "Actor 模型", "long": "当前 Actor 模型与执行配置"},
    "god.telemetry": {"short": "遥测", "long": "GPU / 吞吐遥测（仅展示，不进入 Agent 关键路径）"},
    "god.semantics_ok": {"short": "出生/重置语义 OK", "long": "出生与重置均为环境 setup，非模型动作"},
    "god.semantics_bad": {"short": "出生/重置语义违规", "long": "检测到无法解释的位移（疑似 TeleportFull）"},
    "god.geometry_unavailable": {"short": "3D 几何不可用", "long": "该回合没有可解析的房屋几何"},
    "god.scene_unavailable": {"short": "场景几何不可用", "long": "当前回合没有可解析的 ProcTHOR 场景几何，2D 仅为回合点图"},
    "god.unity_free": {"short": "自由相机", "long": "自由相机（鼠标 orbit/pan/zoom）"},
    "god.setup": {"short": "环境 setup", "long": "环境 setup / 出生（非模型动作）"},
    "god.source": {"short": "几何来源", "long": "几何数据来源"},
    "god.walls": {"short": "墙体", "long": "显示墙体"},
    "god.rooms": {"short": "房间/地面", "long": "房间地面轮廓"},
    "god.doors": {"short": "门窗", "long": "门/窗开口"},
    "god.objects": {"short": "物体代理", "long": "显示物体代理包围盒"},
    "god.terminal": {"short": "终止位置", "long": "回合终止位置"},
    "god.spawn": {"short": "出生点", "long": "出生点（环境 setup）"},
    "god.unavailable": {"short": "不可用字段", "long": "元数据未提供的字段"},
}

_en = {
    "app.title": {"short": "SpatialForge", "long": "SpatialForge Embodied Research Inspector"},
    "top.env": {"short": "Environment", "long": "Current Environment"},
    "top.task": {"short": "Task", "long": "Current Task"},
    "top.agent": {"short": "Agent Status", "long": "Agent Status"},
    "top.lang": {"short": "Language", "long": "Language"},
    "privacy": {
        "short": "Privileged researcher truth · never model input",
        "long": "Privileged truth in the Debug/Environment panels is for researchers & the verifier only and never enters model input.",
    },
    "tab.observation": {"short": "Observation", "long": "Agent first-person observation"},
    "tab.agent": {"short": "Agent", "long": "Researcher-visible agent state"},
    "tab.task": {"short": "Task", "long": "Embodied object-search task"},
    "tab.environment": {"short": "Environment", "long": "ProcTHOR environment"},
    "tab.training": {"short": "Training", "long": "Training execution profile"},
    "tab.debug": {"short": "Debug", "long": "Privileged truth / verifier"},
    "tab.episodes": {"short": "Episodes", "long": "Teacher-rollout episode replay (researcher only)"},
    "tab.settings": {"short": "Settings", "long": "Settings"},
    "ep.researcher": {
        "short": "Researcher-only episode · privileged truth · never model input",
        "long": "Researcher-only teacher-rollout inspection · privileged truth · never model input",
    },
    "ep.list": {"short": "Saved episodes", "long": "Saved teacher episodes"},
    "ep.timeline": {"short": "Timeline", "long": "Click a step to view its frame & privileged state"},
    "ep.target": {"short": "Target (priv)", "long": "Privileged target id / position"},
    "ep.action": {"short": "Action", "long": "Teacher action"},
    "ep.visible": {"short": "Target visible", "long": "authoritative target visible"},
    "obs.firstperson": {"short": "First-person RGB", "long": "Agent first-person RGB"},
    "obs.step": {"short": "Step / timestamp", "long": "Current observation step & timestamp"},
    "obs.horizon": {"short": "Camera horizon", "long": "Current camera horizon (pitch)"},
    "obs.last_action": {"short": "Last action", "long": "Last action and its outcome"},
    "obs.history": {"short": "History", "long": "Model observation-history entry"},
    "agent.pose": {"short": "Pose", "long": "Researcher-visible pose"},
    "agent.stance": {"short": "Stance", "long": "Standing or crouching"},
    "agent.room": {"short": "Room", "long": "Current room (when available)"},
    "task.goal": {"short": "Goal", "long": "Find the target object"},
    "task.category": {"short": "Target category", "long": "Target object category"},
    "task.episode": {"short": "Episode", "long": "Episode status / step budget"},
    "env.house": {"short": "House ID", "long": "ProcTHOR house ID"},
    "env.split": {"short": "Split", "long": "Dataset split"},
    "env.rooms": {"short": "Rooms", "long": "Number of rooms"},
    "env.backend": {"short": "Backend", "long": "Rendering backend / physics runtime"},
    "debug.warn": {
        "short": "Privileged debug data · never model input",
        "long": "PRIVILEGED DEBUG DATA · NEVER MODEL INPUT",
    },
    "common.start": {"short": "Start episode", "long": "Start a new object-search episode"},
    "common.teacher": {"short": "Generate teacher path", "long": "Generate privileged teacher reference path"},
    "agent.pose.long": {
        "short": "Researcher/debug pose",
        "long": "Pose metadata is researcher/debug level and is not auto-added to model input.",
    },
    "note.training": {
        "short": "Target form π(a_t | o≤t, goal)",
        "long": "Training targets predicting the next action from observation/history & goal: π(a_t | o_≤t, goal, history). The legacy toy-camera curriculum has been moved off the primary training entry.",
    },
    "common.running": {"short": "Running", "long": "Running"},
    "common.success": {"short": "Success", "long": "Success"},
    "common.failure": {"short": "Failure", "long": "Failure"},
    "common.step": {"short": "step", "long": "step"},
    # workbench layout (research inspector)
    "wb.subtitle": {"short": "Embodied research workbench", "long": "Embodied research workbench"},
    "mode.live": {"short": "LIVE", "long": "Live mode"},
    "mode.replay": {"short": "REPLAY", "long": "Replay mode"},
    "wb.godview": {"short": "God View", "long": "God View researcher-privileged view"},
    "wb.decision": {"short": "Decision trace", "long": "Model decision and verifier trace"},
    "wb.timeline": {"short": "Episode timeline", "long": "Episode timeline (click a step to inspect)"},
    "wb.episodes": {"short": "Episodes", "long": "Saved episodes"},
    "wb.metrics": {"short": "Metrics", "long": "Episode metrics"},
    "wb.debug": {"short": "Debug", "long": "Researcher debug details"},
    "wb.manual": {"short": "Manual control", "long": "Manual control (secondary panel)"},
    "wb.settings": {"short": "Settings", "long": "Settings"},
    "wb.privileged": {"short": "Privileged metadata", "long": "Researcher-privileged metadata (never model input)"},
    "privacy.short": {
        "short": "Researcher privileged truth · never model input",
        "long": "Researcher privileged truth · never model input",
    },
    "dec.raw": {"short": "Model raw output", "long": "Model raw output"},
    "dec.parsed": {"short": "Parsed action", "long": "Strictly parsed action"},
    "dec.executed": {"short": "Executed action", "long": "Action actually executed in the environment"},
    "dec.teacher": {"short": "Teacher reference", "long": "Teacher reference action (when recorded)"},
    "dec.visible": {"short": "Target visible", "long": "authoritative target visible"},
    "dec.verifier": {"short": "Verifier", "long": "Verifier verdict"},
    "dec.terminal": {"short": "Terminal / failure reason", "long": "Terminal or failure reason"},
    "dec.latency": {"short": "Latency", "long": "Decision latency"},
    "dec.teacher_na": {"short": "not stored per-step for model-controlled episodes", "long": "Teacher reference is not stored per-step for model-controlled episodes"},
    "dec.manual": {"short": "manual episode (no model decision)", "long": "Manual episode (no model decision)"},
    "tl.start": {"short": "start", "long": "episode start"},
    "tl.empty": {"short": "no episode loaded", "long": "No replay episode loaded"},
    "tl.click": {"short": "click a step to inspect frame / pose / verifier", "long": "Click a step to inspect its frame, pose and verifier verdict"},
    "overlay.trajectory": {"short": "Trajectory", "long": "Agent trajectory"},
    "overlay.fov": {"short": "FOV", "long": "Current heading FOV (approximate)"},
    "overlay.target": {"short": "Target truth", "long": "Privileged target truth"},
    "overlay.teacher": {"short": "Teacher path", "long": "Teacher reference path"},
    "god.no_data": {"short": "no world data available", "long": "No world data available"},
    "god.approx": {"short": "approximate heading", "long": "FOV wedge is approximate heading, not a precise projection"},
    "god.plane": {"short": "plane", "long": "top-down plane"},
    "god.step": {"short": "step", "long": "step"},
    "fpv.none": {"short": "no frame", "long": "No first-person frame"},
    "fpv.visible": {"short": "target visible", "long": "target visible"},
    "fpv.not_visible": {"short": "target not visible", "long": "target not visible"},
    "live.poll": {"short": "LIVE poll", "long": "Live polling of a running episode"},
    "live.running": {"short": "live episode running", "long": "live episode running"},
    "live.waiting": {"short": "waiting for live state", "long": "waiting for live state"},
    "live.target_na": {"short": "target positions unavailable during live", "long": "Target positions are unavailable while the episode is running"},
    "metrics.none": {"short": "no decision data", "long": "No decision data"},
    "metrics.outcome": {"short": "Outcome", "long": "Episode outcome"},
    "metrics.steps": {"short": "Executed steps", "long": "Executed steps"},
    "metrics.decisions": {"short": "Decisions", "long": "Decisions"},
    "metrics.invalid": {"short": "Invalid decisions", "long": "Invalid decisions"},
    "metrics.visible": {"short": "Visible at decision", "long": "Target visible at decision time"},
    "metrics.latency": {"short": "Avg latency", "long": "Average decision latency"},
    "metrics.actions": {"short": "Action distribution", "long": "Model action distribution"},
    "metrics.terminal": {"short": "Terminal reason", "long": "Terminal reason"},
    "metrics.spawn": {"short": "Spawn visible", "long": "Target visible at spawn"},
    "ep.refresh": {"short": "Refresh", "long": "Refresh episode list"},
    "ep.load": {"short": "Load", "long": "Load selected episode"},
    "ep.none": {"short": "no saved episodes", "long": "No saved episodes"},
    "ep.one_valid": {"short": "Only one VALID post-fix episode; others are legacy/raw (labelled)", "long": "There is only one VALID post-fix episode; the rest are legacy/raw, clearly labelled and still selectable"},
    "common.view_json": {"short": "view JSON", "long": "View raw JSON"},
    "common.yes": {"short": "yes", "long": "yes"},
    "common.no": {"short": "no", "long": "no"},
    "manual.note": {
        "short": "manual control is secondary in replay mode",
        "long": "Manual control is a secondary panel in replay mode.",
    },
    "manual.house": {"short": "House", "long": "House used for real ProcTHOR manual control"},
    "manual.live": {"short": "Manual LIVE session", "long": "Manual control runs real environment transitions, never privileged teleport"},
    "manual.replay": {"short": "Replay session", "long": "Replay session active; start a real episode to use manual control"},
    "settings.backend": {"short": "backend", "long": "rendering backend"},
    "settings.maxsteps": {"short": "max episode length", "long": "maximum episode length"},
    "debug.privileged_json": {"short": "privileged JSON", "long": "privileged truth JSON"},
    "debug.model_input": {"short": "model input (permitted only)", "long": "model input (permitted fields only)"},
    "debug.teacher": {"short": "teacher path", "long": "teacher path"},
    "drawer.hint": {"short": "secondary panels", "long": "Secondary panels: episodes / metrics / debug / manual / settings"},
    "player.play": {"short": "Play", "long": "Play episode"},
    "player.pause": {"short": "Pause", "long": "Pause playback"},
    "player.prev": {"short": "Previous step", "long": "Previous step"},
    "player.next": {"short": "Next step", "long": "Next step"},
    "player.speed": {"short": "Speed", "long": "Playback speed"},
    "player.step": {"short": "step", "long": "step"},
    "player.hint": {"short": "Space play/pause · ←/→ prev/next", "long": "Space to play/pause, arrow keys to step"},
    "player.end": {"short": "reached terminal step", "long": "Reached the terminal step"},
    # 3D God View
    "god.view3d": {"short": "3D semantic", "long": "3D semantic research view (Three.js/WebGL)"},
    "god.view2d": {"short": "2D fallback", "long": "2D top-down fallback view"},
    "god.unity": {"short": "Unity camera", "long": "Real AI2-THOR Unity inspection camera"},
    "god.unity_main": {"short": "Unity God View", "long": "Real AI2-THOR Unity-rendered primary God View (never model input)"},
    "god.unity_follow": {"short": "Follow", "long": "Real Unity camera following the agent"},
    "god.unity_overview": {"short": "Overview", "long": "Real Unity house overview camera"},
    "god.unity_rendering": {"short": "Rendering real Unity scene…", "long": "Requesting an AI2-THOR render (first engine load is slow)"},
    "god.unity_unavailable": {"short": "Real Unity God View unavailable", "long": "AI2-THOR rendering cannot start (missing engine or display)"},
    "god.unity_replay_only": {"short": "Real Unity God View is for saved replays", "long": "The real Unity main view targets saved replay episodes; live/manual modes have no privileged render engine"},
    "god.diagnostic": {"short": "diagnostic", "long": "Three.js semantic map is a secondary diagnostic view, not the final God View"},
    "ep.condensed": {"short": "Condensed", "long": "Collapse no-progress runs"},
    "ep.collapsed_run": {"short": "{action} × {count} no-progress (Δ{disp}m)", "long": "Collapsed no-progress run: {action} repeated {count}x, displacement {disp}m"},
    "warn.legacy": {"short": "Legacy low-quality replay (pre-fix)", "long": "Episode predates the render-quality / spawn-semantics fix; FPV may be over-exposed and trajectory may contain setup displacement"},
    "warn.sensor": {"short": "Sensor quality invalid", "long": "Render quality missing or a known over-exposed tier; FPV may be unreadable"},
    "warn.trajectory": {"short": "Trajectory contains setup/unexplained displacement (broken)", "long": "Setup/spawn/teleport displacement detected; the trajectory is broken, not drawn through walls"},
    "warn.collapse": {"short": "Policy collapse (one action dominates)", "long": "Policy collapse detected: one action dominates, low research value"},
    "warn.stuck": {"short": "Stuck / no progress", "long": "Long run with no spatial progress; use condensed replay"},
    "god.cam_persp": {"short": "Perspective", "long": "Perspective view"},
    "god.cam_top": {"short": "Top", "long": "Top-down view"},
    "god.cam_follow": {"short": "Follow", "long": "Follow the agent"},
    "god.cam_reset": {"short": "Reset", "long": "Reset camera"},
    "god.model": {"short": "Actor model", "long": "Current Actor model & execution profile"},
    "god.telemetry": {"short": "Telemetry", "long": "GPU / throughput telemetry (display only; never in the Agent critical path)"},
    "god.semantics_ok": {"short": "spawn/reset semantics OK", "long": "Spawn & reset are environment setup, not model actions"},
    "god.semantics_bad": {"short": "spawn/reset semantics VIOLATION", "long": "Unexplained displacement detected (possible TeleportFull)"},
    "god.geometry_unavailable": {"short": "3D geometry unavailable", "long": "No resolvable house geometry for this episode"},
    "god.scene_unavailable": {"short": "Scene geometry unavailable", "long": "No resolvable ProcTHOR scene geometry for this episode; the 2D view is episode-only"},
    "god.unity_free": {"short": "Free camera", "long": "Free camera (mouse orbit/pan/zoom)"},
    "god.setup": {"short": "environment setup", "long": "Environment setup / spawn (not a model action)"},
    "god.source": {"short": "geometry source", "long": "Geometry data source"},
    "god.walls": {"short": "walls", "long": "Show walls"},
    "god.rooms": {"short": "rooms/floors", "long": "Room floor footprints"},
    "god.doors": {"short": "doors/windows", "long": "Door / window openings"},
    "god.objects": {"short": "object proxies", "long": "Show object proxy boxes"},
    "god.terminal": {"short": "terminal position", "long": "Episode terminal position"},
    "god.spawn": {"short": "spawn", "long": "Spawn point (environment setup)"},
    "god.unavailable": {"short": "unavailable fields", "long": "Fields not provided by the metadata"},
}

_i18n = {"en": _en, "zh": _zh}


def _norm(lang: str) -> str:
    if lang in ("zh-CN", "zh"):
        return "zh"
    if lang in ("en",):
        return "en"
    return "en"  # bilingual falls back to en for long prose


def ls(key: str, lang: str = DEFAULT_LANG) -> str:
    """Short compact label. In bilingual mode composes 'zh en'."""
    base = _norm(lang)
    entry = _i18n[base].get(key)
    en_entry = _i18n["en"].get(key)
    if entry is None:
        # missing key fallback: return key itself with a marker
        return f"[{key}]"
    if lang == "bilingual":
        zh_entry = _i18n["zh"].get(key)
        zh_short = zh_entry.get("short") if zh_entry else None
        en_short = entry.get("short")
        if zh_short and zh_short != en_short:
            return f"{zh_short} · {en_short}"
        return en_short
    return entry.get("short", en_entry.get("short", key))


def t(key: str, lang: str = DEFAULT_LANG) -> str:
    """Long prose text for the current language."""
    base = _norm(lang)
    entry = _i18n[base].get(key)
    if entry is None:
        return f"[{key}]"
    return entry.get("long", entry.get("short", key))


def tab_label(key: str, lang: str = DEFAULT_LANG) -> str:
    """Tab title: bilingual combines short labels without redundancy."""
    if lang == "bilingual":
        return ls(key, "bilingual")
    return t(key, lang)


def supported_languages() -> Dict[str, str]:
    return {
        "en": "English",
        "zh-CN": "中文",
        "bilingual": "中英双语",
    }


def get_dictionary() -> Dict[str, Dict]:
    """Raw dictionary for frontend consumption (single source of truth)."""
    return {"en": _en, "zh": _zh}


def translate_keys_with_marker(marker: str = "i18n-") -> None:
    """Utility hook (reserved); does nothing in this build."""
