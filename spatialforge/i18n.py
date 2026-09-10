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
