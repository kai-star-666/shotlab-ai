"""V0.3 投篮评估纯计算层：参考表、时序、问题与训练处方。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProblemFinding:
    title: str
    evidence: str
    explanation: str


@dataclass(frozen=True)
class TrainingDrill:
    name: str
    focus: str
    method: str
    dosage: str


@dataclass(frozen=True)
class DiagnosticResult:
    problems: list[ProblemFinding]
    actions: list[str]
    drills: list[TrainingDrill]
    hand_analysis: str


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _degrees(value: object) -> str:
    return f"{float(value):.1f}°" if _finite(value) else "数据不足"


def build_angle_comparison(metrics: Mapping[str, float]) -> pd.DataFrame:
    """建立技术参考表；区间是观察尺度，不是唯一标准动作。"""
    knee = metrics.get("knee_lowest", math.nan)
    hip = metrics.get("hip_lowest", math.nan)
    elbow = metrics.get("elbow_release", math.nan)
    trunk = metrics.get("trunk_max_abs", math.nan)
    alignment = metrics.get("elbow_alignment", math.nan)
    wrist_height = metrics.get("wrist_release_relative_shoulder", math.nan)
    wrist_direction = metrics.get("wrist_path_deviation", math.nan)
    snap = metrics.get("wrist_snap_change", math.nan)
    ball_angle = metrics.get("ball_launch_angle_2d", math.nan)

    def ranged(value: float, low: float, high: float) -> str:
        if not _finite(value):
            return "数据不足"
        if value < low:
            return "低于常见参考区间"
        if value > high:
            return "高于常见参考区间"
        return "落在常见参考区间"

    rows = [
        ("最低点膝关节角", _degrees(knee), "研究显示会随距离与任务变化；暂无通用唯一区间", "记录个人基线" if _finite(knee) else "数据不足", "A·二维直接测量"),
        ("最低点髋关节角", _degrees(hip), "暂无通用唯一区间，更适合个人对比", "记录个人基线" if _finite(hip) else "数据不足", "A·二维直接测量"),
        ("出手候选帧肘角", _degrees(elbow), "研究显示会随距离、熟练度和阶段定义变化；无通用唯一角度", "低于系统观察阈值，需结合视频复核" if _finite(elbow) and elbow < 145 else ("记录个人基线" if _finite(elbow) else "数据不足"), "A·二维直接测量"),
        ("躯干最大二维倾角", _degrees(trunk), "建议幅度较小且过程稳定，15°仅作观察阈值", "波动偏大" if _finite(trunk) and trunk > 15 else ("当前趋势较小" if _finite(trunk) else "数据不足"), "A·二维直接测量"),
        ("肘部横向对齐", f"{float(alignment):.3f}" if _finite(alignment) else "数据不足", "无可靠通用三维标准，仅观察当前视角横向偏移", "偏移较明显" if _finite(alignment) and alignment > 0.15 else ("当前视角未见明显偏移" if _finite(alignment) else "视角/数据不足"), "B·二维估计"),
        ("出手手腕相对肩高", f"{float(wrist_height):.3f}" if _finite(wrist_height) else "数据不足", "更适合记录个人稳定基线", "可用于连续投篮对比" if _finite(wrist_height) else "数据不足", "A·画面坐标"),
        ("手腕轨迹横向离散", f"{float(wrist_direction):.3f}" if _finite(wrist_direction) else "数据不足", "数值越小表示当前二维轨迹越集中", "轨迹偏移值较大" if _finite(wrist_direction) and wrist_direction > 0.035 else ("轨迹较集中" if _finite(wrist_direction) else "数据不足"), "A·画面轨迹"),
        ("出手后翻腕变化", _degrees(snap), "暂无可靠单摄像头通用标准", "仅作个人趋势" if _finite(snap) else "手部数据不足", "B·手部估计"),
        ("篮球二维初始飞行方向", _degrees(ball_angle), "只有轨迹可信时才输出，不等于真实三维出射角", "二维画面估计" if _finite(ball_angle) else "数据不足", "B·轨迹估计"),
    ]
    return pd.DataFrame(rows, columns=["动作指标", "我的实际数据", "参考范围", "当前判断", "性质"])


def _extension_onset(values: Sequence[float], start: int, stop: int) -> int | None:
    data = np.asarray(values, dtype=float)
    stop = min(stop, len(data))
    if stop - start < 3:
        return None
    segment = data[start:stop]
    if not np.isfinite(segment).any():
        return None
    valid = np.isfinite(segment)
    segment = np.interp(np.arange(len(segment)), np.flatnonzero(valid), segment[valid])
    velocity = np.gradient(segment)
    peak = float(np.max(velocity))
    if peak <= 0:
        return None
    candidates = np.flatnonzero(velocity >= max(2.0, peak * 0.25))
    return start + int(candidates[0]) if candidates.size else None


def build_timing_analysis(
    data: pd.DataFrame, phases: Mapping[str, int], fps: float
) -> dict[str, float]:
    fps = max(float(fps), 1.0)
    ready = int(phases["ready"])
    lowest = int(phases["lowest"])
    release = int(phases["release_candidate"])
    follow = int(phases["followthrough"])
    knee_onset = _extension_onset(data["knee_angle"].tolist(), lowest, release + 1)
    elbow_onset = _extension_onset(data["elbow_angle"].tolist(), lowest, release + 1)
    lag = (
        (elbow_onset - knee_onset) / fps
        if knee_onset is not None and elbow_onset is not None
        else math.nan
    )
    return {
        "准备到最低点": (lowest - ready) / fps,
        "最低点到出手候选": (release - lowest) / fps,
        "出手候选到随挥": (follow - release) / fps,
        "膝部快速伸展起点": knee_onset / fps if knee_onset is not None else math.nan,
        "肘部快速伸展起点": elbow_onset / fps if elbow_onset is not None else math.nan,
        "上下肢快速伸展起点时间差": lag,
        "动作片段时长": (follow - ready) / fps,
    }


def build_stability_scores(metrics: Mapping[str, float]) -> dict[str, int | None]:
    def bounded(value: float) -> int:
        return int(round(float(np.clip(value, 0, 100))))

    knee_smooth = metrics.get("knee_cycle_smoothness", math.nan)
    elbow_smooth = metrics.get("elbow_cycle_smoothness", math.nan)
    trunk_std = metrics.get("trunk_std", math.nan)
    wrist_deviation = metrics.get("wrist_path_deviation", math.nan)
    return {
        "下肢动作稳定性": bounded(float(knee_smooth) * 100) if _finite(knee_smooth) else None,
        "肘部动作平顺性": bounded(float(elbow_smooth) * 100) if _finite(elbow_smooth) else None,
        "躯干稳定性": bounded(100 - float(trunk_std) * 5) if _finite(trunk_std) else None,
        "手腕轨迹稳定性": bounded(100 - float(wrist_deviation) * 1200) if _finite(wrist_deviation) else None,
    }


def estimate_2d_ball_launch_angle(
    ball_x: Sequence[float], ball_y: Sequence[float], release_frame: int
) -> float:
    points = [
        (float(ball_x[index]), float(ball_y[index]))
        for index in range(max(0, release_frame), min(len(ball_x), release_frame + 8))
        if _finite(ball_x[index]) and _finite(ball_y[index])
    ]
    if len(points) < 4:
        return math.nan
    start, end = points[0], points[-1]
    dx = abs(end[0] - start[0])
    upward = start[1] - end[1]
    if upward <= 0.03 or math.hypot(dx, upward) < 0.06:
        return math.nan
    return float(np.degrees(np.arctan2(upward, max(dx, 1e-6))))


def build_diagnostics(
    metrics: Mapping[str, float], *, hand_reliable: bool
) -> DiagnosticResult:
    candidates: list[tuple[str, ProblemFinding, str, TrainingDrill]] = []

    knee_range = metrics.get("knee_motion_range", math.nan)
    if _finite(knee_range) and knee_range < 18:
        candidates.append((
            "knee",
            ProblemFinding("下沉幅度偏小", f"本次屈膝变化约 {knee_range:.1f}°，低于系统 18° 观察阈值。", "当前二维视频中腿部蓄力变化不明显，仍需结合投篮距离和节奏判断。"),
            "保持原有出手节奏，尝试让屈膝—伸膝过程更连贯，不要为追求角度刻意深蹲。",
            TrainingDrill("固定节奏中近距离投篮", "重复相近的下沉深度与起身速度", "每球用相同脚步和口令节拍完成，每 10 球录像复查。", "3 组 × 10 球"),
        ))

    elbow = metrics.get("elbow_release", math.nan)
    if _finite(elbow) and elbow < 145:
        candidates.append((
            "elbow",
            ProblemFinding("出手候选阶段伸肘偏小", f"出手候选帧肘角约 {elbow:.1f}°。", "该数值可能来自伸肘时序、个人 set point 或侧面投影，不代表必须追求某个固定角度。"),
            "用近筐动作检查‘先顺畅抬球，再连续伸肘’，避免在 set point 长时间停顿。",
            TrainingDrill("近筐单手定型投篮", "肘—腕伸展顺序", "在篮框前 1–2 米，只用投篮手完成，随挥停留 1 秒复查肘腕方向。", "3 组 × 10 球"),
        ))

    trunk_ready = metrics.get("trunk_ready", math.nan)
    trunk_release = metrics.get("trunk_release", math.nan)
    trunk_std = metrics.get("trunk_std", math.nan)
    trunk_change = abs(float(trunk_release) - float(trunk_ready)) if _finite(trunk_ready) and _finite(trunk_release) else math.nan
    if (_finite(trunk_change) and trunk_change > 8) or (_finite(trunk_std) and trunk_std > 6):
        candidates.append((
            "trunk",
            ProblemFinding("出手过程躯干变化较大", f"准备到出手候选的躯干二维倾角变化约 {trunk_change:.1f}°，过程标准差约 {float(trunk_std):.1f}°。", "这表示当前画面中躯干轴线波动明显，不能单独区分真实前后仰与左右侧倾。"),
            "投篮时保持头—肩—髋整体上移，减少出手前额外横向摆动或后倒。",
            TrainingDrill("平衡定点投篮", "出手过程躯干稳定", "选择中近距离定点，落地后检查头、肩、髋是否仍在稳定支撑面内。", "3 个点位 × 每点 8 球"),
        ))

    alignment = metrics.get("elbow_alignment", math.nan)
    if _finite(alignment) and alignment > 0.15:
        candidates.append((
            "alignment",
            ProblemFinding("当前视角下肘部横向偏移较明显", f"肘相对肩—腕连线的归一化偏移约 {alignment:.3f}。", "这只是二维对齐趋势，不能断言肘部在三维空间是否对准篮框。"),
            "抬球时关注肘与手腕的横向路径是否突然向外绕，不要强行把肘夹到身体上。",
            TrainingDrill("墙边抬球路径练习", "减少抬球过程的横向绕行", "在不触碰墙面的安全距离内慢速做抬球—伸肘动作，再过渡到近筐投篮。", "2 组徒手 × 8 次 + 2 组 × 10 球"),
        ))

    wrist = metrics.get("wrist_path_deviation", math.nan)
    if _finite(wrist) and wrist > 0.035:
        candidates.append((
            "wrist",
            ProblemFinding("手腕轨迹横向离散偏大", f"最低点到随挥的手腕路径离散值约 {wrist:.3f}。", "该值是画面坐标中的趋势，可用于同机位的个人重复性比较。"),
            "出手后保持手臂向目标方向自然延伸，定格 1 秒观察手腕是否稳定停在同一通道。",
            TrainingDrill("随挥定格练习", "稳定手腕与手指随挥方向", "每次近筐投篮后保持随挥 1 秒，用正面或侧前方视频比较路径。", "3 组 × 10 球"),
        ))

    selected = candidates[:5]
    drills: list[TrainingDrill] = []
    for _, _, _, drill in selected:
        if drill.name not in {item.name for item in drills} and len(drills) < 4:
            drills.append(drill)
    if len(drills) < 2:
        drills.extend([
            TrainingDrill("同机位连续投篮", "建立个人稳定基线", "固定机位、距离和接球方式，录制至少 10 球再比较标准差。", "2 组 × 10 球"),
            TrainingDrill("近筐动作复现", "保持伸膝—伸髋—伸肘—随挥连贯", "不追求速度，用可控节奏完成完整动作。", "2 组 × 10 球"),
        ][: 2 - len(drills)])

    hand_analysis = (
        "手部关键点通过当前置信度校验，可观察翻腕和指向趋势，但不代表真实手指施力。"
        if hand_reliable
        else "当前视频由于篮球遮挡、人物过小或拍摄角度，手部关键点数据不足，本次不对具体手型做判断。"
    )
    return DiagnosticResult(
        problems=[item[1] for item in selected],
        actions=[item[2] for item in selected],
        drills=drills,
        hand_analysis=hand_analysis,
    )
