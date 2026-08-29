"""二维投篮动作的纯计算逻辑，不依赖 MediaPipe 或页面框架。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LandmarkPoint:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass(frozen=True)
class AnalysisPhases:
    ready: int
    lowest: int
    release: int
    followthrough: int

    def as_dict(self) -> dict[str, int]:
        return {
            "ready": self.ready,
            "lowest": self.lowest,
            "release": self.release,
            "followthrough": self.followthrough,
        }


def _xy(point: LandmarkPoint | Sequence[float]) -> np.ndarray:
    if isinstance(point, LandmarkPoint):
        return np.asarray((point.x, point.y), dtype=float)
    return np.asarray(point[:2], dtype=float)


def calculate_angle(
    a: LandmarkPoint | Sequence[float],
    b: LandmarkPoint | Sequence[float],
    c: LandmarkPoint | Sequence[float],
) -> float:
    """返回以 b 为顶点的 0~180 度夹角；退化向量返回 NaN。"""
    vector_ba = _xy(a) - _xy(b)
    vector_bc = _xy(c) - _xy(b)
    denominator = float(np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc))
    if denominator <= 1e-12:
        return math.nan
    cosine = float(np.dot(vector_ba, vector_bc) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def interpolate_short_gaps(values: Sequence[float], max_gap: int = 3) -> list[float]:
    """仅线性填补两端有有效值的短缺口，不掩盖长时间检测失败。"""
    result = np.asarray(values, dtype=float).copy()
    index = 0
    while index < len(result):
        if np.isfinite(result[index]):
            index += 1
            continue
        start = index
        while index < len(result) and not np.isfinite(result[index]):
            index += 1
        gap = index - start
        if start > 0 and index < len(result) and gap <= max_gap:
            left, right = result[start - 1], result[index]
            for offset in range(gap):
                result[start + offset] = left + (right - left) * (offset + 1) / (gap + 1)
    return result.tolist()


def _smooth(values: Sequence[float], window: int = 5) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return data
    finite = np.isfinite(data)
    if not finite.any():
        return data
    indices = np.arange(data.size)
    filled = np.interp(indices, indices[finite], data[finite])
    window = max(1, min(window, data.size))
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(filled, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def estimate_phases(
    knee_angles: Sequence[float], wrist_y: Sequence[float], fps: float
) -> AnalysisPhases:
    """用膝部最低角与手腕向上速度估计关键阶段。结果只用于趋势参考。"""
    if len(knee_angles) != len(wrist_y) or not knee_angles:
        raise ValueError("膝角和手腕轨迹必须长度一致且非空")
    knee = _smooth(knee_angles)
    wrist = _smooth(wrist_y, window=3)
    finite_knee = np.flatnonzero(np.isfinite(knee))
    if finite_knee.size == 0:
        raise ValueError("没有足够的膝关节数据用于阶段估计")

    lowest = int(finite_knee[np.argmin(knee[finite_knee])])
    pre_roll = max(1, int(round(max(fps, 1.0) * 0.35)))
    ready = max(0, lowest - pre_roll)

    velocity_y = np.gradient(wrist) * max(float(fps), 1.0)
    release_start = min(lowest + 1, len(wrist) - 1)
    release_stop = min(len(wrist), release_start + max(2, int(round(max(fps, 1.0) * 1.2))))
    segment = velocity_y[release_start:release_stop]
    if segment.size and np.isfinite(segment).any():
        release = release_start + int(np.nanargmin(segment))
    else:
        release = min(len(wrist) - 1, lowest + max(1, int(round(fps * 0.3))))
    release = max(release, lowest + 1 if lowest < len(wrist) - 1 else lowest)

    followthrough = min(
        len(wrist) - 1, release + max(1, int(round(max(fps, 1.0) * 0.3)))
    )
    return AnalysisPhases(ready, lowest, release, followthrough)


def build_training_feedback(metrics: Mapping[str, float]) -> tuple[list[str], list[str]]:
    """根据测量值生成规则化解释，避免医学判断和伪精确最佳角。"""
    summary: list[str] = []
    suggestions: list[str] = []

    valid_ratio = float(metrics.get("valid_ratio", 0.0))
    if valid_ratio < 0.6:
        summary.append("当前视频有效姿态帧偏少，结论可能受遮挡或拍摄角度影响。")

    knee_range = float(metrics.get("knee_motion_range", math.nan))
    if math.isfinite(knee_range) and knee_range < 18:
        summary.append("数据显示当前下沉幅度较浅，腿部蓄力变化不明显。")
        suggestions.append("可尝试在不破坏节奏的前提下，观察稍充分下沉是否让发力更连贯。")
    else:
        summary.append("当前视频能观察到较清晰的屈膝—伸展过程。")
        suggestions.append("建议用固定站位连续拍摄多次，重点比较每次最低点深度和起身节奏。")

    hip_std = float(metrics.get("hip_motion_std", math.nan))
    if math.isfinite(hip_std) and hip_std > 10:
        summary.append("髋部角度变化较分散，可能反映动力链准备节奏不够稳定。")
        suggestions.append("可尝试固定接球后的屈髋顺序，减少每次准备动作的额外摆动。")
    else:
        summary.append("髋部准备动作在当前单次投篮中总体连贯。")

    elbow_release = float(metrics.get("elbow_release", math.nan))
    elbow_change = float(metrics.get("elbow_extension_change", math.nan))
    elbow_std = float(metrics.get("elbow_extension_std", math.nan))
    if (math.isfinite(elbow_change) and elbow_change < 20) or (
        math.isfinite(elbow_release) and elbow_release < 145
    ):
        summary.append("出手附近肘部伸展不充分或变化较小，建议结合原视频复核。")
        suggestions.append("建议重点观察抬球路径与伸肘时序，避免只为追求角度而僵硬发力。")
    elif math.isfinite(elbow_std) and elbow_std > 12:
        summary.append("伸展阶段肘角波动较明显，单次动作节奏可能不够平顺。")
        suggestions.append("可尝试用近距离定点投篮练习固定抬球和伸肘节奏。")
    else:
        summary.append("当前视频中肘部伸展过程较连贯。")

    trunk_std = float(metrics.get("trunk_std", math.nan))
    trunk_max = float(metrics.get("trunk_max_abs", math.nan))
    if (math.isfinite(trunk_std) and trunk_std > 6) or (
        math.isfinite(trunk_max) and trunk_max > 15
    ):
        summary.append("投篮过程中躯干倾斜波动较明显，可能影响出手方向的重复性。")
        suggestions.append("建议保持稳定核心姿态，并检查相机是否接近投篮侧的正侧面。")
    else:
        summary.append("当前视频中躯干姿态总体稳定。")

    wrist_deviation = float(metrics.get("wrist_path_deviation", math.nan))
    if math.isfinite(wrist_deviation) and wrist_deviation > 0.035:
        summary.append("手腕运动轨迹存在一定横向变化；该指标属于实验性趋势。")
        suggestions.append("可尝试保持随挥方向一致，并用多次投篮对比轨迹，而非只看单次结果。")
    else:
        summary.append("手腕轨迹在当前二维画面中较集中，但无法代表精确释放角。")

    return summary, suggestions
