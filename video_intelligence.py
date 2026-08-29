"""V0.2 视频质量、出手事件和连续投篮切分的纯算法。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from biomechanics import calculate_angle


@dataclass(frozen=True)
class CaptureQuality:
    score: int
    level: str
    issues: list[str]
    corrections: list[str]
    measurements: dict[str, float]


@dataclass(frozen=True)
class ReleaseEvent:
    frame_index: int
    time: float
    source: str
    confidence: float


@dataclass(frozen=True)
class ShotWindow:
    shot_id: int
    start_frame: int
    release_frame: int
    end_frame: int
    release_source: str
    confidence: float


def assess_capture_quality(
    *,
    brightness: float,
    blur_score: float,
    pose_valid_ratio: float,
    full_body_ratio: float,
    body_height_ratio: float,
    side_view_score: float,
    resolution: tuple[int, int],
) -> CaptureQuality:
    score = 100
    issues: list[str] = []
    corrections: list[str] = []
    width, height = resolution

    if brightness < 45:
        score -= 18
        issues.append("画面光线偏暗，手部和篮球边缘容易丢失。")
        corrections.append("增加正面或侧前方光线，避免人物成为逆光剪影。")
    elif brightness > 220:
        score -= 12
        issues.append("画面部分区域可能过曝。")
        corrections.append("降低曝光或避开正对灯源的机位。")
    if blur_score < 35:
        score -= 18
        issues.append("画面运动模糊或失焦较明显。")
        corrections.append("固定相机并提高快门/现场亮度，避免手腕和篮球拖影。")
    if pose_valid_ratio < 0.7:
        score -= 16
        issues.append("人体姿态有效帧率偏低。")
        corrections.append("保证单人无遮挡，并让肩、肘、腕、髋、膝、踝持续入镜。")
    if full_body_ratio < 0.75:
        score -= 14
        issues.append("全身完整入镜比例不足。")
        corrections.append("调整相机高度或稍微后退，避免脚踝、手腕出画。")
    if body_height_ratio < 0.38:
        score -= 20
        issues.append("人物在画面中偏小，手部和篮球细节可信度不足。")
        corrections.append("相机向投篮球员靠近，让人物高度约占画面 45%–70%。")
    elif body_height_ratio > 0.88:
        score -= 8
        issues.append("人物过满，随挥或起跳可能出画。")
        corrections.append("相机稍微后退，给头顶和手臂随挥保留空间。")
    if side_view_score < 0.5:
        score -= 14
        issues.append("当前机位不像稳定的投篮侧面，二维关节角受投影影响较大。")
        corrections.append("把相机移到投篮手一侧约 90°，镜头保持水平并固定。")
    if width < 960 or height < 540:
        score -= 8
        issues.append("视频分辨率偏低。")
        corrections.append("建议至少使用 720p，手部分析优先 1080p/60fps。")

    score = max(0, min(100, score))
    level = "良好" if score >= 80 else "可用" if score >= 60 else "需重拍优化"
    if not issues:
        issues.append("当前拍摄质量满足二维趋势分析的基本条件。")
        corrections.append("保持同一机位连续拍摄，便于比较动作重复性。")
    return CaptureQuality(
        score,
        level,
        issues,
        corrections,
        {
            "brightness": float(brightness),
            "blur_score": float(blur_score),
            "pose_valid_ratio": float(pose_valid_ratio),
            "full_body_ratio": float(full_body_ratio),
            "body_height_ratio": float(body_height_ratio),
            "side_view_score": float(side_view_score),
        },
    )


def calculate_wrist_flexion(
    elbow: Sequence[float], wrist: Sequence[float], middle_mcp: Sequence[float]
) -> float:
    return calculate_angle(elbow, wrist, middle_mcp)


def validated_wrist_snap(
    release_angle: float,
    follow_angle: float,
    release_confidence: float,
    follow_confidence: float,
    *,
    max_change: float = 60.0,
) -> float:
    values = (release_angle, follow_angle, release_confidence, follow_confidence)
    if not all(math.isfinite(float(value)) for value in values):
        return math.nan
    if min(release_confidence, follow_confidence) < 0.5:
        return math.nan
    change = abs(float(follow_angle) - float(release_angle))
    return change if 3.0 <= change <= max_change else math.nan


def select_nearest_hand(
    hand_centers: Sequence[tuple[float, float]],
    pose_wrist: tuple[float, float],
    max_distance: float = 0.25,
) -> int | None:
    if not hand_centers:
        return None
    distances = [math.dist(center, pose_wrist) for center in hand_centers]
    index = int(np.argmin(distances))
    return index if distances[index] <= max_distance else None


def _interpolated(values: Sequence[float]) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).any():
        return data
    indices = np.arange(len(data))
    valid = np.isfinite(data)
    return np.interp(indices, indices[valid], data[valid])


def _ball_hand_events(
    ball_x: Sequence[float],
    ball_y: Sequence[float],
    wrist_x: Sequence[float],
    wrist_y: Sequence[float],
    fps: float,
) -> list[ReleaseEvent]:
    events: list[ReleaseEvent] = []
    armed = False
    previous_distance = math.nan
    for index in range(len(ball_x)):
        values = (ball_x[index], ball_y[index], wrist_x[index], wrist_y[index])
        if not all(math.isfinite(float(value)) for value in values):
            continue
        distance = math.dist((ball_x[index], ball_y[index]), (wrist_x[index], wrist_y[index]))
        if distance <= 0.06:
            armed = True
        if index > 0 and armed and distance >= 0.08 and math.isfinite(previous_distance):
            upward = float(ball_y[index]) < float(ball_y[index - 1]) - 0.008
            separating = distance > previous_distance + 0.015
            if upward and separating:
                confidence = min(0.98, 0.68 + (distance - 0.08) * 1.5)
                events.append(
                    ReleaseEvent(index, index / max(fps, 1.0), "ball_hand_separation", confidence)
                )
                armed = False
        previous_distance = distance
    return events


def _wrist_events(
    wrist_y: Sequence[float], knee_angles: Sequence[float], fps: float
) -> list[ReleaseEvent]:
    wrist = _interpolated(wrist_y)
    knee = _interpolated(knee_angles)
    if len(wrist) < 5 or not np.isfinite(wrist).any() or not np.isfinite(knee).any():
        return []
    window = max(2, int(round(fps * 0.45)))
    min_gap = max(3, int(round(fps * 0.8)))
    knee_candidates: list[int] = []
    for index in range(window, len(knee) - window):
        local = knee[index - window : index + window + 1]
        if knee[index] <= np.min(local) + 1.5:
            if not knee_candidates or index - knee_candidates[-1] >= min_gap:
                knee_candidates.append(index)
            elif knee[index] < knee[knee_candidates[-1]]:
                knee_candidates[-1] = index
    velocity = np.gradient(wrist) * max(fps, 1.0)
    events: list[ReleaseEvent] = []
    for lowest in knee_candidates:
        stop = min(len(wrist), lowest + max(3, int(round(fps * 0.9))))
        if stop <= lowest + 1:
            continue
        preparation_start = max(0, lowest - max(3, int(round(fps * 0.6))))
        knee_cycle = knee[preparation_start:stop]
        knee_range = float(np.max(knee_cycle) - np.min(knee_cycle))
        extension_gain = float(np.max(knee[lowest:stop]) - knee[lowest])
        if knee_range < 12 or knee[lowest] > 150 or extension_gain < 8:
            continue
        segment = velocity[lowest + 1 : stop]
        release = lowest + 1 + int(np.argmin(segment))
        if velocity[release] < -0.08:
            confidence = min(0.72, 0.45 + abs(float(velocity[release])) * 0.08)
            events.append(ReleaseEvent(release, release / fps, "wrist_velocity", confidence))
    return events


def _has_shot_knee_cycle(
    knee_angles: Sequence[float], frame_index: int, fps: float
) -> bool:
    """Require a load-and-extend knee cycle around a candidate release."""
    knee = _interpolated(knee_angles)
    if not np.isfinite(knee).any():
        return False
    before = max(0, frame_index - max(3, int(round(fps * 0.7))))
    after = min(len(knee), frame_index + max(3, int(round(fps * 0.55))) + 1)
    segment = knee[before:after]
    if len(segment) < 4:
        return False
    lowest_local = int(np.argmin(segment))
    lowest = before + lowest_local
    knee_range = float(np.max(segment) - np.min(segment))
    extension_gain = float(np.max(knee[lowest:after]) - knee[lowest])
    release_after_load = frame_index >= lowest - max(1, int(round(fps * 0.15)))
    return bool(knee[lowest] <= 150 and knee_range >= 12 and extension_gain >= 8 and release_after_load)


def is_ball_track_reliable(
    ball_x: Sequence[float],
    ball_y: Sequence[float],
    wrist_x: Sequence[float],
    wrist_y: Sequence[float],
    release_events: Sequence[ReleaseEvent],
    fps: float,
) -> bool:
    """Validate a ball track against the physical release sequence.

    A usable track must pass close to the shooting wrist and then move both
    upward and away from it. This deliberately prefers missing data over a
    visually convincing but false trajectory.
    """
    lengths = {len(ball_x), len(ball_y), len(wrist_x), len(wrist_y)}
    if len(lengths) != 1 or not release_events:
        return False
    total = len(ball_x)
    for event in release_events:
        start = max(0, event.frame_index - max(2, int(round(fps * 0.45))))
        end = min(total, event.frame_index + max(3, int(round(fps * 0.55))) + 1)
        samples: list[tuple[int, float, float, float]] = []
        for index in range(start, end):
            values = (ball_x[index], ball_y[index], wrist_x[index], wrist_y[index])
            if all(math.isfinite(float(value)) for value in values):
                distance = math.dist(
                    (float(ball_x[index]), float(ball_y[index])),
                    (float(wrist_x[index]), float(wrist_y[index])),
                )
                samples.append((index, float(ball_y[index]), distance, float(ball_x[index])))
        if len(samples) < 4:
            continue
        near = [sample for sample in samples if sample[0] <= event.frame_index + 1 and sample[2] <= 0.07]
        after = [sample for sample in samples if sample[0] >= event.frame_index]
        if not near or len(after) < 3:
            continue
        origin = min(near, key=lambda sample: sample[2])
        upward_gain = origin[1] - min(sample[1] for sample in after)
        separation_gain = max(sample[2] for sample in after) - origin[2]
        if upward_gain >= 0.04 and separation_gain >= 0.08:
            return True
    return False


def _deduplicate(events: Sequence[ReleaseEvent], fps: float) -> list[ReleaseEvent]:
    if not events:
        return []
    minimum_gap = max(1, int(round(fps * 0.8)))
    priority = {"ball_hand_separation": 1, "wrist_velocity": 0}
    ordered = sorted(events, key=lambda item: item.frame_index)
    groups: list[list[ReleaseEvent]] = [[ordered[0]]]
    for event in ordered[1:]:
        if event.frame_index - groups[-1][-1].frame_index < minimum_gap:
            groups[-1].append(event)
        else:
            groups.append([event])
    return [
        max(group, key=lambda item: (priority.get(item.source, 0), item.confidence))
        for group in groups
    ]


def detect_release_events(
    ball_x: Sequence[float],
    ball_y: Sequence[float],
    wrist_x: Sequence[float],
    wrist_y: Sequence[float],
    knee_angles: Sequence[float],
    fps: float,
) -> list[ReleaseEvent]:
    lengths = {len(ball_x), len(ball_y), len(wrist_x), len(wrist_y), len(knee_angles)}
    if len(lengths) != 1:
        raise ValueError("出手检测的所有时序必须长度一致")
    events = [
        event
        for event in _ball_hand_events(ball_x, ball_y, wrist_x, wrist_y, fps)
        if _has_shot_knee_cycle(knee_angles, event.frame_index, fps)
    ]
    events.extend(_wrist_events(wrist_y, knee_angles, fps))
    return _deduplicate(events, fps)


def build_shot_windows(
    events: Sequence[ReleaseEvent], total_frames: int, fps: float
) -> list[ShotWindow]:
    if total_frames <= 0 or not events:
        return []
    ordered = sorted(events, key=lambda item: item.frame_index)
    pre = max(1, int(round(fps * 1.2)))
    post = max(1, int(round(fps * 0.9)))
    starts = [max(0, event.frame_index - pre) for event in ordered]
    ends = [min(total_frames - 1, event.frame_index + post) for event in ordered]
    for index in range(len(ordered) - 1):
        if ends[index] >= starts[index + 1]:
            midpoint = (ordered[index].frame_index + ordered[index + 1].frame_index) // 2
            ends[index] = max(ordered[index].frame_index, midpoint - 1)
            starts[index + 1] = min(ordered[index + 1].frame_index, midpoint + 1)
    return [
        ShotWindow(
            shot_id=index + 1,
            start_frame=starts[index],
            release_frame=event.frame_index,
            end_frame=ends[index],
            release_source=event.source,
            confidence=event.confidence,
        )
        for index, event in enumerate(ordered)
    ]
