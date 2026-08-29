"""MediaPipe Pose Landmarker 视频分析管线。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
from pathlib import Path
import shutil
import tempfile
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from biomechanics import (
    AnalysisPhases,
    LandmarkPoint,
    build_training_feedback,
    calculate_angle,
    estimate_phases,
    interpolate_short_gaps,
)
from video_intelligence import (
    CaptureQuality,
    ReleaseEvent,
    ShotWindow,
    assess_capture_quality,
    build_shot_windows,
    detect_release_events,
    is_ball_track_reliable,
    validated_wrist_snap,
)
from vision_models import HandMeasurement, VisionModels, person_bbox_from_landmarks
from shot_evaluation import (
    DiagnosticResult,
    TrainingDrill,
    ProblemFinding,
    build_angle_comparison,
    build_diagnostics,
    build_stability_scores,
    build_timing_analysis,
    estimate_2d_ball_launch_angle,
)


ProgressCallback = Callable[[float, str], None]


class AnalysisError(RuntimeError):
    """可直接展示给用户的分析错误。"""


@dataclass(frozen=True)
class VideoInfo:
    filename: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float


@dataclass
class ShotAnalysis:
    shot_id: int
    window: ShotWindow
    phases: AnalysisPhases
    stats: pd.DataFrame
    metrics: dict[str, float]
    summary: list[str]
    suggestions: list[str]
    keyframes: dict[str, Path]
    clip_path: Path | None = None
    timeline_frames: dict[str, int] | None = None
    timing: dict[str, float] | None = None
    angle_comparison: pd.DataFrame | None = None
    problems: list[ProblemFinding] | None = None
    action_suggestions: list[str] | None = None
    training_drills: list[TrainingDrill] | None = None
    stability_scores: dict[str, int | None] | None = None
    hand_analysis: str = ""


@dataclass
class AnalysisResult:
    output_dir: Path
    processed_video: Path
    csv_path: Path
    keyframes: dict[str, Path]
    data: pd.DataFrame
    info: VideoInfo
    phases: AnalysisPhases
    stats: pd.DataFrame
    metrics: dict[str, float]
    summary: list[str]
    suggestions: list[str]
    valid_frames: int
    shooting_hand: str
    trajectory_video: Path
    shots: list[ShotAnalysis]
    capture_quality: CaptureQuality
    release_points: list[tuple[float, float]]
    consistency: dict[str, float]
    model_notes: list[str]
    stick_figure_video: Path
    shot_summary_path: Path
    ball_trajectory_path: Path | None
    science_boundaries: dict[str, list[str]]
    original_video: Path


LANDMARK_INDEX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

SKELETON_NAMES = (
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)

SKELETON_CONNECTIONS = (
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
)

ANGLE_COLUMNS = ("elbow_angle", "knee_angle", "hip_angle", "trunk_angle")
POSE_COORD_COLUMNS = tuple(
    f"{name}_{axis}" for name in LANDMARK_INDEX for axis in ("x", "y")
)
POSE_DERIVED_COLUMNS = (
    "elbow_alignment",
    "forearm_direction",
    "shoulder_height_difference",
    "shooting_shoulder_lift",
    "wrist_relative_shoulder",
)
PHASE_LABELS = {
    "ready": "准备",
    "dip": "下沉",
    "lowest": "最低点",
    "upward_drive": "向上发力",
    "set_point": "抬球关键位置",
    "release": "出手附近",
    "release_candidate": "出手候选",
    "followthrough": "随挥",
    "landing": "落地/结束",
}
KEYFRAME_OVERLAY_LABELS = {
    "ready": "READY",
    "lowest": "LOWEST",
    "release": "RELEASE",
    "release_candidate": "RELEASE CANDIDATE",
    "set_point": "SET POINT",
    "followthrough": "FOLLOWTHROUGH",
}


def write_image(path: str | Path, image: np.ndarray) -> None:
    """通过内存编码写图片，绕过 Windows OpenCV 对中文路径支持不完整的问题。"""
    output_path = Path(path)
    extension = output_path.suffix.lower() or ".jpg"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise AnalysisError(f"无法编码关键帧图片：{output_path.name}")
    try:
        output_path.write_bytes(encoded.tobytes())
    except OSError as error:
        raise AnalysisError(f"无法写入关键帧图片：{output_path}") from error


def ascii_safe_model_path(model_path: str | Path) -> Path:
    """MediaPipe Windows 原生层无法稳定读取中文路径，必要时复制到 ASCII 缓存。"""
    source = Path(model_path).resolve()
    if not source.exists():
        raise AnalysisError(f"姿态模型不存在：{source}")
    if str(source).isascii():
        return source
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "ai_shot_analyzer_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"pose_landmarker_lite_{digest}.task"
    if not cached.exists() or cached.stat().st_size != source.stat().st_size:
        shutil.copy2(source, cached)
    return cached


def read_video_info(video_path: str | Path) -> VideoInfo:
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AnalysisError("无法打开视频，请检查格式或文件是否损坏。")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise AnalysisError("视频元数据不完整，无法确定 FPS、帧数或分辨率。")
    return VideoInfo(path.name, fps, frame_count, width, height, frame_count / fps)


def create_output_dir(root: str | Path = "outputs") -> Path:
    output_root = Path(root)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_root / f"analysis_{stamp}"
    output_dir.mkdir(parents=False, exist_ok=False)
    return output_dir


def _point(landmarks: list, index: int, threshold: float) -> LandmarkPoint | None:
    landmark = landmarks[index]
    visibility = float(getattr(landmark, "visibility", 1.0))
    presence = float(getattr(landmark, "presence", 1.0))
    if visibility < threshold or presence < threshold:
        return None
    return LandmarkPoint(float(landmark.x), float(landmark.y), float(landmark.z), visibility)


def _angle(*points: LandmarkPoint | None) -> float:
    if any(point is None for point in points):
        return math.nan
    return calculate_angle(points[0], points[1], points[2])  # type: ignore[arg-type]


def _trunk_angle(
    left_shoulder: LandmarkPoint | None,
    right_shoulder: LandmarkPoint | None,
    left_hip: LandmarkPoint | None,
    right_hip: LandmarkPoint | None,
) -> float:
    if any(point is None for point in (left_shoulder, right_shoulder, left_hip, right_hip)):
        return math.nan
    shoulder_x = (left_shoulder.x + right_shoulder.x) / 2  # type: ignore[union-attr]
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2  # type: ignore[union-attr]
    hip_x = (left_hip.x + right_hip.x) / 2  # type: ignore[union-attr]
    hip_y = (left_hip.y + right_hip.y) / 2  # type: ignore[union-attr]
    # 画面向上为 0°；正负表示画面中的左右倾斜，不等同于真实前后倾。
    return float(np.degrees(np.arctan2(shoulder_x - hip_x, hip_y - shoulder_y)))


def _measure_frame(landmarks: list, hand: str, threshold: float) -> dict[str, float]:
    side = "right" if hand == "右手" else "left"
    points = {
        name: _point(landmarks, index, threshold)
        for name, index in LANDMARK_INDEX.items()
    }
    shoulder = points[f"{side}_shoulder"]
    elbow = points[f"{side}_elbow"]
    wrist = points[f"{side}_wrist"]
    hip = points[f"{side}_hip"]
    knee = points[f"{side}_knee"]
    ankle = points[f"{side}_ankle"]
    required = (shoulder, elbow, wrist, hip, knee, ankle)
    confidence_values = [point.visibility for point in required if point is not None]
    result = {
        "elbow_angle": _angle(shoulder, elbow, wrist),
        "knee_angle": _angle(hip, knee, ankle),
        "hip_angle": _angle(shoulder, hip, knee),
        "trunk_angle": _trunk_angle(
            points["left_shoulder"], points["right_shoulder"],
            points["left_hip"], points["right_hip"],
        ),
        "wrist_x": wrist.x if wrist else math.nan,
        "wrist_y": wrist.y if wrist else math.nan,
        "confidence": min(confidence_values) if len(confidence_values) == 6 else 0.0,
    }
    for name, point in points.items():
        result[f"{name}_x"] = point.x if point else math.nan
        result[f"{name}_y"] = point.y if point else math.nan
        result[f"{name}_visibility"] = point.visibility if point else 0.0

    opposite = points["left_shoulder" if side == "right" else "right_shoulder"]
    left_shoulder, right_shoulder = points["left_shoulder"], points["right_shoulder"]
    if shoulder and elbow and wrist and hip:
        line = np.asarray((wrist.x - shoulder.x, wrist.y - shoulder.y), dtype=float)
        elbow_delta = np.asarray((elbow.x - shoulder.x, elbow.y - shoulder.y), dtype=float)
        line_length = float(np.linalg.norm(line))
        torso_length = math.dist((shoulder.x, shoulder.y), (hip.x, hip.y))
        cross = abs(line[0] * elbow_delta[1] - line[1] * elbow_delta[0])
        result["elbow_alignment"] = cross / max(line_length * torso_length, 1e-6)
        result["forearm_direction"] = float(np.degrees(np.arctan2(-(wrist.y - elbow.y), wrist.x - elbow.x)))
    else:
        result["elbow_alignment"] = math.nan
        result["forearm_direction"] = math.nan
    result["shoulder_height_difference"] = (
        abs(left_shoulder.y - right_shoulder.y)
        if left_shoulder and right_shoulder else math.nan
    )
    result["shooting_shoulder_lift"] = (
        opposite.y - shoulder.y if opposite and shoulder else math.nan
    )
    result["wrist_relative_shoulder"] = (
        shoulder.y - wrist.y if shoulder and wrist else math.nan
    )
    return result


def _pose_joint_xy(
    landmarks: list | None, name: str, threshold: float
) -> tuple[float, float] | None:
    if not landmarks:
        return None
    point = _point(landmarks, LANDMARK_INDEX[name], threshold)
    return (point.x, point.y) if point else None


def _pose_quality_values(landmarks: list) -> tuple[bool, float, float]:
    bbox = person_bbox_from_landmarks(landmarks, padding=0.0)
    if bbox is None:
        return False, math.nan, math.nan
    x1, y1, x2, y2 = bbox
    body_height = max(y2 - y1, 1e-6)
    required = (11, 12, 15, 16, 27, 28)
    full_body = all(
        float(getattr(landmarks[index], "visibility", 1.0)) >= 0.45
        and 0.01 <= landmarks[index].x <= 0.99
        and 0.01 <= landmarks[index].y <= 0.99
        for index in required
    )
    shoulder_span = abs(float(landmarks[11].x) - float(landmarks[12].x))
    hip_span = abs(float(landmarks[23].x) - float(landmarks[24].x))
    width_ratio = ((shoulder_span + hip_span) / 2) / body_height
    side_view_score = float(np.clip(1.0 - width_ratio / 0.42, 0.0, 1.0))
    return full_body, body_height, side_view_score


def _draw_pose(
    frame: np.ndarray,
    landmarks: list | None,
    measurements: dict[str, float],
    hand_points: list[tuple[float, float]] | None = None,
    ball_trail: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    if landmarks:
        for start, end in SKELETON_CONNECTIONS:
            a, b = landmarks[start], landmarks[end]
            if min(float(getattr(a, "visibility", 1)), float(getattr(b, "visibility", 1))) >= 0.35:
                pt_a = (int(a.x * width), int(a.y * height))
                pt_b = (int(b.x * width), int(b.y * height))
                cv2.line(canvas, pt_a, pt_b, (64, 220, 120), 3, cv2.LINE_AA)
        for index in LANDMARK_INDEX.values():
            item = landmarks[index]
            if float(getattr(item, "visibility", 1)) >= 0.35:
                cv2.circle(canvas, (int(item.x * width), int(item.y * height)), 5, (40, 170, 255), -1)

    if hand_points:
        for x, y in hand_points:
            cv2.circle(canvas, (int(x * width), int(y * height)), 2, (255, 100, 210), -1)
    if ball_trail:
        pixels = [(int(x * width), int(y * height)) for x, y in ball_trail]
        for start, end in zip(pixels, pixels[1:]):
            cv2.line(canvas, start, end, (0, 210, 255), 3, cv2.LINE_AA)
    ball_x = measurements.get("ball_x", math.nan)
    ball_y = measurements.get("ball_y", math.nan)
    if math.isfinite(ball_x) and math.isfinite(ball_y):
        center = (int(ball_x * width), int(ball_y * height))
        cv2.circle(canvas, center, 13, (0, 90, 255), 3, cv2.LINE_AA)

    labels = (
        ("Elbow", measurements.get("elbow_angle", math.nan)),
        ("Knee", measurements.get("knee_angle", math.nan)),
        ("Hip", measurements.get("hip_angle", math.nan)),
        ("Trunk", measurements.get("trunk_angle", math.nan)),
        ("Wrist", measurements.get("hand_wrist_angle", math.nan)),
    )
    overlay = canvas.copy()
    cv2.rectangle(overlay, (12, 12), (250, 164), (15, 20, 30), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    for row, (name, value) in enumerate(labels):
        text = f"{name}: {value:.1f} deg" if math.isfinite(value) else f"{name}: --"
        cv2.putText(canvas, text, (24, 40 + row * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (245, 245, 245), 2, cv2.LINE_AA)
    return canvas


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise AnalysisError("无法创建分析视频，当前 OpenCV 缺少可用的 MP4 编码器。")
    return writer


def _fill_short_gaps(data: pd.DataFrame, fps: float) -> pd.DataFrame:
    filled = data.copy()
    max_gap = max(1, min(5, int(round(fps * 0.12))))
    for column in (*ANGLE_COLUMNS, "wrist_x", "wrist_y", *POSE_COORD_COLUMNS, *POSE_DERIVED_COLUMNS):
        if column not in filled:
            continue
        filled[column] = interpolate_short_gaps(filled[column].tolist(), max_gap=max_gap)
    for column in ("hand_wrist_angle", "hand_finger_direction", "ball_x", "ball_y"):
        if column in filled:
            filled[column] = interpolate_short_gaps(filled[column].tolist(), max_gap=2)
    return filled


def _safe_stat(series: pd.Series, operation: str) -> float:
    valid = series.dropna()
    if valid.empty:
        return math.nan
    return float(getattr(valid, operation)())


def _build_stats(data: pd.DataFrame, phases: AnalysisPhases) -> pd.DataFrame:
    release_slice = data.iloc[phases.lowest : phases.followthrough + 1]
    rows = []
    names = {
        "elbow_angle": "肘角",
        "knee_angle": "膝角",
        "hip_angle": "髋角",
        "trunk_angle": "躯干倾角",
    }
    for column, label in names.items():
        values = release_slice[column]
        rows.append({
            "指标": label,
            "平均值": _safe_stat(values, "mean"),
            "最小值": _safe_stat(values, "min"),
            "最大值": _safe_stat(values, "max"),
            "标准差": _safe_stat(values, "std"),
        })
    return pd.DataFrame(rows)


def _finite_at(data: pd.DataFrame, column: str, index: int) -> float:
    value = float(data.iloc[index][column])
    return value if math.isfinite(value) else math.nan


def _path_deviation(data: pd.DataFrame, start: int, stop: int) -> float:
    points = data.iloc[start : stop + 1][["wrist_x", "wrist_y"]].dropna().to_numpy()
    if len(points) < 3:
        return math.nan
    origin, destination = points[0], points[-1]
    vector = destination - origin
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        return float(np.std(points[:, 0]))
    deltas = points - origin
    # NumPy 2.5 不再接受 2D 向量的 np.cross，显式计算二维标量叉积。
    cross_values = vector[0] * deltas[:, 1] - vector[1] * deltas[:, 0]
    distances = np.abs(cross_values / length)
    return float(np.sqrt(np.mean(np.square(distances))))


def _finite_near(data: pd.DataFrame, column: str, index: int, radius: int = 2) -> float:
    if column not in data:
        return math.nan
    start, stop = max(0, index - radius), min(len(data), index + radius + 1)
    values = data.iloc[start:stop][column].dropna()
    if values.empty:
        return math.nan
    nearest = min(values.index, key=lambda item: abs(int(item) - index))
    return float(values.loc[nearest])


def _cycle_smoothness(series: pd.Series) -> float:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < 4:
        return math.nan
    path = float(np.sum(np.abs(np.diff(values))))
    useful_range = float(np.max(values) - np.min(values))
    return float(np.clip(useful_range / max(path, 1e-6), 0.0, 1.0))


def _build_metrics(
    data: pd.DataFrame,
    phases: AnalysisPhases,
    valid_frames: int,
    release_confidence: float = math.nan,
    frame_count: int | None = None,
) -> dict[str, float]:
    motion = data.iloc[phases.ready : phases.followthrough + 1]
    extension = data.iloc[phases.lowest : phases.release + 1]
    knee_valid = motion["knee_angle"].dropna()
    metrics = {
        "valid_ratio": valid_frames / max(frame_count if frame_count is not None else len(data), 1),
        "elbow_release": _finite_at(data, "elbow_angle", phases.release),
        "elbow_min": _safe_stat(motion["elbow_angle"], "min"),
        "elbow_max": _safe_stat(motion["elbow_angle"], "max"),
        "elbow_extension_change": (
            _safe_stat(extension["elbow_angle"], "max") - _safe_stat(extension["elbow_angle"], "min")
        ),
        "elbow_extension_std": _safe_stat(extension["elbow_angle"], "std"),
        "knee_lowest": _finite_at(data, "knee_angle", phases.lowest),
        "knee_ready": _finite_at(data, "knee_angle", phases.ready),
        "knee_release": _finite_at(data, "knee_angle", phases.release),
        "knee_motion_range": (
            float(knee_valid.max() - knee_valid.min()) if not knee_valid.empty else math.nan
        ),
        "hip_lowest": _finite_at(data, "hip_angle", phases.lowest),
        "hip_ready": _finite_at(data, "hip_angle", phases.ready),
        "hip_release": _finite_at(data, "hip_angle", phases.release),
        "hip_motion_std": _safe_stat(motion["hip_angle"], "std"),
        "trunk_max_abs": _safe_stat(motion["trunk_angle"].abs(), "max"),
        "trunk_std": _safe_stat(motion["trunk_angle"], "std"),
        "trunk_ready": _finite_at(data, "trunk_angle", phases.ready),
        "trunk_release": _finite_at(data, "trunk_angle", phases.release),
        "elbow_alignment": _finite_near(data, "elbow_alignment", phases.release, 3),
        "forearm_direction_release": _finite_near(data, "forearm_direction", phases.release, 3),
        "shoulder_height_difference": _finite_near(data, "shoulder_height_difference", phases.release, 3),
        "shooting_shoulder_lift": _finite_near(data, "shooting_shoulder_lift", phases.release, 3),
        "wrist_release_relative_shoulder": _finite_near(data, "wrist_relative_shoulder", phases.release, 3),
        "knee_cycle_smoothness": _cycle_smoothness(motion["knee_angle"]),
        "elbow_cycle_smoothness": _cycle_smoothness(extension["elbow_angle"]),
        "wrist_path_deviation": _path_deviation(data, phases.lowest, phases.followthrough),
        "hand_release_angle": _finite_near(data, "hand_wrist_angle", phases.release, 3),
        "hand_followthrough_angle": _finite_near(data, "hand_wrist_angle", phases.followthrough, 4),
        "finger_direction_release": _finite_near(data, "hand_finger_direction", phases.release, 3),
        "finger_direction_followthrough": _finite_near(data, "hand_finger_direction", phases.followthrough, 4),
        "ball_release_x": _finite_near(data, "ball_x", phases.release, 2),
        "ball_release_y": _finite_near(data, "ball_y", phases.release, 2),
        "release_confidence": float(release_confidence),
    }
    release_angle = metrics["hand_release_angle"]
    follow_angle = metrics["hand_followthrough_angle"]
    metrics["wrist_snap_change"] = validated_wrist_snap(
        release_angle,
        follow_angle,
        _finite_near(data, "hand_confidence", phases.release, 3),
        _finite_near(data, "hand_confidence", phases.followthrough, 4),
    )
    return metrics


def _phases_for_window(data: pd.DataFrame, window: ShotWindow, fps: float) -> AnalysisPhases:
    before_release = data.iloc[window.start_frame : window.release_frame + 1]["knee_angle"].dropna()
    if before_release.empty:
        lowest = max(window.start_frame, window.release_frame - max(1, int(fps * 0.25)))
    else:
        lowest = int(before_release.idxmin())
    ready = max(window.start_frame, lowest - max(1, int(round(fps * 0.35))))
    release = min(window.end_frame, max(lowest, window.release_frame))
    followthrough = min(window.end_frame, release + max(1, int(round(fps * 0.3))))
    return AnalysisPhases(ready, lowest, release, followthrough)


def _timeline_frames(
    phases: AnalysisPhases, window: ShotWindow, fps: float
) -> dict[str, int]:
    dip = phases.ready + max(1, (phases.lowest - phases.ready) // 2)
    upward = min(phases.release, phases.lowest + max(1, int(round(fps * 0.08))))
    set_point = max(upward, phases.release - max(1, int(round(fps * 0.16))))
    result = {
        "ready": phases.ready,
        "dip": min(dip, phases.lowest),
        "lowest": phases.lowest,
        "upward_drive": upward,
        "set_point": set_point,
        "release_candidate": phases.release,
        "followthrough": phases.followthrough,
    }
    landing = phases.followthrough + max(1, int(round(fps * 0.2)))
    if landing <= window.end_frame:
        result["landing"] = landing
    return result


def _stage_at_frame(frame_index: int, timeline: dict[str, int]) -> str:
    ordered = sorted(timeline.items(), key=lambda item: item[1])
    current = ""
    for name, start in ordered:
        if frame_index >= start:
            current = name
        else:
            break
    return current


def _row_point(
    row: pd.Series,
    name: str,
    width: int,
    height: int,
    transform: Callable[[float, float], tuple[float, float]] | None = None,
) -> tuple[int, int] | None:
    x, y = float(row.get(f"{name}_x", math.nan)), float(row.get(f"{name}_y", math.nan))
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    if transform:
        x, y = transform(x, y)
    return int(x * width), int(y * height)


def _draw_reconstructed_pose(
    canvas: np.ndarray,
    row: pd.Series,
    shooting_hand: str,
    stage: str,
    *,
    transform: Callable[[float, float], tuple[float, float]] | None = None,
    stick_only: bool = False,
) -> np.ndarray:
    height, width = canvas.shape[:2]
    color = (65, 235, 150) if not stick_only else (70, 220, 255)
    points = {
        name: _row_point(row, name, width, height, transform)
        for name in LANDMARK_INDEX
    }
    for start, end in SKELETON_NAMES:
        if points[start] and points[end]:
            cv2.line(canvas, points[start], points[end], color, 4 if stick_only else 3, cv2.LINE_AA)
    for name, point in points.items():
        if point:
            cv2.circle(canvas, point, 6 if stick_only else 5, (50, 140, 255), -1, cv2.LINE_AA)

    side = "right" if shooting_hand == "右手" else "left"
    labels = (("shoulder", "S"), ("elbow", "E"), ("wrist", "W"), ("hip", "H"), ("knee", "K"), ("ankle", "A"))
    for joint, label in labels:
        point = points.get(f"{side}_{joint}")
        if point:
            cv2.putText(canvas, label, (point[0] + 7, point[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)

    panel_width = min(330, max(250, width // 3))
    overlay = canvas.copy()
    cv2.rectangle(overlay, (12, 12), (panel_width, 190), (10, 16, 27), -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
    stage_text = KEYFRAME_OVERLAY_LABELS.get(stage, stage.replace("_", " ").upper()) if stage else "OUTSIDE SHOT"
    cv2.putText(canvas, stage_text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (50, 220, 255), 2, cv2.LINE_AA)
    for index, (label, column) in enumerate((("Elbow", "elbow_angle"), ("Knee", "knee_angle"), ("Hip", "hip_angle"), ("Trunk", "trunk_angle"))):
        value = float(row.get(column, math.nan))
        text = f"{label}: {value:.1f} deg" if math.isfinite(value) else f"{label}: --"
        cv2.putText(canvas, text, (24, 72 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2, cv2.LINE_AA)
    return canvas


def _stick_transform(data: pd.DataFrame) -> Callable[[float, float], tuple[float, float]]:
    x_values = pd.concat([data[column] for column in POSE_COORD_COLUMNS if column.endswith("_x")]).dropna()
    y_values = pd.concat([data[column] for column in POSE_COORD_COLUMNS if column.endswith("_y")]).dropna()
    if x_values.empty or y_values.empty:
        return lambda x, y: (x, y)
    x1, x2 = float(x_values.quantile(0.01)), float(x_values.quantile(0.99))
    y1, y2 = float(y_values.quantile(0.01)), float(y_values.quantile(0.99))
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    scale = min(0.72 / max(x2 - x1, 0.05), 0.82 / max(y2 - y1, 0.05))
    return lambda x, y: ((x - center_x) * scale + 0.56, (y - center_y) * scale + 0.52)


def _create_analysis_videos(
    source: Path,
    overlay_path: Path,
    stick_path: Path,
    data: pd.DataFrame,
    info: VideoInfo,
    shooting_hand: str,
) -> tuple[Path, Path]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise AnalysisError("无法重新读取原视频以生成骨架和火柴人视频。")
    overlay_writer = _open_writer(overlay_path, info.fps, (info.width, info.height))
    stick_writer = _open_writer(stick_path, info.fps, (info.width, info.height))
    transform = _stick_transform(data)
    try:
        frame_index = 0
        while frame_index < len(data):
            ok, frame = capture.read()
            if not ok:
                break
            row = data.iloc[frame_index]
            stage = str(row.get("stage_name", ""))
            overlay_writer.write(_draw_reconstructed_pose(frame.copy(), row, shooting_hand, stage))
            dark = np.full((info.height, info.width, 3), (18, 24, 36), dtype=np.uint8)
            stick_writer.write(_draw_reconstructed_pose(dark, row, shooting_hand, stage, transform=transform, stick_only=True))
            frame_index += 1
    finally:
        capture.release()
        overlay_writer.release()
        stick_writer.release()
    return overlay_path, stick_path


def _enhance_feedback(
    metrics: dict[str, float],
    summary: list[str],
    suggestions: list[str],
    release_source: str,
) -> tuple[list[str], list[str]]:
    summary = list(summary)
    suggestions = list(suggestions)
    snap = metrics.get("wrist_snap_change", math.nan)
    hand_release = metrics.get("hand_release_angle", math.nan)
    finger = metrics.get("finger_direction_followthrough", math.nan)
    if math.isfinite(hand_release) and math.isfinite(snap):
        if snap >= 8:
            summary.append("出手到随挥阶段能观察到较清晰的屈腕变化。")
            suggestions.append("保持伸肘之后自然屈腕，让中指/食指朝目标方向延伸，不要提前压腕。")
        elif snap < 3:
            summary.append("当前二维画面中出手后的翻腕变化偏小，或手部关键点受模糊影响。")
            suggestions.append("可做近筐单手投篮：先完成伸肘，再自然屈腕，并短暂停留随挥姿势。")
        else:
            summary.append("当前视频能观察到翻腕动作，但幅度仍需结合多次投篮比较。")
            suggestions.append("重点保持每次伸肘—屈腕的先后顺序一致，不追求固定职业球员角度。")
        if math.isfinite(finger) and abs(finger) > 25:
            summary.append("随挥末端中指方向在画面中偏向一侧，可能影响左右方向重复性。")
            suggestions.append("练习结束姿势定格 1 秒，观察中指是否稳定指向篮筐中心方向。")
    else:
        summary.append("当前视频手部细节不足，未输出具体翻腕角，避免伪精确结论。")
        suggestions.append("若要分析翻腕，请把相机靠近并使用 1080p/60fps，让投篮手至少约 60×60 像素。")
    if release_source == "ball_hand_separation":
        summary.append("出手时刻由篮球与投篮手分离轨迹辅助确定。")
    else:
        summary.append("篮球轨迹证据不足，出手时刻由手腕速度与下肢阶段规则估计。")
    return summary, suggestions


def _build_consistency(shots: list[ShotAnalysis]) -> dict[str, float]:
    keys = (
        "elbow_release",
        "knee_lowest",
        "hip_lowest",
        "trunk_max_abs",
        "wrist_snap_change",
        "finger_direction_followthrough",
    )
    result: dict[str, float] = {"shot_count": float(len(shots))}
    for key in keys:
        values = [shot.metrics.get(key, math.nan) for shot in shots]
        valid = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
        result[f"{key}_mean"] = float(np.mean(valid)) if valid.size else math.nan
        result[f"{key}_std"] = float(np.std(valid, ddof=1)) if valid.size >= 2 else math.nan
    return result


def _extract_keyframes(
    video_path: Path,
    output_dir: Path,
    data: pd.DataFrame,
    phases: AnalysisPhases | dict[str, int],
    filename_prefix: str = "keyframe",
) -> dict[str, Path]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise AnalysisError("分析完成但无法重新读取视频以提取关键帧。")
    paths: dict[str, Path] = {}
    phase_frames = phases.as_dict() if isinstance(phases, AnalysisPhases) else phases
    for phase_name, frame_index in phase_frames.items():
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        row = data.iloc[frame_index].to_dict()
        # 第二遍不重复跑模型，骨架已在分析视频；关键帧突出显示可靠的角度和阶段。
        annotated = _draw_pose(frame, None, row)
        cv2.putText(
            annotated,
            KEYFRAME_OVERLAY_LABELS.get(phase_name, phase_name.replace("_", " ").upper()),
            (24, annotated.shape[0] - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (50, 220, 255),
            2,
            cv2.LINE_AA,
        )
        path = output_dir / f"{filename_prefix}_{phase_name}.jpg"
        write_image(path, annotated)
        paths[phase_name] = path
    capture.release()
    return paths


def _create_trajectory_video(
    processed_video: Path,
    output_path: Path,
    data: pd.DataFrame,
    shots: list[ShotAnalysis],
    info: VideoInfo,
) -> Path:
    capture = cv2.VideoCapture(str(processed_video))
    if not capture.isOpened():
        raise AnalysisError("无法读取骨架视频以生成篮球轨迹视频。")
    writer = _open_writer(output_path, info.fps, (info.width, info.height))
    trail: list[tuple[float, float]] = []
    release_points = [
        (shot.shot_id, shot.phases.release, shot.metrics.get("ball_release_x", math.nan), shot.metrics.get("ball_release_y", math.nan))
        for shot in shots
    ]
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame_index >= len(data):
                break
            row = data.iloc[frame_index]
            ball_x, ball_y = float(row["ball_x"]), float(row["ball_y"])
            if math.isfinite(ball_x) and math.isfinite(ball_y):
                trail.append((ball_x, ball_y))
                trail = trail[-45:]
            pixels = [(int(x * info.width), int(y * info.height)) for x, y in trail]
            for start, end in zip(pixels, pixels[1:]):
                cv2.line(frame, start, end, (0, 210, 255), 4, cv2.LINE_AA)
            for shot_id, release_frame, x, y in release_points:
                if frame_index >= release_frame and math.isfinite(x) and math.isfinite(y):
                    point = (int(x * info.width), int(y * info.height))
                    cv2.drawMarker(frame, point, (40, 40, 255), cv2.MARKER_CROSS, 28, 4)
                    cv2.putText(frame, f"SHOT {shot_id} RELEASE", (point[0] + 12, max(28, point[1] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 255), 2, cv2.LINE_AA)
            active = next((shot for shot in shots if shot.window.start_frame <= frame_index <= shot.window.end_frame), None)
            if active:
                cv2.putText(frame, f"SHOT {active.shot_id}", (info.width - 170, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 255), 3, cv2.LINE_AA)
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    return output_path


def _extract_shot_clip(
    video_path: Path, output_path: Path, window: ShotWindow, info: VideoInfo
) -> Path:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise AnalysisError("无法读取轨迹视频以导出单次投篮片段。")
    writer = _open_writer(output_path, info.fps, (info.width, info.height))
    capture.set(cv2.CAP_PROP_POS_FRAMES, window.start_frame)
    try:
        for _ in range(window.start_frame, window.end_frame + 1):
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
    return output_path


def analyze_video(
    video_path: str | Path,
    model_path: str | Path,
    shooting_hand: str = "右手",
    output_root: str | Path = "outputs",
    confidence_threshold: float = 0.45,
    progress: ProgressCallback | None = None,
    hand_model_path: str | Path | None = None,
    ball_model_path: str | Path | None = None,
) -> AnalysisResult:
    """执行 V0.3 连续投篮分析；手/球模型不可用时自动降级。"""
    source = Path(video_path)
    model = Path(model_path)
    if not source.exists():
        raise AnalysisError("待分析视频不存在。")
    if not model.exists():
        raise AnalysisError("缺少 pose_landmarker_lite.task，请先放到项目根目录。")
    if shooting_hand not in {"右手", "左手"}:
        raise AnalysisError("投篮手必须选择右手或左手。")

    info = read_video_info(source)
    output_dir = create_output_dir(output_root)
    original_path = output_dir / f"original_video{source.suffix.lower()}"
    shutil.copy2(source, original_path)
    processed_path = output_dir / "pose_overlay_video.mp4"
    stick_path = output_dir / "stick_figure_video.mp4"
    trajectory_path = output_dir / "trajectory_video.mp4"
    csv_path = output_dir / "pose_data.csv"
    shot_summary_path = output_dir / "shot_summary.csv"
    capture = cv2.VideoCapture(str(source))
    writer = _open_writer(processed_path, info.fps, (info.width, info.height))

    native_model = ascii_safe_model_path(model)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(native_model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=confidence_threshold,
        min_pose_presence_confidence=confidence_threshold,
        min_tracking_confidence=confidence_threshold,
        output_segmentation_masks=False,
    )
    rows: list[dict[str, float | int]] = []
    valid_frames = 0
    brightness_samples: list[float] = []
    blur_samples: list[float] = []
    full_body_samples: list[float] = []
    body_height_samples: list[float] = []
    side_view_samples: list[float] = []
    previous_ball: tuple[float, float] | None = None
    missing_ball_frames = 0
    ball_trail: list[tuple[float, float]] = []
    hand_error: str | None = None
    ball_error: str | None = None
    pose_frame_errors = 0
    pose_last_error: str | None = None
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker, VisionModels(
            hand_model_path, ball_model_path
        ) as vision_models:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                measurements = {
                    name: math.nan
                    for name in (
                        *ANGLE_COLUMNS,
                        "wrist_x",
                        "wrist_y",
                        *POSE_COORD_COLUMNS,
                        *POSE_DERIVED_COLUMNS,
                        "hand_wrist_angle",
                        "hand_finger_direction",
                        "ball_x",
                        "ball_y",
                        "ball_score",
                    )
                }
                measurements["confidence"] = 0.0
                measurements["hand_confidence"] = 0.0
                measurements["ball_source"] = ""
                landmarks = None
                hand_measurement = HandMeasurement()
                person_bbox = None
                if frame_index % max(1, int(round(info.fps / 6))) == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightness_samples.append(float(np.mean(gray)))
                    blur_samples.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp_ms = int(round(frame_index * 1000 / info.fps))
                    result = landmarker.detect_for_video(image, timestamp_ms)
                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks[0]
                        measurements = _measure_frame(landmarks, shooting_hand, confidence_threshold)
                        measurements.update(
                            {
                                "hand_wrist_angle": math.nan,
                                "hand_finger_direction": math.nan,
                                "hand_confidence": 0.0,
                                "ball_x": math.nan,
                                "ball_y": math.nan,
                                "ball_score": math.nan,
                                "ball_source": "",
                            }
                        )
                        if all(math.isfinite(measurements[name]) for name in ANGLE_COLUMNS):
                            valid_frames += 1
                        full_body, body_height, side_view = _pose_quality_values(landmarks)
                        full_body_samples.append(float(full_body))
                        if math.isfinite(body_height):
                            body_height_samples.append(body_height)
                        if math.isfinite(side_view):
                            side_view_samples.append(side_view)
                except Exception as error:
                    # 单帧解码或推理异常不终止整段视频。
                    landmarks = None
                    pose_frame_errors += 1
                    pose_last_error = str(error)

                side = "right" if shooting_hand == "右手" else "left"
                pose_wrist = _pose_joint_xy(landmarks, f"{side}_wrist", confidence_threshold)
                pose_elbow = _pose_joint_xy(landmarks, f"{side}_elbow", confidence_threshold)
                if landmarks:
                    person_bbox = person_bbox_from_landmarks(landmarks)
                if frame_index % 2 == 0:
                    hand_measurement = vision_models.detect_hand(
                        frame, pose_wrist, pose_elbow, person_bbox
                    )
                    measurements["hand_wrist_angle"] = hand_measurement.wrist_angle
                    measurements["hand_finger_direction"] = hand_measurement.finger_direction
                    measurements["hand_confidence"] = hand_measurement.confidence

                ball = vision_models.detect_ball(frame, pose_wrist, previous_ball, frame_index)
                if ball is not None:
                    measurements["ball_x"] = ball.x
                    measurements["ball_y"] = ball.y
                    measurements["ball_score"] = ball.score
                    measurements["ball_source"] = ball.source
                    previous_ball = (ball.x, ball.y)
                    missing_ball_frames = 0
                    ball_trail.append(previous_ball)
                    ball_trail = ball_trail[-30:]
                else:
                    missing_ball_frames += 1
                    if missing_ball_frames > max(5, int(info.fps * 0.25)):
                        previous_ball = None
                        ball_trail = []

                row: dict[str, float | int | str] = {
                    "frame_index": frame_index,
                    "time": frame_index / info.fps,
                    **measurements,
                }
                rows.append(row)
                pose_overlay = dict(measurements)
                pose_overlay["ball_x"] = math.nan
                pose_overlay["ball_y"] = math.nan
                writer.write(
                    _draw_pose(
                        frame,
                        landmarks,
                        pose_overlay,
                        hand_points=hand_measurement.points,
                        ball_trail=None,
                    )
                )
                frame_index += 1
                if progress and (frame_index % max(1, int(info.fps / 2)) == 0 or frame_index == info.frame_count):
                    progress(min(frame_index / info.frame_count, 0.82), f"正在分析第 {frame_index}/{info.frame_count} 帧")
            hand_error = vision_models.hand_error
            ball_error = vision_models.ball_error
    finally:
        capture.release()
        writer.release()

    if not rows:
        raise AnalysisError("视频中没有可读取的画面。")
    minimum_valid = max(3, int(len(rows) * 0.08))
    if valid_frames < minimum_valid:
        raise AnalysisError(
            f"只获得 {valid_frames} 个完整有效姿态帧，无法形成可靠曲线。请使用全身清晰、单人、少遮挡的视频。"
        )

    data = _fill_short_gaps(pd.DataFrame(rows), info.fps)
    data["wrist_speed"] = np.sqrt(
        np.square(data["wrist_x"].diff()) + np.square(data["wrist_y"].diff())
    ) * info.fps
    release_events = detect_release_events(
        data["ball_x"].tolist(),
        data["ball_y"].tolist(),
        data["wrist_x"].tolist(),
        data["wrist_y"].tolist(),
        data["knee_angle"].tolist(),
        info.fps,
    )
    ball_track_reliable = is_ball_track_reliable(
        data["ball_x"].tolist(),
        data["ball_y"].tolist(),
        data["wrist_x"].tolist(),
        data["wrist_y"].tolist(),
        release_events,
        info.fps,
    )
    if not ball_track_reliable:
        data[["ball_x", "ball_y", "ball_score"]] = math.nan
        data["ball_source"] = ""
        release_events = detect_release_events(
            data["ball_x"].tolist(),
            data["ball_y"].tolist(),
            data["wrist_x"].tolist(),
            data["wrist_y"].tolist(),
            data["knee_angle"].tolist(),
            info.fps,
        )
    if not release_events:
        try:
            fallback_phases = estimate_phases(
                data["knee_angle"].tolist(), data["wrist_y"].tolist(), info.fps
            )
        except ValueError as error:
            raise AnalysisError(str(error)) from error
        release_events = [
            ReleaseEvent(
                fallback_phases.release,
                fallback_phases.release / info.fps,
                "wrist_velocity",
                0.4,
            )
        ]
    windows = build_shot_windows(release_events, len(data), info.fps)
    if progress:
        progress(0.86, f"识别到 {len(windows)} 次投篮，正在生成逐次结果")

    pose_valid_ratio = valid_frames / max(len(data), 1)
    capture_quality = assess_capture_quality(
        brightness=float(np.mean(brightness_samples)) if brightness_samples else math.nan,
        blur_score=float(np.mean(blur_samples)) if blur_samples else math.nan,
        pose_valid_ratio=pose_valid_ratio,
        full_body_ratio=float(np.mean(full_body_samples)) if full_body_samples else 0.0,
        body_height_ratio=float(np.mean(body_height_samples)) if body_height_samples else 0.0,
        side_view_score=float(np.mean(side_view_samples)) if side_view_samples else 0.0,
        resolution=(info.width, info.height),
    )

    shots: list[ShotAnalysis] = []
    phase_by_frame: dict[int, str] = {}
    data["shot_id"] = 0
    data["stage_name"] = ""
    for window in windows:
        phases_for_shot = _phases_for_window(data, window, info.fps)
        shot_valid = int(
            data.iloc[window.start_frame : window.end_frame + 1]["confidence"].gt(0).sum()
        )
        shot_metrics = _build_metrics(
            data,
            phases_for_shot,
            shot_valid,
            release_confidence=window.confidence,
            frame_count=window.end_frame - window.start_frame + 1,
        )
        shot_metrics["ball_launch_angle_2d"] = estimate_2d_ball_launch_angle(
            data["ball_x"].tolist(), data["ball_y"].tolist(), phases_for_shot.release
        )
        shot_stats = _build_stats(data, phases_for_shot)
        shot_summary, shot_suggestions = build_training_feedback(shot_metrics)
        shot_summary, shot_suggestions = _enhance_feedback(
            shot_metrics, shot_summary, shot_suggestions, window.release_source
        )
        timeline = _timeline_frames(phases_for_shot, window, info.fps)
        timing = build_timing_analysis(data, timeline, info.fps)
        hand_samples = data.iloc[window.start_frame : window.end_frame + 1]["hand_confidence"]
        hand_reliable = bool(
            hand_samples.ge(0.5).sum() >= 3
            and math.isfinite(shot_metrics.get("wrist_snap_change", math.nan))
        )
        diagnostic = build_diagnostics(shot_metrics, hand_reliable=hand_reliable)
        shots.append(
            ShotAnalysis(
                shot_id=window.shot_id,
                window=window,
                phases=phases_for_shot,
                stats=shot_stats,
                metrics=shot_metrics,
                summary=shot_summary,
                suggestions=shot_suggestions,
                keyframes={},
                timeline_frames=timeline,
                timing=timing,
                angle_comparison=build_angle_comparison(shot_metrics),
                problems=diagnostic.problems,
                action_suggestions=diagnostic.actions,
                training_drills=diagnostic.drills,
                stability_scores=build_stability_scores(shot_metrics),
                hand_analysis=diagnostic.hand_analysis,
            )
        )
        data.loc[window.start_frame : window.end_frame, "shot_id"] = window.shot_id
        for frame_index in range(window.start_frame, window.end_frame + 1):
            data.loc[frame_index, "stage_name"] = _stage_at_frame(frame_index, timeline)
        for phase_name, frame in phases_for_shot.as_dict().items():
            phase_by_frame[frame] = f"第{window.shot_id}次·{PHASE_LABELS[phase_name]}"

    data["phase_marker"] = data["frame_index"].map(phase_by_frame).fillna("")
    consistency = _build_consistency(shots)
    primary = shots[0]
    phases = primary.phases
    stats = primary.stats
    metrics = primary.metrics
    summary = list(primary.summary)
    suggestions = list(primary.suggestions)
    if len(shots) >= 2:
        elbow_std = consistency.get("elbow_release_std", math.nan)
        knee_std = consistency.get("knee_lowest_std", math.nan)
        summary.insert(0, f"本段视频自动切分出 {len(shots)} 次投篮，可进行跨次重复性比较。")
        if math.isfinite(elbow_std):
            summary.append(f"多次出手肘角标准差约 {elbow_std:.1f}°。")
            if elbow_std > 10:
                suggestions.append("优先固定抬球路径和伸肘时序，降低多次出手肘角波动。")
        if math.isfinite(knee_std) and knee_std > 10:
            suggestions.append("多次最低点膝角差异较大，可用固定脚步和固定下沉节奏练习。")
    else:
        summary.insert(0, "当前只识别到 1 次投篮，不能据此评价跨次动作重复性。")
    suggestions.extend(capture_quality.corrections[:3])

    data.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_rows = []
    for shot in shots:
        row: dict[str, float | int | str] = {
            "shot_id": shot.shot_id,
            "start_time": shot.window.start_frame / info.fps,
            "release_candidate_time": shot.window.release_frame / info.fps,
            "end_time": shot.window.end_frame / info.fps,
            "release_source": shot.window.release_source,
        }
        row.update(shot.metrics)
        if shot.timing:
            row.update({f"timing_{key}": value for key, value in shot.timing.items()})
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(shot_summary_path, index=False, encoding="utf-8-sig")
    ball_trajectory_path: Path | None = None
    if ball_track_reliable and data["ball_x"].notna().any():
        ball_trajectory_path = output_dir / "ball_trajectory.csv"
        data.loc[data["ball_x"].notna(), ["frame_index", "time", "ball_x", "ball_y", "ball_score", "ball_source"]].to_csv(
            ball_trajectory_path, index=False, encoding="utf-8-sig"
        )
    _create_analysis_videos(source, processed_path, stick_path, data, info, shooting_hand)
    _create_trajectory_video(processed_path, trajectory_path, data, shots, info)
    # 保留 V0.1 的关键帧文件名，同时为每次投篮输出独立关键帧与片段。
    primary_keyframes = {
        key: value for key, value in (primary.timeline_frames or {}).items()
        if key in {"ready", "lowest", "set_point", "release_candidate", "followthrough"}
    }
    keyframes = _extract_keyframes(trajectory_path, output_dir, data, primary_keyframes)
    for shot in shots:
        shot_keyframes = {
            key: value for key, value in (shot.timeline_frames or {}).items()
            if key in {"ready", "lowest", "set_point", "release_candidate", "followthrough"}
        }
        shot.keyframes = _extract_keyframes(
            trajectory_path,
            output_dir,
            data,
            shot_keyframes,
            filename_prefix=f"shot_{shot.shot_id:02d}",
        )
        shot.clip_path = _extract_shot_clip(
            trajectory_path,
            output_dir / f"shot_{shot.shot_id:02d}.mp4",
            shot.window,
            info,
        )

    ball_valid = int(data["ball_x"].notna().sum())
    hand_valid = int(data["hand_wrist_angle"].notna().sum())
    model_notes: list[str] = []
    if pose_frame_errors:
        model_notes.append(f"Pose 有 {pose_frame_errors} 帧推理失败，其余帧已继续分析；最后错误：{pose_last_error}")
    if hand_error:
        model_notes.append(f"手部模型加载失败，已降级：{hand_error}")
    elif hand_valid / max(len(data), 1) < 0.08:
        model_notes.append("手部有效帧不足，翻腕指标只作缺失提示，不给具体结论。")
    if ball_error:
        model_notes.append(f"篮球模型加载失败，已降级：{ball_error}")
    elif not ball_track_reliable:
        model_notes.append("篮球候选轨迹未通过‘靠近手腕—向上分离’可信度校验，已隐藏轨迹和假出手点。")
    elif ball_valid / max(len(data), 1) < 0.08:
        model_notes.append("篮球有效轨迹不足，出手时刻主要由手腕速度规则估计。")
    release_points = [
        (shot.metrics["ball_release_x"], shot.metrics["ball_release_y"])
        for shot in shots
        if math.isfinite(shot.metrics["ball_release_x"])
        and math.isfinite(shot.metrics["ball_release_y"])
    ]
    science_boundaries = {
        "A·可直接测量": ["二维肘/膝/髋角", "躯干二维倾角", "阶段时间", "手腕画面轨迹"],
        "B·可规则估计": ["出手候选时机", "动作阶段", "肘部横向偏移", "上下肢时序", "可信轨迹的二维初始飞行方向"],
        "C·当前不能可靠判断": ["真实三维篮球出射角", "篮球旋转", "手指真实施力", "精确球速", "肌肉发力", "三维肘朝向"],
    }
    if progress:
        progress(1.0, "分析完成，正在生成页面结果")
    return AnalysisResult(
        output_dir=output_dir,
        processed_video=processed_path,
        csv_path=csv_path,
        keyframes=keyframes,
        data=data,
        info=info,
        phases=phases,
        stats=stats,
        metrics=metrics,
        summary=summary,
        suggestions=suggestions,
        valid_frames=valid_frames,
        shooting_hand=shooting_hand,
        trajectory_video=trajectory_path,
        shots=shots,
        capture_quality=capture_quality,
        release_points=release_points,
        consistency=consistency,
        model_notes=model_notes,
        stick_figure_video=stick_path,
        shot_summary_path=shot_summary_path,
        ball_trajectory_path=ball_trajectory_path,
        science_boundaries=science_boundaries,
        original_video=original_path,
    )
