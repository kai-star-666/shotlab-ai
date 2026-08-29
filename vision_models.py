"""V0.2 可选手部与篮球视觉模型，以及篮球颜色候选跟踪。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np

from video_intelligence import calculate_wrist_flexion, select_nearest_hand


@dataclass(frozen=True)
class BallCandidate:
    x: float
    y: float
    width: float
    height: float
    score: float
    source: str


@dataclass
class HandMeasurement:
    wrist_angle: float = math.nan
    finger_direction: float = math.nan
    confidence: float = 0.0
    points: list[tuple[float, float]] | None = None


def find_colored_ball_candidates(frame: np.ndarray) -> list[BallCandidate]:
    """查找红/橙色、近圆形的小区域；时序跟踪会进一步排除背景。"""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    orange_red = cv2.inRange(hsv, np.array((0, 65, 45)), np.array((28, 255, 255)))
    deep_red = cv2.inRange(hsv, np.array((165, 65, 45)), np.array((179, 255, 255)))
    mask = cv2.bitwise_or(orange_red, deep_red)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = height * width
    minimum_area = max(18.0, frame_area * 0.00005)
    maximum_area = frame_area * 0.012
    candidates: list[BallCandidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < minimum_area or area > maximum_area:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width <= 0 or box_height <= 0:
            continue
        aspect = box_width / box_height
        if not 0.45 <= aspect <= 2.2:
            continue
        if max(box_width / width, box_height / height) < 0.015:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
        fill_ratio = area / (box_width * box_height)
        if circularity < 0.28 or fill_ratio < 0.32:
            continue
        score = float(np.clip(0.45 * circularity + 0.35 * fill_ratio + 0.20, 0, 0.95))
        candidates.append(
            BallCandidate(
                (x + box_width / 2) / width,
                (y + box_height / 2) / height,
                box_width / width,
                box_height / height,
                score,
                "color_motion",
            )
        )
    return candidates


def choose_ball_candidate(
    candidates: Sequence[BallCandidate],
    wrist: tuple[float, float] | None,
    previous: tuple[float, float] | None,
    max_jump: float = 0.16,
) -> BallCandidate | None:
    if not candidates:
        return None
    if previous is not None:
        continuing = [
            candidate
            for candidate in candidates
            if math.dist((candidate.x, candidate.y), previous) <= max_jump
        ]
        if continuing:
            return max(
                continuing,
                key=lambda item: item.score - 1.8 * math.dist((item.x, item.y), previous),
            )
    if wrist is None:
        return None
    near_wrist = [
        candidate
        for candidate in candidates
        if math.dist((candidate.x, candidate.y), wrist) <= 0.11
    ]
    if not near_wrist:
        return None
    return max(
        near_wrist,
        key=lambda item: item.score - 1.2 * math.dist((item.x, item.y), wrist),
    )


def person_bbox_from_landmarks(landmarks: Sequence, padding: float = 0.06) -> tuple[float, float, float, float] | None:
    valid = [
        item
        for item in landmarks
        if float(getattr(item, "visibility", 1.0)) >= 0.35
        and math.isfinite(float(item.x))
        and math.isfinite(float(item.y))
    ]
    if len(valid) < 8:
        return None
    x1 = max(0.0, min(item.x for item in valid) - padding)
    y1 = max(0.0, min(item.y for item in valid) - padding)
    x2 = min(1.0, max(item.x for item in valid) + padding)
    y2 = min(1.0, max(item.y for item in valid) + padding)
    return x1, y1, x2, y2


class VisionModels:
    """模型失败时返回空结果，调用方继续完成姿态分析。"""

    def __init__(self, hand_model: str | Path | None, ball_model: str | Path | None):
        self.hand_landmarker = None
        self.object_detector = None
        self.hand_error: str | None = None
        self.ball_error: str | None = None
        if hand_model and Path(hand_model).exists():
            try:
                options = mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_buffer=Path(hand_model).read_bytes()),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.22,
                    min_hand_presence_confidence=0.22,
                    min_tracking_confidence=0.25,
                )
                self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
            except Exception as error:
                self.hand_error = str(error)
        if ball_model and Path(ball_model).exists():
            try:
                options = mp.tasks.vision.ObjectDetectorOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_buffer=Path(ball_model).read_bytes()),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    max_results=10,
                    score_threshold=0.15,
                    category_allowlist=["sports ball"],
                )
                self.object_detector = mp.tasks.vision.ObjectDetector.create_from_options(options)
            except Exception as error:
                self.ball_error = str(error)

    def close(self) -> None:
        if self.hand_landmarker is not None:
            self.hand_landmarker.close()
        if self.object_detector is not None:
            self.object_detector.close()

    def __enter__(self) -> "VisionModels":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def detect_hand(
        self,
        frame: np.ndarray,
        pose_wrist: tuple[float, float] | None,
        pose_elbow: tuple[float, float] | None,
        person_bbox: tuple[float, float, float, float] | None,
    ) -> HandMeasurement:
        if self.hand_landmarker is None or pose_wrist is None or pose_elbow is None:
            return HandMeasurement()
        height, width = frame.shape[:2]
        body_height = (person_bbox[3] - person_bbox[1]) * height if person_bbox else height * 0.45
        radius = int(np.clip(body_height * 0.22, 42, min(width, height) * 0.28))
        center_x, center_y = int(pose_wrist[0] * width), int(pose_wrist[1] * height)
        x1, x2 = max(0, center_x - radius), min(width, center_x + radius)
        y1, y2 = max(0, center_y - radius), min(height, center_y + radius)
        if x2 - x1 < 24 or y2 - y1 < 24:
            return HandMeasurement()
        crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_CUBIC)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        try:
            result = self.hand_landmarker.detect(image)
        except Exception as error:
            if self.hand_error is None:
                self.hand_error = f"手部单帧推理失败：{error}"
            return HandMeasurement()
        mapped_hands: list[list[tuple[float, float]]] = []
        centers: list[tuple[float, float]] = []
        for hand in result.hand_landmarks:
            points = [
                ((x1 + item.x * (x2 - x1)) / width, (y1 + item.y * (y2 - y1)) / height)
                for item in hand
            ]
            mapped_hands.append(points)
            centers.append(points[0])
        selected = select_nearest_hand(centers, pose_wrist, max_distance=0.18)
        if selected is None:
            return HandMeasurement()
        points = mapped_hands[selected]
        wrist_angle = calculate_wrist_flexion(pose_elbow, pose_wrist, points[9])
        direction = float(np.degrees(np.arctan2(points[12][0] - points[0][0], points[0][1] - points[12][1])))
        confidence = 0.5
        if selected < len(result.handedness) and result.handedness[selected]:
            confidence = float(result.handedness[selected][0].score)
        return HandMeasurement(wrist_angle, direction, confidence, points)

    def detect_ball(
        self,
        frame: np.ndarray,
        wrist: tuple[float, float] | None,
        previous: tuple[float, float] | None,
        frame_index: int,
    ) -> BallCandidate | None:
        height, width = frame.shape[:2]
        candidates = find_colored_ball_candidates(frame)
        if self.object_detector is not None and frame_index % 3 == 0:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                result = self.object_detector.detect(image)
                for detection in result.detections:
                    if not detection.categories:
                        continue
                    category = detection.categories[0]
                    box = detection.bounding_box
                    candidates.append(
                        BallCandidate(
                            (box.origin_x + box.width / 2) / width,
                            (box.origin_y + box.height / 2) / height,
                            box.width / width,
                            box.height / height,
                            float(category.score),
                            "efficientdet_sports_ball",
                        )
                    )
            except Exception as error:
                if self.ball_error is None:
                    self.ball_error = f"篮球单帧推理失败：{error}"
        return choose_ball_candidate(candidates, wrist, previous)
