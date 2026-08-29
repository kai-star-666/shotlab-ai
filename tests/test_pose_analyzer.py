import unittest
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd

import pose_analyzer
from pose_analyzer import _build_metrics, _path_deviation
from biomechanics import AnalysisPhases


class WristPathDeviationTest(unittest.TestCase):
    def test_two_dimensional_path_works_with_numpy_2_5(self):
        data = pd.DataFrame(
            {"wrist_x": [0.0, 0.5, 1.0], "wrist_y": [0.0, 0.1, 0.0]}
        )
        self.assertAlmostEqual(_path_deviation(data, 0, 2), (0.01 / 3) ** 0.5)


class ShotValidRatioTest(unittest.TestCase):
    def test_uses_shot_window_length_not_whole_video_length(self):
        rows = 100
        data = pd.DataFrame({
            "elbow_angle": [150.0] * rows,
            "knee_angle": [130.0] * rows,
            "hip_angle": [125.0] * rows,
            "trunk_angle": [4.0] * rows,
            "wrist_x": np.linspace(0.4, 0.6, rows),
            "wrist_y": np.linspace(0.6, 0.3, rows),
            "hand_wrist_angle": [np.nan] * rows,
            "hand_finger_direction": [np.nan] * rows,
            "hand_confidence": [0.0] * rows,
            "ball_x": [np.nan] * rows,
            "ball_y": [np.nan] * rows,
            "elbow_alignment": [0.1] * rows,
            "forearm_direction": [80.0] * rows,
            "shoulder_height_difference": [0.02] * rows,
            "shooting_shoulder_lift": [0.01] * rows,
            "wrist_relative_shoulder": [0.2] * rows,
        })
        metrics = _build_metrics(
            data,
            AnalysisPhases(10, 12, 16, 19),
            valid_frames=8,
            frame_count=10,
        )
        self.assertAlmostEqual(metrics["valid_ratio"], 0.8)


class KeyframeOverlayLabelTest(unittest.TestCase):
    def test_opencv_overlay_labels_are_ascii(self):
        self.assertTrue(hasattr(pose_analyzer, "KEYFRAME_OVERLAY_LABELS"))
        self.assertTrue(all(label.isascii() for label in pose_analyzer.KEYFRAME_OVERLAY_LABELS.values()))


class UnicodeImageWriteTest(unittest.TestCase):
    def test_keyframe_writer_supports_unicode_windows_path(self):
        if not hasattr(pose_analyzer, "write_image"):
            self.fail("pose_analyzer.write_image is not implemented")
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "中文结果"
            output_dir.mkdir()
            output_path = output_dir / "关键帧.jpg"
            image = np.full((24, 32, 3), 127, dtype=np.uint8)
            pose_analyzer.write_image(output_path, image)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            decoded = cv2.imdecode(np.fromfile(output_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(decoded)


if __name__ == "__main__":
    unittest.main()
