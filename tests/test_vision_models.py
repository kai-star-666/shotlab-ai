import math
import unittest
from pathlib import Path

import cv2
import numpy as np


class VisionModelModuleContractTest(unittest.TestCase):
    def test_vision_models_module_exists(self):
        self.assertTrue(
            (Path(__file__).resolve().parents[1] / "vision_models.py").exists(),
            "vision_models.py should exist",
        )


try:
    from vision_models import BallCandidate, choose_ball_candidate, find_colored_ball_candidates
except ImportError:
    BallCandidate = None


if BallCandidate is not None:
    class _FailingDetector:
        def detect(self, image):
            raise RuntimeError("synthetic detector failure")


    class RuntimeFailureRecordingTest(unittest.TestCase):
        def test_hand_failure_is_recorded_without_crashing_pose_pipeline(self):
            from vision_models import VisionModels

            models = VisionModels.__new__(VisionModels)
            models.hand_landmarker = _FailingDetector()
            models.object_detector = None
            models.hand_error = None
            models.ball_error = None
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            result = models.detect_hand(frame, (0.5, 0.5), (0.5, 0.4), (0.2, 0.1, 0.8, 0.9))
            self.assertTrue(math.isnan(result.wrist_angle))
            self.assertIn("synthetic detector failure", models.hand_error)

        def test_ball_failure_is_recorded_without_crashing_pose_pipeline(self):
            from vision_models import VisionModels

            models = VisionModels.__new__(VisionModels)
            models.hand_landmarker = None
            models.object_detector = _FailingDetector()
            models.hand_error = None
            models.ball_error = None
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            models.detect_ball(frame, (0.5, 0.5), None, frame_index=0)
            self.assertIn("synthetic detector failure", models.ball_error)

    class ColoredBallCandidateTest(unittest.TestCase):
        def test_detects_orange_circular_candidate(self):
            frame = np.zeros((200, 300, 3), dtype=np.uint8)
            cv2.circle(frame, (180, 70), 14, (0, 110, 245), -1)
            candidates = find_colored_ball_candidates(frame)
            self.assertTrue(candidates)
            best = min(candidates, key=lambda item: abs(item.x - 0.6) + abs(item.y - 0.35))
            self.assertAlmostEqual(best.x, 0.6, delta=0.03)
            self.assertAlmostEqual(best.y, 0.35, delta=0.03)
            self.assertEqual(best.source, "color_motion")

        def test_rejects_large_red_background_region(self):
            frame = np.zeros((200, 300, 3), dtype=np.uint8)
            cv2.rectangle(frame, (0, 0), (299, 80), (0, 0, 230), -1)
            self.assertEqual(find_colored_ball_candidates(frame), [])

        def test_rejects_tiny_red_background_speck(self):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.circle(frame, (500, 300), 3, (0, 0, 240), -1)
            self.assertEqual(find_colored_ball_candidates(frame), [])


    class CandidateSelectionTest(unittest.TestCase):
        def test_prefers_candidate_continuing_previous_ball_track(self):
            candidates = [
                BallCandidate(0.22, 0.25, 0.03, 0.03, 0.8, "color_motion"),
                BallCandidate(0.62, 0.42, 0.03, 0.03, 0.7, "color_motion"),
            ]
            selected = choose_ball_candidate(
                candidates,
                wrist=(0.20, 0.60),
                previous=(0.60, 0.45),
                max_jump=0.2,
            )
            self.assertIsNotNone(selected)
            self.assertAlmostEqual(selected.x, 0.62)

        def test_uses_wrist_proximity_to_start_track(self):
            candidates = [
                BallCandidate(0.2, 0.2, 0.03, 0.03, 0.9, "color_motion"),
                BallCandidate(0.7, 0.7, 0.04, 0.04, 0.7, "color_motion"),
            ]
            selected = choose_ball_candidate(candidates, wrist=(0.68, 0.69), previous=None)
            self.assertAlmostEqual(selected.x, 0.7)

        def test_does_not_start_track_from_candidate_too_far_from_wrist(self):
            candidates = [
                BallCandidate(0.65, 0.50, 0.04, 0.04, 0.9, "color_motion")
            ]
            self.assertIsNone(
                choose_ball_candidate(candidates, wrist=(0.50, 0.50), previous=None)
            )


if __name__ == "__main__":
    unittest.main()
