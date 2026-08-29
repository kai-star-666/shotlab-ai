import math
import unittest

import numpy as np


class V02ModuleContractTest(unittest.TestCase):
    def test_video_intelligence_module_exists(self):
        from pathlib import Path

        self.assertTrue(
            (Path(__file__).resolve().parents[1] / "video_intelligence.py").exists(),
            "video_intelligence.py should exist",
        )


try:
    from video_intelligence import (
        ReleaseEvent,
        assess_capture_quality,
        build_shot_windows,
        calculate_wrist_flexion,
        detect_release_events,
        is_ball_track_reliable,
        select_nearest_hand,
        validated_wrist_snap,
    )
except ImportError:
    ReleaseEvent = None


if ReleaseEvent is not None:
    class CaptureQualityTest(unittest.TestCase):
        def test_flags_dark_blurry_distant_and_non_side_view_video(self):
            result = assess_capture_quality(
                brightness=28,
                blur_score=18,
                pose_valid_ratio=0.55,
                full_body_ratio=0.45,
                body_height_ratio=0.24,
                side_view_score=0.30,
                resolution=(1280, 720),
            )
            combined = " ".join(result.issues + result.corrections)
            self.assertLess(result.score, 60)
            self.assertIn("光线", combined)
            self.assertIn("模糊", combined)
            self.assertIn("靠近", combined)
            self.assertIn("侧面", combined)


    class ReleaseDetectionTest(unittest.TestCase):
        def test_ball_hand_separation_is_preferred_over_wrist_fallback(self):
            nan = math.nan
            ball_x = [nan, nan, 0.50, 0.51, 0.52, 0.58, 0.67, 0.76]
            ball_y = [nan, nan, 0.45, 0.43, 0.40, 0.32, 0.22, 0.15]
            wrist_x = [0.50] * 8
            wrist_y = [0.48, 0.47, 0.45, 0.43, 0.42, 0.41, 0.40, 0.40]
            knee = [155, 145, 125, 110, 120, 140, 155, 165]
            events = detect_release_events(
                ball_x, ball_y, wrist_x, wrist_y, knee, fps=10
            )
            self.assertEqual(len(events), 1)
            self.assertIn(events[0].frame_index, (4, 5))
            self.assertEqual(events[0].source, "ball_hand_separation")

        def test_close_candidates_are_deduplicated(self):
            wrist_y = [0.7, 0.68, 0.62, 0.45, 0.30, 0.28, 0.31, 0.29, 0.25, 0.24]
            knee = [160, 150, 130, 110, 120, 140, 155, 160, 165, 168]
            events = detect_release_events(
                [math.nan] * 10,
                [math.nan] * 10,
                [0.5] * 10,
                wrist_y,
                knee,
                fps=10,
            )
            self.assertLessEqual(len(events), 1)

        def test_wrist_motion_without_knee_loading_is_not_a_new_shot(self):
            wrist_y = [0.52] * 18 + [0.50, 0.46, 0.40, 0.33, 0.29, 0.31, 0.36, 0.43] + [0.50] * 24
            nearly_flat_knee = [166.0] * 50
            nearly_flat_knee[17:22] = [165.5, 164.5, 163.5, 164.5, 165.5]
            events = detect_release_events(
                [math.nan] * 50,
                [math.nan] * 50,
                [0.5] * 50,
                wrist_y,
                nearly_flat_knee,
                fps=30,
            )
            self.assertEqual(events, [])

        def test_ball_separation_without_knee_loading_is_rejected(self):
            ball_x = [0.50, 0.51, 0.52, 0.58, 0.68, 0.77, 0.83, 0.87]
            ball_y = [0.48, 0.46, 0.43, 0.35, 0.25, 0.18, 0.14, 0.12]
            events = detect_release_events(
                ball_x,
                ball_y,
                [0.50] * 8,
                [0.48, 0.47, 0.45, 0.43, 0.42, 0.42, 0.43, 0.44],
                [166.0] * 8,
                fps=10,
            )
            self.assertEqual(events, [])


    class BallTrackReliabilityTest(unittest.TestCase):
        def test_rejects_track_that_never_passes_shooting_wrist(self):
            self.assertFalse(
                is_ball_track_reliable(
                    [0.20, 0.19, 0.18, 0.17, 0.16, 0.15],
                    [0.30, 0.29, 0.28, 0.27, 0.26, 0.25],
                    [0.50] * 6,
                    [0.48, 0.46, 0.43, 0.40, 0.39, 0.40],
                    [ReleaseEvent(3, 0.3, "wrist_velocity", 0.6)],
                    fps=10,
                )
            )

        def test_accepts_track_that_leaves_wrist_and_rises_after_release(self):
            self.assertTrue(
                is_ball_track_reliable(
                    [0.50, 0.51, 0.52, 0.58, 0.67, 0.76, 0.82, 0.86],
                    [0.48, 0.46, 0.43, 0.35, 0.25, 0.18, 0.14, 0.12],
                    [0.50] * 8,
                    [0.48, 0.47, 0.45, 0.43, 0.42, 0.42, 0.43, 0.44],
                    [ReleaseEvent(3, 0.3, "ball_hand_separation", 0.9)],
                    fps=10,
                )
            )


    class ShotWindowTest(unittest.TestCase):
        def test_builds_non_overlapping_windows_around_multiple_releases(self):
            events = [
                ReleaseEvent(30, 1.0, "ball_hand_separation", 0.9),
                ReleaseEvent(90, 3.0, "wrist_velocity", 0.6),
            ]
            windows = build_shot_windows(events, total_frames=130, fps=30)
            self.assertEqual(len(windows), 2)
            self.assertLess(windows[0].start_frame, 30)
            self.assertGreater(windows[0].end_frame, 30)
            self.assertLess(windows[0].end_frame, windows[1].start_frame)
            self.assertEqual([window.shot_id for window in windows], [1, 2])


    class HandMetricTest(unittest.TestCase):
        def test_calculates_wrist_flexion_angle(self):
            angle = calculate_wrist_flexion((0, 1), (0, 0), (1, 0))
            self.assertAlmostEqual(angle, 90.0)

        def test_selects_hand_nearest_pose_wrist(self):
            index = select_nearest_hand([(0.2, 0.3), (0.81, 0.71)], (0.8, 0.7))
            self.assertEqual(index, 1)

        def test_returns_none_when_all_hands_are_too_far(self):
            self.assertIsNone(select_nearest_hand([(0.1, 0.1)], (0.9, 0.9), max_distance=0.2))

        def test_rejects_implausible_or_low_confidence_wrist_snap(self):
            self.assertTrue(math.isnan(validated_wrist_snap(88, 179, 0.8, 0.9)))
            self.assertTrue(math.isnan(validated_wrist_snap(150, 125, 0.2, 0.9)))

        def test_returns_absolute_wrist_snap_change_when_reliable(self):
            self.assertAlmostEqual(validated_wrist_snap(158, 132, 0.8, 0.9), 26.0)


if __name__ == "__main__":
    unittest.main()
