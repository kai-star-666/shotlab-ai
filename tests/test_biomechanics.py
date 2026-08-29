import math
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "biomechanics.py"


class ProjectStructureTest(unittest.TestCase):
    def test_biomechanics_module_exists(self):
        self.assertTrue(MODULE_PATH.exists(), "biomechanics.py should exist")


if MODULE_PATH.exists():
    from biomechanics import (
        AnalysisPhases,
        LandmarkPoint,
        build_training_feedback,
        calculate_angle,
        estimate_phases,
        interpolate_short_gaps,
    )


    class AngleCalculationTest(unittest.TestCase):
        def test_calculates_right_angle(self):
            self.assertAlmostEqual(
                calculate_angle((1.0, 0.0), (0.0, 0.0), (0.0, 1.0)), 90.0
            )

        def test_returns_nan_for_zero_length_vector(self):
            self.assertTrue(math.isnan(calculate_angle((0, 0), (0, 0), (1, 1))))

        def test_accepts_landmark_points(self):
            a = LandmarkPoint(1, 0, 0, 0.9)
            b = LandmarkPoint(0, 0, 0, 0.9)
            c = LandmarkPoint(-1, 0, 0, 0.9)
            self.assertAlmostEqual(calculate_angle(a, b, c), 180.0)


    class GapInterpolationTest(unittest.TestCase):
        def test_only_fills_short_internal_gaps(self):
            values = [100.0, math.nan, 120.0, math.nan, math.nan, math.nan, 150.0]
            filled = interpolate_short_gaps(values, max_gap=2)
            self.assertEqual(filled[1], 110.0)
            self.assertTrue(math.isnan(filled[3]))
            self.assertTrue(math.isnan(filled[4]))
            self.assertTrue(math.isnan(filled[5]))


    class PhaseEstimationTest(unittest.TestCase):
        def test_orders_key_phases_and_uses_wrist_speed_for_release(self):
            knee = [160, 150, 130, 105, 120, 145, 165, 170]
            wrist_y = [0.70, 0.68, 0.64, 0.60, 0.50, 0.30, 0.18, 0.16]
            phases = estimate_phases(knee, wrist_y, fps=10)
            self.assertIsInstance(phases, AnalysisPhases)
            self.assertEqual(phases.lowest, 3)
            self.assertLess(phases.ready, phases.lowest)
            self.assertGreater(phases.release, phases.lowest)
            self.assertGreater(phases.followthrough, phases.release)


    class FeedbackTest(unittest.TestCase):
        def test_feedback_uses_cautious_language_and_flags_unstable_trunk(self):
            metrics = {
                "valid_ratio": 0.9,
                "elbow_release": 150.0,
                "elbow_extension_change": 28.0,
                "elbow_extension_std": 4.0,
                "knee_lowest": 115.0,
                "knee_motion_range": 45.0,
                "hip_motion_std": 5.0,
                "trunk_max_abs": 18.0,
                "trunk_std": 8.0,
                "wrist_path_deviation": 0.02,
            }
            summary, suggestions = build_training_feedback(metrics)
            combined = " ".join(summary + suggestions)
            self.assertIn("躯干", combined)
            self.assertNotIn("诊断", combined)
            self.assertTrue(any(word in combined for word in ("可能", "数据显示", "建议", "可尝试")))


if __name__ == "__main__":
    unittest.main()
