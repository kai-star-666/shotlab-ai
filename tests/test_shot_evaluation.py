import math
import unittest

import pandas as pd


class ShotEvaluationContractTest(unittest.TestCase):
    def test_module_exports_v03_evaluation_api(self):
        from shot_evaluation import (
            build_angle_comparison,
            build_diagnostics,
            build_stability_scores,
            build_timing_analysis,
            estimate_2d_ball_launch_angle,
        )

        self.assertTrue(callable(build_angle_comparison))
        self.assertTrue(callable(build_diagnostics))
        self.assertTrue(callable(build_stability_scores))
        self.assertTrue(callable(build_timing_analysis))
        self.assertTrue(callable(estimate_2d_ball_launch_angle))


class AngleComparisonTest(unittest.TestCase):
    def test_uses_reference_ranges_and_marks_unreliable_values(self):
        from shot_evaluation import build_angle_comparison

        table = build_angle_comparison(
            {
                "knee_lowest": 118.0,
                "hip_lowest": 126.0,
                "elbow_release": 143.0,
                "trunk_max_abs": 18.0,
                "elbow_alignment": math.nan,
                "wrist_snap_change": math.nan,
                "ball_launch_angle_2d": math.nan,
            }
        )
        self.assertEqual(list(table.columns), ["动作指标", "我的实际数据", "参考范围", "当前判断", "性质"])
        knee = table.loc[table["动作指标"] == "最低点膝关节角"].iloc[0]
        elbow = table.loc[table["动作指标"] == "出手候选帧肘角"].iloc[0]
        ball = table.loc[table["动作指标"] == "篮球二维初始飞行方向"].iloc[0]
        self.assertIn("暂无通用唯一区间", knee["参考范围"])
        self.assertEqual(knee["当前判断"], "记录个人基线")
        self.assertIn("系统观察阈值", elbow["当前判断"])
        self.assertEqual(ball["我的实际数据"], "数据不足")


class TimingAnalysisTest(unittest.TestCase):
    def test_returns_ordered_stage_times_and_chain_lag(self):
        from shot_evaluation import build_timing_analysis

        data = pd.DataFrame(
            {
                "elbow_angle": [90, 91, 92, 95, 103, 120, 145, 165, 170, 172],
                "knee_angle": [160, 150, 135, 115, 120, 138, 155, 168, 172, 174],
                "wrist_y": [0.60, 0.61, 0.62, 0.60, 0.55, 0.47, 0.38, 0.30, 0.29, 0.30],
            }
        )
        result = build_timing_analysis(
            data,
            {"ready": 0, "lowest": 3, "release_candidate": 7, "followthrough": 9},
            fps=10,
        )
        self.assertAlmostEqual(result["准备到最低点"], 0.3)
        self.assertAlmostEqual(result["最低点到出手候选"], 0.4)
        self.assertTrue(math.isfinite(result["上下肢快速伸展起点时间差"]))


class DiagnosticTest(unittest.TestCase):
    def test_problems_include_data_evidence_and_drills_match_problem(self):
        from shot_evaluation import build_diagnostics

        result = build_diagnostics(
            {
                "knee_motion_range": 12.0,
                "elbow_release": 139.0,
                "trunk_ready": 3.0,
                "trunk_release": 17.0,
                "trunk_std": 8.0,
                "elbow_alignment": 0.22,
                "wrist_path_deviation": 0.05,
            },
            hand_reliable=False,
        )
        self.assertGreaterEqual(len(result.problems), 3)
        self.assertLessEqual(len(result.problems), 5)
        self.assertTrue(all(item.evidence for item in result.problems))
        self.assertGreaterEqual(len(result.drills), 2)
        self.assertLessEqual(len(result.drills), 4)
        self.assertTrue(all(item.dosage for item in result.drills))
        self.assertIn("数据不足", result.hand_analysis)


class StabilityAndBallTest(unittest.TestCase):
    def test_scores_are_bounded_and_missing_hand_is_none(self):
        from shot_evaluation import build_stability_scores

        scores = build_stability_scores(
            {
                "knee_cycle_smoothness": 0.8,
                "elbow_cycle_smoothness": 0.7,
                "trunk_std": 4.0,
                "wrist_path_deviation": math.nan,
            }
        )
        self.assertTrue(all(value is None or 0 <= value <= 100 for value in scores.values()))
        self.assertIsNone(scores["手腕轨迹稳定性"])

    def test_launch_angle_requires_reliable_early_flight_track(self):
        from shot_evaluation import estimate_2d_ball_launch_angle

        angle = estimate_2d_ball_launch_angle(
            [math.nan, 0.50, 0.55, 0.61, 0.68, 0.75],
            [math.nan, 0.42, 0.36, 0.29, 0.23, 0.18],
            release_frame=1,
        )
        self.assertGreater(angle, 25)
        self.assertLess(angle, 75)
        self.assertTrue(
            math.isnan(
                estimate_2d_ball_launch_angle(
                    [0.5, math.nan, math.nan], [0.4, math.nan, math.nan], 0
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
