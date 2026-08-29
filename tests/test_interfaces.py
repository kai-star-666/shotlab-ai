import inspect
import unittest

from pose_analyzer import analyze_video


class AnalyzerInterfaceTest(unittest.TestCase):
    def test_v02_model_paths_are_part_of_public_analyzer_interface(self):
        parameters = inspect.signature(analyze_video).parameters
        self.assertIn("hand_model_path", parameters)
        self.assertIn("ball_model_path", parameters)


if __name__ == "__main__":
    unittest.main()
