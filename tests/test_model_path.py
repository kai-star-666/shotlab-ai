import unittest
from pathlib import Path
import tempfile

import pose_analyzer


class NativeModelPathTest(unittest.TestCase):
    def test_analyzer_exposes_ascii_safe_model_path_helper(self):
        self.assertTrue(
            hasattr(pose_analyzer, "ascii_safe_model_path"),
            "pose_analyzer should protect MediaPipe native loading from Unicode paths",
        )

    def test_helper_copies_unicode_model_to_ascii_cache(self):
        if not hasattr(pose_analyzer, "ascii_safe_model_path"):
            self.fail("ascii_safe_model_path is not implemented")
        with tempfile.TemporaryDirectory() as temporary:
            unicode_dir = Path(temporary) / "模型目录"
            unicode_dir.mkdir()
            source = unicode_dir / "姿态.task"
            source.write_bytes(b"test-model")
            safe_path = pose_analyzer.ascii_safe_model_path(source)
            self.assertTrue(safe_path.exists())
            self.assertTrue(str(safe_path).isascii())
            self.assertEqual(safe_path.read_bytes(), b"test-model")


if __name__ == "__main__":
    unittest.main()
