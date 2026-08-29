import unittest

from web_ui import MAX_UPLOAD_BYTES, validate_upload


class UploadValidationTest(unittest.TestCase):
    def test_accepts_supported_video_within_limit(self):
        self.assertIsNone(validate_upload("shot.MP4", MAX_UPLOAD_BYTES))

    def test_rejects_unsupported_extension(self):
        self.assertIn("MP4", validate_upload("shot.webm", 1024))

    def test_rejects_empty_upload(self):
        self.assertIn("空文件", validate_upload("shot.mp4", 0))

    def test_rejects_upload_over_limit(self):
        self.assertIn("200 MB", validate_upload("shot.mp4", MAX_UPLOAD_BYTES + 1))


if __name__ == "__main__":
    unittest.main()
