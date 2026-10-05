"""Fast checks for subtitle files and URL/output handling (no model download)."""
import json
import tempfile
import unittest
from pathlib import Path

from bilibili_audio_subtitles import main, normalize_chunks, save_subtitles, timecode, video_identity


class SubtitleTests(unittest.TestCase):
    def test_url_keeps_selected_part_and_discards_tracking(self):
        self.assertEqual(
            video_identity("https://www.bilibili.com/video/BV1w88n6uE1X/?p=2&spm_id_from=tracking"),
            ("https://www.bilibili.com/video/BV1w88n6uE1X/?p=2", "BV1w88n6uE1X-p2"),
        )
        self.assertEqual(video_identity("av123")[1], "av123-p1")
        for invalid in ("https://bilibili.com.evil.test/video/BV123", "BV123?p=2", "https://www.bilibili.com/video/BV123/?p=0"):
            with self.assertRaises(ValueError):
                video_identity(invalid)

    def test_timestamp_rounding_carries_into_next_hour(self):
        self.assertEqual(timecode(3599.9996, ","), "01:00:00,000")
        self.assertEqual(timecode(3723.456, "."), "01:02:03.456")

    def test_missing_end_and_overlap_are_repaired_without_changing_raw(self):
        chunks = [
            {"timestamp": (0, None), "text": " 开始 "},
            {"timestamp": (2, 4), "text": "继续"},
            {"timestamp": (3.8, None), "text": "结束"},
        ]
        cues, notes = normalize_chunks(chunks, 6)
        self.assertEqual([(c["start"], c["end"]) for c in cues], [(0, 2), (2, 4), (4, 6)])
        self.assertTrue(notes)
        self.assertIsNone(chunks[0]["timestamp"][1])

    def test_invalid_or_empty_timestamps_do_not_become_fake_subtitles(self):
        for chunks in ([], [{"text": "只有文字"}], [{"text": "零长度", "timestamp": (1, 1)}],
                       [{"text": "非法", "timestamp": (float("nan"), 2)}]):
            with self.assertRaises(ValueError):
                normalize_chunks(chunks, 6)

    def test_written_subtitles_have_valid_format_and_unicode(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = {suffix: Path(temporary) / f"字幕.{suffix}" for suffix in ("srt", "vtt", "txt", "json")}
            result = {"text": "你好世界", "chunks": [
                {"text": "你好", "timestamp": (0, 1.5)},
                {"text": "世界", "timestamp": (1.5, 2.3)},
            ]}
            save_subtitles(paths, result, {"return_timestamps": True}, 2.3)
            self.assertEqual(paths["srt"].read_text(),
                             "1\n00:00:00,000 --> 00:00:01,500\n你好\n\n2\n00:00:01,500 --> 00:00:02,300\n世界\n\n")
            self.assertTrue(paths["vtt"].read_text().startswith("WEBVTT\n\n00:00:00.000"))
            self.assertEqual(paths["txt"].read_text(), "你好世界\n")
            self.assertEqual(len(json.loads(paths["json"].read_text())["segments"]), 2)

    def test_invalid_model_output_preserves_raw_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = {suffix: Path(temporary) / f"result.{suffix}" for suffix in ("srt", "vtt", "txt", "json")}
            with self.assertRaises(ValueError):
                save_subtitles(paths, {"text": "无法对齐", "chunks": []}, {}, 3)
            self.assertEqual(json.loads(paths["json"].read_text())["text"], "无法对齐")
            self.assertFalse(paths["srt"].exists())

    def test_existing_output_is_preserved_before_downloading(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "BV123-p1.srt"
            existing.write_text("已有字幕")
            with self.assertRaises(FileExistsError):
                main(["BV123", "--project-dir", temporary])
            self.assertEqual(existing.read_text(), "已有字幕")


if __name__ == "__main__":
    unittest.main()
