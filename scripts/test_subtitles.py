"""Fast checks for structure, quality alerts, raw preservation, and SRT-only output."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bilibili_audio_subtitles import (
    SCRIPT_VERSION, check_quality, choose_output, main, normalize_chunks,
    save_subtitles, timecode, video_identity,
)


class SubtitleTests(unittest.TestCase):
    def test_url_keeps_selected_part_and_discards_tracking(self):
        self.assertEqual(
            video_identity('https://www.bilibili.com/video/BV1w88n6uE1X/?p=2&spm_id_from=tracking'),
            ('https://www.bilibili.com/video/BV1w88n6uE1X/?p=2', 'BV1w88n6uE1X-p2'),
        )
        self.assertEqual(video_identity('av123')[1], 'av123-p1')
        for invalid in ('https://bilibili.com.evil.test/video/BV123', 'BV123?p=2',
                        'https://www.bilibili.com/video/BV123/?p=0'):
            with self.assertRaises(ValueError):
                video_identity(invalid)

    def test_timestamp_rounding_carries_into_next_hour(self):
        self.assertEqual(timecode(3599.9996, ','), '01:00:00,000')
        self.assertEqual(timecode(3723.456, '.'), '01:02:03.456')

    def test_missing_end_and_overlap_are_repaired_without_changing_raw(self):
        chunks = [
            {'timestamp': (0, None), 'text': ' 开始 '},
            {'timestamp': (2, 4), 'text': '继续'},
            {'timestamp': (3.8, None), 'text': '结束'},
        ]
        cues, notes = normalize_chunks(chunks, 6)
        self.assertEqual([(c['start'], c['end']) for c in cues], [(0, 2), (2, 4), (4, 6)])
        self.assertTrue(notes)
        self.assertIsNone(chunks[0]['timestamp'][1])

    def test_invalid_or_empty_timestamps_do_not_become_fake_subtitles(self):
        for chunks in ([], [{'text': '只有文字'}], [{'text': '零长度', 'timestamp': (1, 1)}],
                       [{'text': '非法', 'timestamp': (float('nan'), 2)}]):
            with self.assertRaises(ValueError):
                normalize_chunks(chunks, 6)

    def test_only_srt_is_delivered_and_reproduction_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            srt, diagnostics = root / '字幕.srt', root / 'cache' / 'run.json'
            result = {'text': '你好世界', 'chunks': [
                {'text': '你好', 'timestamp': (0, 1.5)},
                {'text': '世界', 'timestamp': (1.5, 2.3)},
            ]}
            metadata = {'transcription_mode': 'native-long-form',
                        'generation_parameters': {'return_timestamps': True, 'task': 'transcribe'}}
            payload = save_subtitles(srt, diagnostics, result, metadata, 2.3)
            self.assertEqual(srt.read_text(),
                             '1\n00:00:00,000 --> 00:00:01,500\n你好\n\n2\n00:00:01,500 --> 00:00:02,300\n世界\n\n')
            self.assertEqual([p.name for p in root.iterdir() if p.is_file()], ['字幕.srt'])
            cached = json.loads(diagnostics.read_text())
            self.assertEqual(cached['script_version'], SCRIPT_VERSION)
            self.assertEqual(cached['chunks'][0]['timestamp'], [0, 1.5])
            self.assertEqual(cached['generation_parameters'], metadata['generation_parameters'])
            self.assertEqual(payload['quality_check']['status'], 'no_obvious_anomaly')
            self.assertFalse(payload['quality_check']['human_verified'])

    def test_invalid_model_output_preserves_raw_json_and_failure_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                save_subtitles(root / 'result.srt', root / 'cache' / 'run.json',
                               {'text': '无法对齐', 'chunks': []}, {}, 3)
            cached = json.loads((root / 'cache' / 'run.json').read_text())
            self.assertEqual(cached['text'], '无法对齐')
            self.assertEqual(cached['structure_check']['status'], 'failed')
            self.assertFalse((root / 'result.srt').exists())

    def test_retry_uses_independent_filename_and_keeps_previous_srt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / 'BV123-p1.srt'
            original.write_text('原版')
            retry = choose_output(root, 'BV123-p1', 'native', 'unique-run')
            self.assertNotEqual(retry, original)
            save_subtitles(retry, root / 'cache' / 'retry.json',
                           {'text': '新版', 'chunks': [{'text': '新版', 'timestamp': (0, 1)}]}, {}, 1)
            self.assertEqual(original.read_text(), '原版')
            self.assertEqual(len(list(root.glob('*.srt'))), 2)

    def test_structurally_valid_five_minute_cue_requires_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = save_subtitles(root / 'long.srt', root / 'cache' / 'run.json',
                                    {'text': '长段落', 'chunks': [
                                        {'text': '长段落', 'timestamp': (43.24, 346.06)}]}, {}, 758.5)
            self.assertEqual(payload['structure_check']['status'], 'passed')
            self.assertEqual(payload['quality_check']['status'], 'needs_review')
            self.assertIn('long_cue', {w['code'] for w in payload['quality_check']['warnings']})
            self.assertEqual(payload['recommendation'], 'review_required')

    def test_repeated_single_word_cues_across_feedback_window_are_flagged(self):
        cues = [{'start': 490.5 + i, 'end': 491.5 + i, 'text': '好'} for i in range(23)]
        quality = check_quality(cues)
        warning = next(w for w in quality['warnings'] if w['code'] == 'identical_cues')
        self.assertEqual(warning['cue_numbers'], [1, 23])
        self.assertEqual(warning['start'], 490.5)
        self.assertEqual(quality['status'], 'needs_review')

    def test_repeated_phrase_inside_cue_is_flagged(self):
        for text in ('好，' * 8, '因为所以' * 6, 'Thank you! ' * 6):
            quality = check_quality([{'start': 0, 'end': 6, 'text': text}])
            self.assertIn('repeated_phrase', {w['code'] for w in quality['warnings']})

    def test_text_duration_mismatch_is_flagged(self):
        dense = check_quality([{'start': 0, 'end': 0.2, 'text': '这是一句无法在这么短的时间内读完的话'}])
        sparse = check_quality([{'start': 0, 'end': 10, 'text': '你好'}])
        self.assertIn('dense_text', {w['code'] for w in dense['warnings']})
        self.assertIn('sparse_text', {w['code'] for w in sparse['warnings']})

    def test_timestamp_repairs_require_review_even_when_structure_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = save_subtitles(root / 'repair.srt', root / 'cache' / 'run.json',
                                    {'text': '你好', 'chunks': [{'text': '你好', 'timestamp': (0, None)}]}, {}, 1)
            self.assertEqual(payload['structure_check']['status'], 'passed')
            self.assertEqual(payload['quality_check']['status'], 'needs_review')
            self.assertIn('timestamp_repairs', {w['code'] for w in payload['quality_check']['warnings']})

    def test_command_reports_needs_review_exit_code_and_preserves_srt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / 'input.m4a'
            audio.touch()
            result = {'text': '好' * 8, 'chunks': [{'text': '好' * 8, 'timestamp': (0, 1)}]}
            with patch('bilibili_audio_subtitles.duration_seconds', return_value=1), \
                 patch('bilibili_audio_subtitles.decode_audio', return_value=[0] * 16000), \
                 patch('bilibili_audio_subtitles.transcribe', return_value=(result, {'device': 'cpu'})):
                status = main(['--audio', str(audio), '--project-dir', str(root / 'project'),
                               '--diagnostics-dir', str(root / 'cache')])
            self.assertEqual(status, 2)
            self.assertEqual([p.suffix for p in (root / 'project').iterdir()], ['.srt'])
            diagnostics = json.loads(next((root / 'cache').glob('*.json')).read_text())
            self.assertEqual(diagnostics['structure_check']['status'], 'passed')
            self.assertEqual(diagnostics['quality_check']['status'], 'needs_review')


if __name__ == '__main__':
    unittest.main()
