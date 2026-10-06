"""Offline tests for platform routing and bounded, update-first download recovery."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bilibili_audio_subtitles import download_audio, run_download, source_identity, update_downloader


class DownloaderTests(unittest.TestCase):
    def test_youtube_urls_select_the_same_single_video(self):
        identity = ('youtube', 'https://www.youtube.com/watch?v=jNQXAC9IVRw', 'youtube-jNQXAC9IVRw')
        for url in (
            'https://www.youtube.com/watch?v=jNQXAC9IVRw&list=PLAYLIST&t=3',
            'https://youtu.be/jNQXAC9IVRw?si=tracking',
            'https://www.youtube.com/shorts/jNQXAC9IVRw',
            'https://m.youtube.com/live/jNQXAC9IVRw',
            'https://music.youtube.com/watch?v=jNQXAC9IVRw',
            'https://www.youtube.com/embed/jNQXAC9IVRw',
        ):
            self.assertEqual(source_identity(url), identity)

    def test_playlist_only_and_lookalike_hosts_are_rejected(self):
        for url in ('https://www.youtube.com/playlist?list=PLAYLIST',
                    'https://youtube.com.evil.test/watch?v=jNQXAC9IVRw',
                    'https://www.youtube.com/watch?v=bad',
                    'ftp://youtu.be/jNQXAC9IVRw'):
            with self.assertRaises(ValueError):
                source_identity(url)

    def test_bilibili_routing_preserves_the_part(self):
        self.assertEqual(source_identity('https://www.bilibili.com/video/BV1w88n6uE1X/?p=2&tracking=1'),
                         ('bilibili', 'https://www.bilibili.com/video/BV1w88n6uE1X/?p=2', 'BV1w88n6uE1X-p2'))

    def test_youtube_command_uses_audio_only_and_does_not_expand_playlist(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch('bilibili_audio_subtitles.run_download') as run, \
                 patch('bilibili_audio_subtitles.shutil.which', side_effect=lambda name: '/test/node' if name == 'node' else None):
                download_audio('https://www.youtube.com/watch?v=jNQXAC9IVRw', 'youtube-jNQXAC9IVRw',
                               Path(temporary), cookies_file=Path('/user/cookies.txt'))
            command, package, audio, _ = run.call_args.args
            self.assertEqual(package, 'yt-dlp')
            self.assertEqual(command[2], 'yt_dlp')
            self.assertIn('--no-playlist', command)
            self.assertEqual(command[command.index('-f') + 1], 'bestaudio')
            self.assertEqual(command[command.index('--audio-format') + 1], 'm4a')
            self.assertIn('--no-write-subs', command)
            self.assertIn('--no-write-info-json', command)
            self.assertIn('node:/test/node', command)
            self.assertEqual(command[command.index('--cookies') + 1], '/user/cookies.txt')
            self.assertEqual(audio.name, 'youtube-jNQXAC9IVRw.m4a')

    def test_bilibili_command_still_uses_yutto(self):
        with tempfile.TemporaryDirectory() as temporary, patch('bilibili_audio_subtitles.run_download') as run:
            download_audio('https://www.bilibili.com/video/BV1w88n6uE1X/?p=1', 'BV1w88n6uE1X-p1',
                           Path(temporary), auth_file=Path('/user/auth.json'))
            command, package, _, _ = run.call_args.args
            self.assertEqual(package, 'yutto')
            self.assertEqual(command[2], 'yutto')
            self.assertIn('--audio-only', command)
            self.assertIn('--auth-file', command)

    def test_valid_cached_audio_skips_downloading_and_update_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary) / 'youtube-jNQXAC9IVRw.m4a'
            audio.write_bytes(b'cached')
            with patch('bilibili_audio_subtitles.duration_seconds', return_value=18), \
                 patch('bilibili_audio_subtitles.run_download') as run, \
                 patch('bilibili_audio_subtitles.update_downloader') as update:
                actual = download_audio('https://www.youtube.com/watch?v=jNQXAC9IVRw', audio.stem, Path(temporary))
            self.assertEqual(actual, audio)
            run.assert_not_called()
            update.assert_not_called()

    def test_update_check_precedes_only_one_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary) / 'audio.m4a'
            order, events = [], []

            def run(command, **kwargs):
                order.append('download')
                if len(order) == 1:
                    raise subprocess.CalledProcessError(1, command)
                audio.write_bytes(b'valid')

            def update(package):
                order.append('update')
                return {'status': 'updated', 'package': package}

            with patch('bilibili_audio_subtitles.subprocess.run', side_effect=run), \
                 patch('bilibili_audio_subtitles.update_downloader', side_effect=update), \
                 patch('bilibili_audio_subtitles.duration_seconds', return_value=18):
                run_download(['downloader'], 'yt-dlp', audio, events)
            self.assertEqual(order, ['download', 'update', 'download'])
            self.assertEqual([event['status'] for event in events], ['failed', 'downloaded'])

    def test_second_failure_stops_without_further_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            error = subprocess.CalledProcessError(1, ['downloader'])
            with patch('bilibili_audio_subtitles.subprocess.run', side_effect=error) as run, \
                 patch('bilibili_audio_subtitles.update_downloader', return_value={'status': 'updated'}) as update:
                with self.assertRaisesRegex(RuntimeError, '重试仍失败'):
                    run_download(['downloader'], 'yutto', Path(temporary) / 'audio.m4a', [])
            self.assertEqual(run.call_count, 2)
            update.assert_called_once_with('yutto')

    def test_no_update_or_failed_check_does_not_retry(self):
        for status in ('up_to_date', 'check_failed', 'upgrade_failed'):
            with tempfile.TemporaryDirectory() as temporary:
                with patch('bilibili_audio_subtitles.subprocess.run', side_effect=subprocess.TimeoutExpired(['downloader'], 600)) as run, \
                     patch('bilibili_audio_subtitles.update_downloader', return_value={'status': status}):
                    with self.assertRaisesRegex(RuntimeError, status):
                        run_download(['downloader'], 'yt-dlp', Path(temporary) / 'audio.m4a', [])
                self.assertEqual(run.call_count, 1)

    def test_update_targets_only_the_failed_package_in_the_current_environment(self):
        responses = [subprocess.CompletedProcess([], 0, stdout=json.dumps({'latest': '2026.9.1'})),
                     subprocess.CompletedProcess([], 0)]
        with patch('bilibili_audio_subtitles.package_version', side_effect=['2026.8.19', '2026.9.1']), \
             patch('bilibili_audio_subtitles.subprocess.run', side_effect=responses) as run:
            event = update_downloader('yt-dlp')
        self.assertEqual(event['status'], 'updated')
        query, install = [call.args[0] for call in run.call_args_list]
        self.assertEqual(query[0], install[0])
        self.assertEqual(query[2:5], ['pip', 'index', 'versions'])
        self.assertIn('yt-dlp', query)
        self.assertIn('yt-dlp[default]==2026.9.1', install)
        self.assertNotIn('--pre', install)
        self.assertNotIn('torch', install)

    def test_latest_stable_does_not_downgrade_a_newer_installed_release(self):
        with patch('bilibili_audio_subtitles.package_version', return_value='2026.10.1.dev1'), \
             patch('bilibili_audio_subtitles.subprocess.run', return_value=subprocess.CompletedProcess(
                 [], 0, stdout=json.dumps({'latest': '2026.9.1'}))) as run:
            event = update_downloader('yt-dlp')
        self.assertEqual(event['status'], 'up_to_date')
        self.assertEqual(run.call_count, 1)

    def test_failed_index_query_does_not_install(self):
        with patch('bilibili_audio_subtitles.package_version', return_value='2026.8.19'), \
             patch('bilibili_audio_subtitles.subprocess.run', side_effect=subprocess.TimeoutExpired(['pip'], 60)) as run:
            self.assertEqual(update_downloader('yt-dlp')['status'], 'check_failed')
        self.assertEqual(run.call_count, 1)

    def test_absent_output_is_a_download_failure_even_with_zero_exit_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch('bilibili_audio_subtitles.subprocess.run', return_value=subprocess.CompletedProcess([], 0)), \
                 patch('bilibili_audio_subtitles.update_downloader', return_value={'status': 'up_to_date'}) as update:
                with self.assertRaises(RuntimeError):
                    run_download(['downloader'], 'yutto', Path(temporary) / 'missing.m4a', [])
            update.assert_called_once_with('yutto')


if __name__ == '__main__':
    unittest.main()
