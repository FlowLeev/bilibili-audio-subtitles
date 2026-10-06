"""Check preservation and timing when turning aligned words into SRT cues."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from bilibili_audio_subtitles import normalize_chunks, save_subtitles, transcribe
from qwen_subtitles import group_timestamps, timestamp_language


def item(text, start, end):
    return dict(text=text, start_time=start, end_time=end)


class QwenTests(unittest.TestCase):
    def test_chinese_punctuation_and_actual_word_times_are_preserved(self):
        text='“鼠标”，订单流。'
        words=[item(ch, i*0.3, (i+1)*0.3) for i,ch in enumerate('鼠标订单流')]
        chunks=group_timestamps(text, words)
        self.assertEqual(''.join(c['text'] for c in chunks), text)
        self.assertEqual(chunks[0]['timestamp'], [0, 1.5])

    def test_english_spaces_apostrophes_and_sentence_boundaries(self):
        text="It's cool. And that's all."
        words=[item(t, i, i+0.7) for i,t in enumerate(["It's",'cool','And',"that's",'all'])]
        chunks=group_timestamps(text, words)
        self.assertEqual([c['text'] for c in chunks], ["It's cool.", "And that's all."])
        self.assertEqual(chunks[1]['timestamp'], [2, 4.7])

    def test_grouping_uses_model_boundaries_and_keeps_every_character(self):
        text='一二三四五六七八九十'
        words=[item(ch, i, i+0.8) for i,ch in enumerate(text)]
        chunks=group_timestamps(text, words, max_seconds=3)
        self.assertEqual(''.join(c['text'] for c in chunks), text)
        self.assertTrue(all(end-start<=3 for start,end in [c['timestamp'] for c in chunks]))
        predicted={u['start_time'] for u in words}|{u['end_time'] for u in words}
        self.assertTrue(all(t in predicted for c in chunks for t in c['timestamp']))

    def test_zero_duration_final_word_is_kept_without_inventing_time(self):
        chunks=group_timestamps('All to say.', [item('All',0,1),item('to',1,2),item('say',2,2)], max_characters=7)
        cues, notes=normalize_chunks(chunks, 3)
        self.assertEqual(cues,[{'text':'All to say.','start':0,'end':2}])
        self.assertEqual(notes, [])

    def test_fully_zero_duration_alignment_cannot_pass_structure(self):
        chunks=group_timestamps('你好', [item('你',1,1),item('好',1,1)])
        with self.assertRaises(ValueError):
            normalize_chunks(chunks, 3)

    def test_mismatched_missing_or_invalid_alignment_is_rejected(self):
        for words in ([item('错',0,1)], [item('你',0,1)],
                      [item('你',2,3),item('好',1,2)],
                      [item('你好',float('nan'),3)]):
            with self.assertRaises(ValueError):
                group_timestamps('你好', words)

    def test_supported_language_aliases_and_unsupported_timestamp_language(self):
        self.assertIsNone(timestamp_language(None))
        self.assertEqual(timestamp_language('zh'), 'Chinese')
        self.assertEqual(timestamp_language('eNGLISH'), 'English')
        with self.assertRaisesRegex(ValueError, '--engine whisper'):
            timestamp_language('Thai')

    def test_default_adapter_dispatch_does_not_load_whisper(self):
        with patch('qwen_subtitles.transcribe_qwen',return_value=({'text':'你好'},{})) as qwen:
            args=SimpleNamespace(engine='qwen')
            self.assertEqual(transcribe([0],args),({'text':'你好'},{}))
            qwen.assert_called_once_with([0],args)

    def test_raw_repetition_alert_survives_sdk_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            result={'text':'好', 'chunks':[{'text':'好','timestamp':[0,1]}],
                    'raw_asr_outputs':['好'*30], 'raw_alignment':{'items':[item('好',0,1)]},
                    'model_quality_warnings':[{'code':'raw_repeated_phrase','detail':'原始结果重复'}]}
            p=save_subtitles(root/'out.srt',root/'run.json',result,{},1)
            self.assertEqual(p['quality_check']['status'],'needs_review')
            self.assertEqual(json.loads((root/'run.json').read_text())['raw_asr_outputs'],['好'*30])

    def test_alignment_failure_preserves_original_text_and_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            result={'text':'原始文字','chunks':[], 'raw_alignment':{'items':[]},'alignment_error':'未对齐'}
            with self.assertRaisesRegex(ValueError,'未对齐'):
                save_subtitles(root/'out.srt',root/'run.json',result,{},3)
            raw=json.loads((root/'run.json').read_text())
            self.assertEqual(raw['text'],'原始文字')
            self.assertEqual(raw['raw_alignment'],{'items':[]})
            self.assertFalse((root/'out.srt').exists())


if __name__=='__main__':
    unittest.main()
