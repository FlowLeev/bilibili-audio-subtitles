"""Qwen ASR/forced alignment and SRT grouping without fabricated word times."""
from dataclasses import asdict
from importlib.metadata import version
import math
import re
import sys
import time
import unicodedata

MODEL_ID = 'Qwen/Qwen3-ASR-1.7B'
ALIGNER_ID = 'Qwen/Qwen3-ForcedAligner-0.6B'
ALIGNER_LANGUAGES = {'Chinese', 'English', 'Cantonese', 'French', 'German', 'Italian',
                     'Japanese', 'Korean', 'Portuguese', 'Russian', 'Spanish'}
LANGUAGE_CODES = dict(zip(
    ['zh', 'en', 'yue', 'fr', 'de', 'it', 'ja', 'ko', 'pt', 'ru', 'es'],
    ['Chinese', 'English', 'Cantonese', 'French', 'German', 'Italian',
     'Japanese', 'Korean', 'Portuguese', 'Russian', 'Spanish']))


def timestamp_language(value):
    if value is None:
        return None
    language = LANGUAGE_CODES.get(value.strip().lower(), value.strip().capitalize())
    if language not in ALIGNER_LANGUAGES:
        raise ValueError(f'Qwen 时间戳对齐不支持 {value}；请显式使用 --engine whisper。')
    return language


def kept(ch):
    return ch == "'" or unicodedata.category(ch)[0] in {'L', 'N'}


def group_timestamps(text, items, max_seconds=6.0, max_characters=42, gap_seconds=0.6):
    positions=[i for i,ch in enumerate(text) if kept(ch)]
    clean=''.join(text[i] for i in positions)
    units=[]; cursor=0; previous_start=0.0
    for i,item in enumerate(items):
        token=''.join(ch for ch in item['text'] if kept(ch))
        if not token or clean[cursor:cursor+len(token)] != token:
            raise ValueError(f'对齐文字与原始转写不匹配：第 {i+1} 个单位 {item["text"]!r}')
        end_cursor=cursor+len(token)
        lo=0 if cursor==0 else positions[cursor]
        hi=positions[end_cursor] if end_cursor<len(positions) else len(text)
        start,end=float(item['start_time']),float(item['end_time'])
        if not math.isfinite(start) or not math.isfinite(end) or start<previous_start or start<0 or end<start:
            raise ValueError(f'第 {i+1} 个对齐时间无效')
        units.append({'text':text[lo:hi], 'start':start,'end':end})
        cursor=end_cursor;previous_start=start
    if cursor!=len(clean):
        raise ValueError('原始转写包含未对齐的文字，不能丢弃')
    chunks=[];group=[]
    def flush():
        if group:
            chunks.append({'text':''.join(u['text'] for u in group).strip(), 'timestamp':[group[0]['start'],max(u['end'] for u in group)]})
            group.clear()
    for u in units:
        if group and (u['start']-group[-1]['end']>=gap_seconds or u['end']-group[0]['start']>max_seconds or sum(len(x['text']) for x in group)+len(u['text'])>max_characters):
            # Keep zero-duration units with their neighbours; do not fabricate times.
            if max(x['end'] for x in group)>group[0]['start']:
                flush()
        group.append(u)
        span=max(x['end'] for x in group)-group[0]['start']
        if span>0 and (re.search(r'[。！？!?;；]|\.(?=\s|$)', u['text']) or (span>=2 and any(ch in u['text'] for ch in '，,'))):
            flush()
    # Quantized alignment can assign zero duration to a final word. Keep that
    # word with the preceding cue, using only existing start/end predictions.
    if group and max(u['end'] for u in group) == group[0]['start'] and chunks:
        tail = ''.join(u['text'] for u in group).strip()
        previous = chunks[-1]['text']
        separator = ' ' if previous[-1].isascii() and previous[-1].isalnum() and tail[0].isascii() and tail[0].isalnum() else ''
        chunks[-1]['text'] += separator + tail
        chunks[-1]['timestamp'][1] = max(chunks[-1]['timestamp'][1], max(u['end'] for u in group))
        group.clear()
    flush()
    return chunks


def transcribe_qwen(samples, args):
    import torch
    import transformers
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as error:
        raise RuntimeError('codex 环境缺少 Qwen 依赖；请安装 qwen-asr==0.0.6。') from error
    language = timestamp_language(args.language)
    torch.set_num_threads(args.threads)
    device = args.device
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用，请使用 --device cpu 或修复 CUDA 环境。')
    dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
    print(f'加载 {MODEL_ID} 和 {ALIGNER_ID}，设备 {device}。', flush=True)
    model = Qwen3ASRModel.from_pretrained(
        MODEL_ID, dtype=dtype, device_map=device, attn_implementation='sdpa',
        max_inference_batch_size=args.batch_size, max_new_tokens=4096,
        forced_aligner=ALIGNER_ID,
        forced_aligner_kwargs=dict(dtype=dtype, device_map=device, attn_implementation='sdpa'))
    raw_outputs = []
    raw_chunks = []
    infer = model._infer_asr
    def capture(contexts, wavs, languages):
        outputs = infer(contexts, wavs, languages)
        raw_outputs.extend(outputs)
        offset = 0.0
        for text, wav in zip(outputs, wavs):
            end = min(len(samples)/16000, offset + len(wav)/16000)
            raw_chunks.append({'text': text, 'start': offset, 'end': end})
            offset += len(wav)/16000
        return outputs
    # The pinned SDK postprocesses severe repetitions; retain the original
    # decoded strings so that postprocessing cannot conceal quality problems.
    model._infer_asr = capture
    configuration = {
        'engine': 'qwen', 'model': MODEL_ID, 'aligner_model': ALIGNER_ID,
        'device': device, 'dtype': str(dtype),
        'transcription_mode': 'qwen-asr-forced-alignment',
        'generation_parameters': {'max_new_tokens': 4096, 'return_time_stamps': True,
                                  'language': language, 'context': ''},
        'model_generation_config': model.model.generation_config.to_dict(),
        'model_revision': getattr(model.model.config, '_commit_hash', None),
        'aligner_revision': getattr(model.forced_aligner.model.config, '_commit_hash', None),
        'pipeline_parameters': {'backend': 'transformers', 'attn_implementation': 'sdpa',
                                'batch_size': args.batch_size, 'sdk_max_alignment_chunk_seconds': 180},
        'subtitle_grouping': {'max_seconds': 6, 'max_characters': 42, 'gap_seconds': 0.6},
        'versions': {'python': sys.version.split()[0], 'torch': torch.__version__,
                     'transformers': transformers.__version__, 'qwen_asr': version('qwen-asr')},
    }
    print(f'开始转写并对齐 {len(samples)/16000:.2f} 秒音频，return_time_stamps=True。', flush=True)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    failure = None
    try:
        with torch.inference_mode():
            output = model.transcribe(audio=(samples, 16000), language=language, return_time_stamps=True)[0]
    except (ValueError, RuntimeError) as error:
        failure = str(error)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
        configuration['peak_memory_gib'] = torch.cuda.max_memory_allocated()/1024**3
    configuration['inference_seconds'] = time.perf_counter()-started
    if failure is not None:
        # Alignment may fail after ASR has completed (e.g. an automatically
        # detected language unsupported by the aligner). Preserve that ASR.
        return {'text': '', 'chunks': [], 'raw_asr_outputs': raw_outputs,
                'raw_asr_chunks': raw_chunks,
                'alignment_error': failure + '；请检查对齐语言或显式使用 --engine whisper。'}, configuration
    raw = asdict(output)
    items = (raw.get('time_stamps') or {}).get('items', [])
    result = {'text': output.text, 'chunks': [], 'raw_alignment': raw.get('time_stamps'),
              'raw_asr_outputs': raw_outputs, 'raw_asr_chunks': raw_chunks,
              'detected_language': output.language,
              'alignment_statistics': {'unit_count': len(items),
                  'zero_duration_units': sum(u['start_time'] == u['end_time'] for u in items)},
              'model_quality_warnings': []}
    for index, text in enumerate(raw_outputs):
        window = {k: raw_chunks[index][k] for k in ['start', 'end']}
        compact = ''.join(ch.casefold() for ch in text if ch.isalnum())
        if re.search(r'(.{1,20}?)\1{4,}', compact):
            result['model_quality_warnings'].append({
                'code': 'raw_repeated_phrase', 'asr_chunk': index + 1, **window,
                'detail': '原始模型输出存在连续重复；SDK 后处理不能作为准确性证明。'})
        if len(model.processor.tokenizer.encode(text)) >= 4095:
            result['model_quality_warnings'].append({
                'code': 'generation_limit', 'asr_chunk': index + 1, **window,
                'detail': '输出接近生成长度限制，需核对是否截断。'})
    try:
        result['chunks'] = group_timestamps(output.text, items)
    except (ValueError, KeyError, TypeError) as error:
        result['alignment_error'] = str(error)
    return result, configuration
