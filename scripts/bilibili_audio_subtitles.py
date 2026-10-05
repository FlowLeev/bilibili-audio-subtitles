#!/usr/bin/env python3
"""Download one Bilibili video part with yutto and generate Whisper subtitles."""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

MODEL_ID = "openai/whisper-large-v3-turbo"


def log(message):
    print(message, flush=True)


def video_identity(value):
    if re.fullmatch(r"BV[0-9A-Za-z]+|av[0-9]+", value):
        value = f"https://www.bilibili.com/video/{value}/"
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not (
        parsed.hostname == "bilibili.com" or (parsed.hostname or "").endswith(".bilibili.com")
    ):
        raise ValueError("请提供标准 Bilibili /video/ 链接或 BV/av 号。")
    match = re.fullmatch(r"/video/(BV[0-9A-Za-z]+|av[0-9]+)/?", parsed.path)
    if not match:
        raise ValueError("此脚本只支持 /video/BV… 或 /video/av… 链接。")
    part = int(parse_qs(parsed.query).get("p", ["1"])[0])
    if part < 1:
        raise ValueError("分 P 必须是正整数。")
    video_id = match.group(1)
    return f"https://www.bilibili.com/video/{video_id}/?p={part}", f"{video_id}-p{part}"


def duration_seconds(audio):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(audio)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    duration = float(json.loads(result.stdout)["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"音频时长无效：{audio}")
    return duration


def download_audio(url, stem, directory, auth_file=None):
    directory.mkdir(parents=True, exist_ok=True)
    audio = directory / f"{stem}.m4a"
    if audio.is_file() and audio.stat().st_size:
        duration_seconds(audio)
        log(f"复用音频：{audio}")
        return audio
    command = [
        sys.executable, "-m", "yutto", url, "--audio-only", "--no-danmaku",
        "--no-subtitle", "--no-cover", "--no-chapter-info",
        "--output-format-audio-only", "m4a", "-aq", "30280",
        "-d", str(directory), "-tp", stem, "--no-color", "--no-progress",
    ]
    if auth_file:
        command.extend(["--auth-file", str(auth_file)])
    log(f"下载音频到：{audio}")
    subprocess.run(command, check=True, timeout=600)
    if not audio.is_file() or not audio.stat().st_size:
        raise RuntimeError("yutto 没有生成预期音频；检查下载日志、登录要求或网络状态。")
    duration_seconds(audio)
    return audio


def decode_audio(audio, limit=None):
    import numpy as np
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(audio)]
    if limit is not None:
        command.extend(["-t", str(limit)])
    command.extend(["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"])
    result = subprocess.run(command, capture_output=True, check=True, timeout=600)
    samples = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if not samples.size:
        raise RuntimeError("FFmpeg 解码结果为空。")
    return samples


def transcribe(samples, args):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请使用 --device cpu 或修复 CUDA 环境。")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    log(f"加载 {MODEL_ID}，设备 {device}；首次运行需要下载模型。")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    asr = pipeline(
        "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor, device=device,
        torch_dtype=dtype, chunk_length_s=30, batch_size=args.batch_size,
    )
    generation = {"task": "transcribe", "condition_on_prev_tokens": False}
    if args.language:
        generation["language"] = args.language
    log(f"开始转写 {len(samples) / 16000:.2f} 秒音频，return_timestamps=True。")
    with torch.inference_mode():
        return asr(
            {"raw": samples, "sampling_rate": 16000},
            return_timestamps=True, generate_kwargs=generation,
        ), device


def normalize_chunks(chunks, duration):
    """Keep raw chunks separately; fill missing bounds and constrain cue times."""
    cues, notes = [], []
    previous_end = 0.0
    for index, chunk in enumerate(chunks):
        text = " ".join(str(chunk.get("text", "")).split())
        if not text:
            continue
        bounds = chunk.get("timestamp")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"第 {index + 1} 段缺少 timestamp。")
        start, end = bounds
        if start is None:
            start = previous_end
            notes.append(f"chunk {index}: missing start, used previous end")
        if end is None:
            end = next((
                following["timestamp"][0] for following in chunks[index + 1:]
                if following.get("timestamp") and following["timestamp"][0] is not None
            ), duration)
            notes.append(f"chunk {index}: missing end, used next start or audio duration")
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"第 {index + 1} 段时间戳不是有限数字。")
        fixed_start = min(duration, max(0.0, previous_end, start))
        fixed_end = min(duration, max(fixed_start, end))
        start_ms, end_ms = round(fixed_start * 1000), round(fixed_end * 1000)
        if (fixed_start, fixed_end) != (start, end):
            notes.append(f"chunk {index}: clamped timestamps to ordered audio bounds")
        if end_ms <= start_ms:
            raise ValueError(f"第 {index + 1} 段没有有效时长；原始结果保留在 JSON，请检查。")
        cues.append({"start": start_ms / 1000, "end": end_ms / 1000, "text": text})
        previous_end = end_ms / 1000
    if not cues:
        raise ValueError("Whisper 未返回非空的带时间戳语音片段；没有生成字幕。")
    return cues, notes


def timecode(seconds, separator):
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{separator}{milliseconds:03d}"


def save_subtitles(paths, result, metadata, duration):
    # Preserve raw output even if validation rejects subtitle creation.
    payload = {**metadata, "text": result.get("text", ""), "chunks": result.get("chunks", [])}
    paths["json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cues, notes = normalize_chunks(result.get("chunks", []), duration)
    payload.update({"segments": cues, "timestamp_repairs": notes})
    paths["json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    srt, vtt = [], ["WEBVTT\n"]
    for number, cue in enumerate(cues, 1):
        srt.append(f"{number}\n{timecode(cue['start'], ',')} --> {timecode(cue['end'], ',')}\n{cue['text']}\n")
        vtt.append(f"{timecode(cue['start'], '.')} --> {timecode(cue['end'], '.')}\n{cue['text']}\n")
    paths["srt"].write_text("\n".join(srt) + "\n", encoding="utf-8")
    paths["vtt"].write_text("\n".join(vtt) + "\n", encoding="utf-8")
    paths["txt"].write_text(result.get("text", "").strip() + "\n", encoding="utf-8")
    log(f"已生成 {len(cues)} 段字幕，时间戳修正 {len(notes)} 项（详见 JSON）。")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是有限正数")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", help="标准 Bilibili 视频链接或 BV/av 号")
    parser.add_argument("--audio", type=Path, help="继续转写本地音频，不传 URL")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--download-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output-name")
    parser.add_argument("--language", help="默认自动识别；可传 chinese / english 等")
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda:N")
    parser.add_argument("--threads", type=positive_int, default=8)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--limit-seconds", type=positive_float, help="仅转写开头片段，输出标记 sample")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--auth-file", type=Path)
    args = parser.parse_args(argv)
    if bool(args.url) == bool(args.audio):
        parser.error("必须提供 URL 或 --audio，二选一。")
    if args.download_only and args.audio:
        parser.error("--download-only 需要 URL。")
    if not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", args.device):
        parser.error("--device 只支持 auto、cpu、cuda 或 cuda:N。")
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RuntimeError(f"PATH 中缺少 {tool}。")
    if args.auth_file:
        args.auth_file = args.auth_file.expanduser().resolve(strict=True)
    url, download_stem = video_identity(args.url) if args.url else (None, args.audio.stem)
    stem = args.output_name or download_stem
    if stem in {".", ".."} or not stem.strip() or "/" in stem or "\\" in stem:
        raise ValueError("--output-name 必须是非空文件名，不能含路径。")
    if args.limit_seconds is not None:
        stem += f"-sample-{args.limit_seconds:g}s"
    project = args.project_dir.expanduser().resolve()
    paths = {suffix: project / f"{stem}.{suffix}" for suffix in ("srt", "vtt", "txt", "json")}
    if not args.download_only and not args.overwrite:
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError("拒绝覆盖已有结果；使用新名字或 --overwrite：" + ", ".join(existing))
    audio = args.audio.expanduser().resolve(strict=True) if args.audio else download_audio(
        url, download_stem, args.download_dir.expanduser().resolve(), args.auth_file,
    )
    duration = duration_seconds(audio)
    log(f"音频：{audio}；完整时长 {duration:.2f} 秒。")
    if args.download_only:
        return 0
    samples = decode_audio(audio, args.limit_seconds)
    transcribed_duration = len(samples) / 16000
    result, device = transcribe(samples, args)
    project.mkdir(parents=True, exist_ok=True)
    save_subtitles(paths, result, {
        "model": MODEL_ID, "return_timestamps": True, "task": "transcribe",
        "source_url": url, "audio_path": str(audio), "language": args.language or "auto",
        "device": device, "audio_duration_seconds": duration,
        "transcribed_duration_seconds": transcribed_duration,
        "partial": args.limit_seconds is not None and transcribed_duration < duration - 0.01,
    }, transcribed_duration)
    for path in paths.values():
        log(f"输出：{path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"失败：{error}", file=sys.stderr, flush=True)
        sys.exit(1)
