"""Check the known Bilibili segmentation/repetition regression using local reports."""
import argparse
import json
from pathlib import Path

from bilibili_audio_subtitles import check_quality, normalize_chunks

WINDOWS = [(43.24, 346.06), (490.0, 513.0)]


def summarize(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    cues = data.get("segments")
    if cues is None:
        cues, _ = normalize_chunks(data["chunks"], data["transcribed_duration_seconds"])
    quality = check_quality(cues)
    windows = []
    for start, end in WINDOWS:
        matching = [cue for cue in cues if cue["start"] < end and cue["end"] > start]
        windows.append({
            "start": start, "end": end, "cue_count": len(matching),
            "max_cue_seconds": max((cue["end"] - cue["start"] for cue in matching), default=0),
            "warnings": [warning for warning in quality["warnings"]
                         if warning["start"] < end and warning["end"] > start],
        })
    return data, {"cue_count": len(cues), "max_cue_seconds": quality["max_cue_seconds"],
                  "quality_status": quality["status"], "windows": windows}


def check_candidate(data, report):
    if data.get("transcription_mode") != "native-long-form" or data.get("partial") is not False:
        raise ValueError("回归候选必须是完整音频的原生长音频转写。")
    if data.get("structure_check", {}).get("status") != "passed":
        raise ValueError("候选结果的结构检查未通过。")
    if data.get("quality_check", {}).get("status") != "no_obvious_anomaly":
        raise ValueError("候选结果已有质量告警，需要复核。")
    if report["quality_status"] != "no_obvious_anomaly":
        raise ValueError("重新运行质量检查时发现异常。")
    for window in report["windows"]:
        if not window["cue_count"] or window["max_cue_seconds"] > 12 or window["warnings"]:
            raise ValueError(f"重点时间窗回归未通过：{window}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path, help="新版缓存诊断 JSON")
    parser.add_argument("--baseline", type=Path, help="可选，旧版保留的 JSON")
    args = parser.parse_args()
    candidate, summary = summarize(args.candidate)
    reports = {"candidate": summary}
    if args.baseline:
        _, reports["baseline"] = summarize(args.baseline)
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    check_candidate(candidate, summary)
    print("分段与重复问题回归通过；此检查不等同于术语或逐句准确性校对。")


if __name__ == "__main__":
    main()
