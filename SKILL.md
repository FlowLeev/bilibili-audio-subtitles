---
name: bilibili-audio-subtitles
description: 用 yutto 下载 Bilibili 音频、用 yt-dlp 下载 YouTube 音频，再用 Hugging Face openai/whisper-large-v3-turbo 生成一份 SRT 并检查质量。用于 B 站或 YouTube 链接转字幕、继续转写已有音频；下载失败时优先检查对应下载包更新。
---

# Bilibili / YouTube 音频转字幕

使用名为 `codex` 的 conda 环境；模型固定为 `openai/whisper-large-v3-turbo`，调用 Transformers ASR pipeline 的 `return_timestamps=True`，转写原语言。默认使用 **Whisper 原生长音频模式**，不传 `chunk_length_s`。实验性外部分块曾在回归样例中产生五分钟字幕段和连续重复，不应默认启用。

## 运行与输出

先记录用户当前项目的绝对路径，不要把 skill 安装目录当成项目目录。将 `<skill目录>` 替换为本 SKILL.md 所在目录，将 `<项目目录>` 替换为调用时的项目路径；URL 作为单个带引号的参数传入。

```bash
conda run --no-capture-output -n codex python "<skill目录>/scripts/bilibili_audio_subtitles.py" \
  'https://www.bilibili.com/video/BV1w88n6uE1X/' \
  --project-dir '<项目目录>'
```

同一命令也接受 YouTube 链接，例如 `'https://youtu.be/jNQXAC9IVRw'`。保留原有 skill 名称与脚本名，已有 `$bilibili-audio-subtitles` 调用可继续使用。

Bilibili 音频由 **yutto** 下载，保存为 `~/Downloads/<BV号或av号>-p<N>.m4a`；YouTube 音频由 **yt-dlp** 下载，保存为 `~/Downloads/youtube-<视频ID>.m4a`。**项目目录每次只输出一份 SRT**，不生成 VTT、TXT 或项目内 JSON。原始模型 text/chunks 和复现诊断保存在 `~/.cache/bilibili-audio-subtitles/runs/` 的独立 JSON 中，运行日志会提供路径；这是内部诊断记录，不是额外的字幕交付文件。

已有音频会复用。已有同名 SRT 时，新结果自动使用包含模式、脚本版本和运行编号的独立文件名，保留旧版本；不覆盖或删除旧字幕。重试后向用户明确指出推荐使用的 SRT 版本，不仅列出所有路径。

常用选项：

- `--audio '/绝对路径/音频.m4a'`：转写本地音频，不同时传 URL。
- `--language chinese`：已知语言时明确指定，默认自动识别。
- `--project-dir` / `--download-dir`：覆盖项目输出目录或 Downloads。
- `--device auto|cpu|cuda:0`：默认优先 CUDA；不要自动改动系统驱动。
- `--download-only`：只下载，之后使用 `--audio` 继续。
- `--output-name '文件名'`：自定义 SRT 文件名，不含路径或扩展名。
- `--diagnostics-dir '/绝对路径'`：覆盖内部原始结果及诊断目录。
- `--auth-file '/绝对路径'`：仅用于用户已提供的 Bilibili/yutto 认证文件。
- `--cookies-file '/绝对路径'`：仅用于用户已提供的 YouTube Netscape Cookie 文件，不扫描浏览器。
- `--max-cue-seconds 12`：超长字幕告警阈值，默认 12 秒。
- `--limit-seconds 30`：用户要求的片段或冒烟测试；文件名标记 sample，报告中明确不是完整转写。
- `--threads 8` / `--batch-size 1`：控制线程和推理批大小。
- `--chunk-length-seconds 30`：仅在有明确理由且说明实验性风险时启用外部分块；常规转写不使用。

支持 Bilibili 标准 `/video/BV…`、`/video/av…` 链接和直接 BV/av 号。保留 `?p=N` 指定的分 P，移除追踪参数。默认只处理一 P；多 P 请求逐 P 调用。B 站短链接先解析为标准链接。

YouTube 支持 `youtube.com/watch?v=…`、`youtu.be/…`、`shorts/…`、`live/…` 和 `embed/…`（包括 www、m、music 子域）。统一为单视频标准链接，移除追踪、起播和播放列表参数，并显式使用 `--no-playlist`；仅有播放列表的链接直接拒绝。下载选择 `bestaudio`，经 FFmpeg 提取为 M4A，不下载平台字幕、封面或视频作为交付文件。不扫描浏览器、不打印 Cookie，不擅自扩大到整套合集。

## 下载失败优先检查更新

此规则同时适用于两种平台。实际下载失败或下载器返回成功却没有有效音频时，脚本先查询**对应下载包**的兼容稳定版本：Bilibili 检查 `yutto`，YouTube 检查 `yt-dlp`。用当前 `codex` Python 的 `pip index versions <包> --json` 比较本机和包索引版本，尊重已有 pip 镜像配置。

若有新版，脚本只在该环境升级这个下载包及其必要依赖，然后重试下载**一次**。yt-dlp 使用 `yt-dlp[default]`，保持 YouTube EJS 依赖匹配。无新版、查询失败、升级失败或第二次下载仍失败时，保存原始错误与更新事件并停止；不要无限重试或自动切到 nightly/master。检查包更新之后，再根据真实错误检查网络、登录、视频是否下架、地区限制和 JS 运行时。

更新需要联网和环境写权限。权限受限时申请本次必要权限，不更换系统 Python、不为下载问题更新 Whisper/PyTorch。更新查询失败不等于已经是最新版。已有损坏音频须明确报告并保留，不直接覆盖；失败重试新产生的无效音频也保留独立名称。

## 结构检查与质量检查

两类检查必须分别报告：

- **结构检查**：完整音频解码时长匹配，非空字幕，时间戳有限、有序、起止有效且不超过转写时长。失败时保留原始诊断，不生成 SRT。
- **质量检查**：字幕超过 12 秒、单条超过 120 个文字字符、文字密度超过 25 字符/秒、持续至少 8 秒但低于 0.5 字符/秒、单条连续重复词句至少 5 次、连续至少 3 条同文字幕。时间戳需要修正也会标记复核。阈值和告警明细写入诊断。

退出码 `0` 表示结构通过且未发现明显质量异常；退出码 `2` 表示 SRT 已生成但**需要复核**；退出码 `1` 表示运行或结构失败。不要把退出码 2 宣称为检查成功。质量规则是启发式提示，不是语音内容正确性的证明；误报须对照音频判断，不为消除告警擅自调高阈值、删除文字、按等分时长重切或无限重试。

质量异常时给出告警时间窗与诊断路径。若异常来自显式外部分块，可用默认原生模式重试一次，保留两轮输出；原生模式仍异常时说明需复核，不自动改模型。**术语纠正单独处理**，不要凭猜测全局替换“订单流”“鼠标”等词。即使没有告警，也只称为机器转写草稿，不能称为逐句校对过的字幕。

诊断 JSON 保留原始 text/chunks、规范化 segments、结构状态、质量状态和告警、修正记录、推荐状态、脚本版本、转写模式、生成参数、模型 GenerationConfig 与 revision、库版本、设备、时长和推理耗时。还记录来源平台、下载包版本、更新检查和重试事件；下载失败也会保存独立诊断，供复现比较。

## 环境

检查 `codex` 环境的 Python、对应平台下载包、torch、transformers、accelerate、numpy，以及 PATH 中的 ffmpeg 和 ffprobe。只将缺少依赖安装到该环境：

```bash
conda run --no-capture-output -n codex python -m pip install yutto 'yt-dlp[default]' 'transformers>=4.46,<5' accelerate numpy
```

保留已有可用的 CUDA PyTorch。本机验证环境：Python 3.12、yutto 2.3.1、Transformers 4.57.6、PyTorch 2.14.1+cu130。若需重建本机 GPU 环境：

```bash
conda run --no-capture-output -n codex python -m pip install 'torch==2.14.1+cu130' --index-url https://download.pytorch.org/whl/cu130
```

若 shell 找不到 conda，定位其绝对路径。本机可使用 `/home/leev/miniforge3/bin/conda` 或 `/home/leev/miniforge3/envs/codex/bin/python`。受限沙箱内 GPU 检测失败时，先在允许访问 NVIDIA 设备的环境重新检测，不直接判断驱动损坏或改装 CPU 版。真正无 GPU 的机器才使用匹配的 CPU PyTorch。

YouTube 还需要 yt-dlp 的默认 EJS 支持依赖及受支持的 JS 运行时。脚本会查找 PATH 中的 Deno、Node、Bun 或 QuickJS，并显式传给 yt-dlp；本机已验证 Node 24 与 yt-dlp 2026.08.19。找不到运行时或版本不兼容时明确报告，不因下载失败反复重装其他包。

首次运行会下载模型到 Hugging Face 标准缓存，尊重 `HF_HOME` / `HF_HUB_CACHE`；已有完整缓存且网络不稳定时可设置 `HF_HUB_OFFLINE=1`。B 站仅用 yutto 下载，禁用视频、弹幕、平台字幕、封面和章节信息；YouTube 仅用 yt-dlp 下载音频。下载失败先按更新规则处理再报告真实错误；转写失败时保留音频。权限受限时申请本次必要的目录和 GPU 执行权限。

## 回归验证

固定样例：[BV1w88n6uE1X](https://www.bilibili.com/video/BV1w88n6uE1X/)。用同一音频运行完整原生转写，保留旧版诊断并调用：

```bash
python -m unittest discover -s '<skill目录>/scripts' -p 'test_*.py'
python '<skill目录>/scripts/check_regression.py' \
  --candidate '<新版缓存诊断JSON>' --baseline '<旧版JSON>'
```

重点时间窗为 `00:43.240–05:46.060` 和 `08:10–08:33`。实测原生结果为 438 段、最长 5.18 秒，两个窗口无超长和重复告警；不要把这一固定段数作为其他环境的通过条件。旧版 117 段结果中的 302.82 秒字幕和连续 23 条“好”，应被新版质量检查标记为需要复核。回归通过只证明已知分段与重复问题未复现，语义和术语仍需对照音频。

YouTube 回归包含 watch/短链接/Shorts 路由、只下载单个视频的音频、失败优先查询更新、升级后最多重试一次、无更新或查询失败时停止。实测 `jNQXAC9IVRw` 的约 19 秒音频完整下载并生成 4 段 SRT；已不可用的测试视频也验证了更新检查和失败诊断路径。

实现依据：[yutto 官方项目](https://github.com/yutto-dev/yutto)、[yt-dlp 官方项目](https://github.com/yt-dlp/yt-dlp)、[Whisper 模型说明](https://huggingface.co/openai/whisper-large-v3-turbo)。
