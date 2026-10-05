---
name: bilibili-audio-subtitles
description: 用 yutto 下载 Bilibili 视频音频，并用 Hugging Face openai/whisper-large-v3-turbo 生成带时间戳的 SRT/VTT 字幕。用于 B 站链接转字幕、提取音频并转写，以及继续转写本 skill 已下载的音频。
---

# Bilibili 音频转字幕

使用名为 `codex` 的 conda 环境。音频默认保存到 `~/Downloads`，字幕默认保存到调用 skill 时的项目目录。模型固定为 `openai/whisper-large-v3-turbo`，通过 Transformers ASR pipeline 调用 `return_timestamps=True`，转写原语言，默认自动识别语言。

## 运行

先记录用户当前项目的绝对路径；不要把 skill 安装目录当成项目目录。将下面的 `<skill目录>` 替换为本 SKILL.md 所在目录，将 `<项目目录>` 替换为调用时的项目路径。URL 必须作为单个带引号的参数传入。

```bash
conda run --no-capture-output -n codex python "<skill目录>/scripts/bilibili_audio_subtitles.py" \
  'https://www.bilibili.com/video/BV1w88n6uE1X/' \
  --project-dir '<项目目录>'
```

若 shell 找不到 conda，定位其可执行文件后使用绝对路径。本机为 `/home/leev/miniforge3/bin/conda`；也可直接使用 `/home/leev/miniforge3/envs/codex/bin/python`，确保没有误用系统 Python。

输出：`<BV号或av号>-p<N>.m4a` 位于 Downloads；同名 `.srt`、`.vtt`、`.txt`、`.json` 位于项目目录。JSON 保留模型返回的原始文本与 chunks、规范化后的时间戳、音频路径、模型名和实际转写时长。脚本复用已下载的有效音频，默认拒绝覆盖已有转写结果。

常用选项：

- `--language chinese`：明确知道音频语言时使用；默认自动识别。
- `--audio '/绝对路径/音频.m4a'`：直接转写本地音频，无需重复下载（此时不传 URL）。
- `--download-dir '/绝对路径'`：覆盖 Downloads 默认目录。
- `--device auto|cpu|cuda:0`：默认优先 CUDA，无法使用时采用 CPU；不要自动改动系统驱动。
- `--download-only`：只下载，之后用 `--audio` 继续。
- `--limit-seconds 30`：仅用于用户要求的片段或冒烟测试；输出名带 `sample`，报告中明确只转写了片段。通常运行不加此参数。
- `--output-name '文件名'`：自定义输出文件名（不带路径或扩展名）。
- `--overwrite`：仅在用户要求重新生成时覆盖同名转写文件。
- `--auth-file '/绝对路径/认证文件'`：用户已经提供 yutto 认证文件时传入。
- `--threads 8` / `--batch-size 1`：控制 CPU 线程和推理批大小；内存不足时减小批大小或改用 CPU。

支持标准 `/video/BV…` 和 `/video/av…` 链接，也支持直接 BV/av 号。保留 `?p=N` 指定的分 P，其他追踪参数自动移除。默认只处理链接指向的一 P；多 P 请求逐 P 调用，不擅自下载整套合集。短链接先解析为标准视频链接再调用。

## 环境与首次运行

检查 `codex` 环境的 Python、`yutto`、`torch`、`transformers`、`accelerate`、`numpy`，以及 PATH 中的 `ffmpeg`、`ffprobe`。缺少依赖时仅安装到 `codex` 环境：

```bash
conda run --no-capture-output -n codex python -m pip install yutto 'transformers>=4.46,<5' accelerate numpy
```

先确保 PyTorch 与机器硬件匹配。已有可用 CUDA PyTorch 时保留它。本机 NVIDIA 驱动已在沙箱外确认可用，使用官方 CUDA 13.0 版 PyTorch：

```bash
conda run --no-capture-output -n codex python -m pip install 'torch==2.14.1+cu130' --index-url https://download.pytorch.org/whl/cu130
```

若 `nvidia-smi` 或 `torch.cuda.is_available()` 在受限沙箱内失败，先在允许访问 GPU 的执行环境重新检测；不能直接据此判断宿主机驱动损坏或安装 CPU 版。GPU 转写需要允许访问 NVIDIA 设备；在 Codex 中必要时为本次运行申请沙箱外执行权限。真正无 GPU 的机器可先安装 CPU PyTorch，再执行上面的依赖安装命令：

```bash
conda run --no-capture-output -n codex python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

首次运行会从 Hugging Face 下载模型到标准缓存（约 1.6 GB 权重），后续复用缓存；尊重已有 `HF_HOME` / `HF_HUB_CACHE` 设置。CPU 运行耗时可能较长，持续报告实际阶段，不把片段测试宣称为完整转换。

## 完成与故障处理

下载只使用 yutto，显式禁用视频、弹幕、平台字幕、封面和章节信息。字幕来自指定的 Whisper 模型。下载完成后验证音频可解码，再生成带时间戳字幕；检查 SRT 非空、起止时间有序、末尾不超过音频时长，并向用户提供音频与字幕的绝对路径链接。

B 站登录要求、风控或下载失败时，报告真实错误；只有已有有效音频时才能继续。确需认证时请用户提供 yutto 支持的认证文件路径，不扫描浏览器或打印 Cookie。网络失败先确认连接/已有代理设置，仅做有限重试；不自动切换第三方模型镜像或其他模型。转写失败时保留 Downloads 音频，可用 `--audio` 继续。权限受限时申请仅覆盖当前操作的执行权限，不把文件写到其他目录冒充默认输出。

实现依据：[yutto 官方项目](https://github.com/yutto-dev/yutto)、[Whisper 模型说明与时间戳用法](https://huggingface.co/openai/whisper-large-v3-turbo)。验证环境：Python 3.12，yutto 2.3.1，Transformers 4.57.6，PyTorch 2.14.1+cu130。
