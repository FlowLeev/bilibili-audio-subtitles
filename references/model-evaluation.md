# Qwen 与 Whisper 的模型选择依据

2026-10-06 在同一 RTX 4080 SUPER、conda `codex` 环境上依次测试，复用已经下载的相同音频，模型权重缓存后计时。Qwen 使用官方 qwen-asr 0.0.6 的 Transformers 后端、bfloat16、SDPA、batch size 1、max_new_tokens 4096；Whisper 使用 float16、原生长音频、return_timestamps=True、condition_on_prev_tokens=False。未提供术语提示或人工纠正文本。

## 完整中文视频

样例：[BV1w88n6uE1X](https://www.bilibili.com/video/BV1w88n6uE1X/)，758.50 秒。

| 指标 | Whisper large-v3-turbo | Qwen3-ASR-1.7B + ForcedAligner-0.6B |
| --- | --- | --- |
| 推理时间（Qwen 包含对齐） | 15.49 秒 | 27.18 秒 |
| 解码、模型加载、转写与分段耗时 | 18.70 秒 | 32.35 秒 |
| SRT 段数 | 438 | 282 |
| 最长字幕 | 5.18 秒 | 5.76 秒 |
| 结构检查 / 明显质量异常 | 通过 / 无 | 通过 / 无 |
| “订单流”识别次数 | 1 | 4 |
| “订单炉”错误次数 | 2 | 0 |
| “鼠标”识别次数 | 0 | 2 |
| “主标”错误次数 | 2 | 0 |

Qwen 两次完整测试的推理加对齐时间为 27.18–27.77 秒；单次结果和段数有轻微差异，不能将固定段数当作回归条件。Qwen 的推理峰值 PyTorch 显存约 7.16 GiB，不是整张卡的所有进程显存占用。首次模型下载不包含在上述耗时内。

重点窗口 `43.24–346.06`：Qwen 119 段、最长 5.76 秒；Whisper 167 段、最长 5.18 秒。`490–513`：Qwen 8 段、最长 4.40 秒；Whisper 16 段、最长 2.24 秒。均未复现旧版五分钟字幕或连续 23 条“好”。分段方式不同，段数更多不等于识别更准。

Qwen 仍然有错误：约 315 秒输出“图层级的”，Whisper 输出“如此码机的”；约 497 秒 Qwen 输出“他单单杆子打不着”，Whisper 输出“扒竿子打不着”。Qwen 在堵车例子中也仍有漏字。无质量告警不能证明这些文字正确。完整视频没有逐句人工参考稿，本次未计算 CER/WER，不能宣称全篇准确率或所有音频都优于 Whisper。

默认选择 Qwen 的依据是用户已指出的中文术语有改善，字幕分段稳定，且本机完整转换仍约半分钟。Whisper 保留为速度优先、模型对比和 Qwen 对齐不支持的语言的显式选项。

## 英语短片

[jNQXAC9IVRw](https://youtu.be/jNQXAC9IVRw)，19.008 秒。Qwen 实测推理加对齐 0.51–1.08 秒，独立加载运行约 6.13 秒，7 段 SRT、最长 2.16 秒。它识别出了 `trunks`，旧 Whisper 输出 `fronts`；两者仍将开头有关站在大象前面的短语识别得不准确。一个英语短片不足以判断英语的总体识别优势。

Qwen 对齐中个别字/词的起止值相同：完整中文测试 162/3667 个单位，英语测试 2/37 个单位。SRT 分组保留这些文字，使用相邻组的已有真实边界；不生成等分词时间、不丢掉词、不把词级时间戳称为人工验证。整个字幕组没有有效时长或对齐文字不匹配时，保存原始结果并失败。

## 能力和复现

[官方项目](https://github.com/QwenLM/Qwen3-ASR)说明：时间戳需要独立 ForcedAligner，调用参数是 `return_time_stamps=True`。对齐支持 11 种语言；ASR 能识别更多语言，但这些语言不一定能输出对齐时间戳。官方 SDK 的带对齐长音频路径使用最长 180 秒的低能量边界分块并恢复全局偏移；不能把 Whisper 的实验性 pipeline 分块经验直接套用于 Qwen。

本机评估原始结果、诊断和 SRT 保存在 `/home/leev/codex_workspace/skill开发/qwen3-evaluation/`。完整对比诊断为 `diagnostics/qwen-integrated-20261006T052517Z-fcb88374.json` 和 `diagnostics/whisper-fresh-20261006T052251Z-d1bb1269.json`。这些音频和大体积运行记录不作为 skill 的安装资源提交；用户重新评估时用同一音频运行两种 engine，再调用 `scripts/check_regression.py`。
