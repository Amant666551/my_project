# ASR -> MT -> TTS Pipeline

一个本地语音链路项目，整体流程是：

`麦克风输入 -> ASR 识别 -> MT 翻译 -> TTS 播报`

当前版本已经接通了：
- 本地 ASR
- 远程 MT
- 可切换的本地 / 远程 TTS

适合继续做实时语音翻译、语音助手、语音交互类实验。

## Current Stack

当前代码默认和可选链路如下：

- ASR: `sherpa-onnx` + `zipformer`
- VAD: `snakers4/silero-vad`
- Denoise: `RNNoise`
- MT: `DeepSeek API`
- TTS primary: `Qwen TTS VC API`
- TTS fallback: `XTTS v2`

也就是说，当前 TTS 的实际策略是：

- 如果 `.env` 里 `USE_QWEN_TTS_API=true`，优先使用 Qwen TTS
- 如果 Qwen 失败，则自动回退到本地 XTTS

## Project Structure

```text
files/
├─ orchestrator.py         # 主入口：ASR -> MT -> TTS
├─ main.py                 # TTS 模块：Qwen TTS / XTTS / OpenVoice / edge-tts
├─ create_qwen_voice.py    # 创建 Qwen voice 的脚本
├─ api.py                  # 历史保留的本地翻译 API
├─ translator.py           # 历史保留的本地翻译封装
├─ record_voice.py         # 录制参考音频
├─ requirements.txt
├─ README.md
├─ models/
│  └─ zipformer/           # 本地 ASR 模型
└─ voice_samples/
   └─ my_voice.wav         # 默认参考音频
```

## How It Works

### 1. ASR

[`orchestrator.py`](/C:/Users/30909/Desktop/document/files/orchestrator.py) 中的 `StreamingASR` 负责实时识别。

当前实现：
- `sherpa-onnx` 的 `zipformer` 负责识别
- `RNNoise` 做降噪
- `Silero VAD` 判断静音和断句

静音持续一段时间后，会把当前识别结果视为一句完整语音。

### 2. MT

当前翻译直接在 [`orchestrator.py`](/C:/Users/30909/Desktop/document/files/orchestrator.py) 中调用 `DeepSeek API`。

默认方向：
- `zh -> en`

相关环境变量：
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`

### 3. TTS

[`orchestrator.py`](/C:/Users/30909/Desktop/document/files/orchestrator.py) 会把翻译结果送进串行 TTS 队列，避免后一条播报打断前一条。

真正的语音合成在 [`main.py`](/C:/Users/30909/Desktop/document/files/main.py) 中完成。

当前 TTS 路由：
- 主路由：`Qwen TTS VC API`
- 回退路由：`XTTS v2`

如果启用了 Qwen：
- `main.py` 会优先走远程 Qwen TTS
- 如果远程失败，会自动回退到本地 XTTS

## Requirements

建议环境：
- Python `3.10`

安装依赖：

```powershell
pip install -r requirements.txt
```

## Environment Variables

项目主要依赖 [`.env`](/C:/Users/30909/Desktop/document/files/.env) 配置。

### DeepSeek MT

```env
DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

### Qwen TTS

```env
DASHSCOPE_API_KEY=your_key
USE_QWEN_TTS_API=true
QWEN_TTS_MODEL=qwen3-tts-vc-2026-01-22
QWEN_TTS_VOICE=your_voice_id
QWEN_TTS_BASE_HTTP_API_URL=https://dashscope.aliyuncs.com/api/v1
```

### XTTS Fallback

[`main.py`](/C:/Users/30909/Desktop/document/files/main.py) 中默认配置：

```python
TTS_BACKEND = "xtts"
VOICE_SAMPLE = "voice_samples/my_voice.wav"
```

这意味着：
- Qwen 是主路由
- XTTS 是本地兜底

## Create Qwen Voice

如果要使用 Qwen TTS VC，需要先创建 voice。

项目里已经提供脚本：

[`create_qwen_voice.py`](/C:/Users/30909/Desktop/document/files/create_qwen_voice.py)

运行：

```powershell
python create_qwen_voice.py
```

默认会读取：
- `DASHSCOPE_API_KEY`
- `QWEN_TTS_MODEL`
- `voice_samples/my_voice.wav`

成功后会打印：

```text
QWEN_TTS_VOICE=qwen-tts-vc-...
```

把这行填回 [`.env`](/C:/Users/30909/Desktop/document/files/.env) 即可。

## Run

启动主流程：

```powershell
python orchestrator.py
```

启动后你会看到：

```text
ASR -> MT -> TTS pipeline starting...
System ready - start speaking!
```

然后直接对麦克风说话即可。

## Important Logs

当前日志里最值得关注的是这些：

### 启动时

- `[MT model ] provider=deepseek | model=...`
- `[ASR model] provider=sherpa-onnx | model_path=...`
- `[VAD model] repo=snakers4/silero-vad | model=silero_vad ...`
- `[TTS] INFO    TTS primary model | provider=qwen_api | model=...`
- `[TTS] INFO    TTS fallback model | provider=xtts | model=xtts_v2 | device=...`
- `[TTS] INFO    TTS startup | primary=qwen_api | fallback=xtts`

### 每句播报时

- `[ASR final  ]: ...`
- `[MT  ]: ...`
- `[TTS] INFO    TTS provider | provider=qwen_api | model=...`

如果 Qwen 失败回退，你会看到：

```text
Qwen TTS API failed, falling back to local backend.
TTS provider | provider=xtts | mode=...
```

## Record Voice Sample

如果你想替换默认参考音频，可以运行：

```powershell
python record_voice.py
```

默认输出：

```text
voice_samples/my_voice.wav
```

建议参考音频：
- 10 到 20 秒
- 单人说话
- 安静环境
- 无背景音乐和混响
- 发音自然

## Notes

- 当前 TTS 队列是串行播放，避免互相打断
- Qwen TTS 是远程 API，XTTS 是本地回退
- 当前 MT 是远程 DeepSeek，不再走本地 `api.py`
- `api.py` 和 `translator.py` 目前属于历史保留模块，不是主流程核心

## Troubleshooting

### 1. Qwen TTS 没有生效

检查：
- `USE_QWEN_TTS_API=true`
- `DASHSCOPE_API_KEY` 是否存在
- `QWEN_TTS_VOICE` 是否存在
- `QWEN_TTS_MODEL` 是否和创建 voice 时的 `target_model` 一致

### 2. Qwen TTS 超时

常见原因：
- 网络不稳定
- 挂代理导致请求异常

建议：
- 尽量直连
- 不要挂梯子

### 3. TTS 回退到 XTTS

如果日志里出现：

```text
Qwen TTS API failed, falling back to local backend.
```

说明远程 Qwen 没成功，本地 XTTS 开始接管。

### 4. ASR 效果一般

当前 ASR 是本地轻量方案，后续可以考虑升级到更强的流式中文 ASR。

## License / Notes

请自行确认：
- 参考音频的使用权限
- API Key 的安全保管
- 上传到 GitHub 时不要提交 `.env`、`models/`、`voice_samples/`
