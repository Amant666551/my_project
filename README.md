# ASR -> MT -> TTS Pipeline

一个实时语音链路项目，整体流程是：

`麦克风输入 -> ASR 识别 -> MT 翻译 -> TTS 播报`

当前版本支持三段链路都可切换：

- ASR：本地 `zipformer` / 远程 `Qwen ASR`
- MT：本地翻译服务 / 远程 `DeepSeek`
- TTS：远程 `Qwen TTS VC` / 本地 `XTTS` 回退

当前已经实际跑通的主链是：

`Qwen ASR -> DeepSeek MT -> Qwen TTS`

## Current Stack

- ASR primary: `qwen3-asr-flash-realtime-2025-10-27`
- ASR fallback: `sherpa-onnx zipformer`
- MT primary: `DeepSeek API`
- MT optional local: `http://127.0.0.1:8000/translate`
- TTS primary: `qwen3-tts-vc-2026-01-22`
- TTS fallback: `XTTS v2`
- VAD: `Silero VAD`
- Denoise: `RNNoise`

## Project Structure

```text
files/
├─ orchestrator.py         # 主入口：ASR -> MT -> TTS
├─ main.py                 # TTS 模块：Qwen TTS / XTTS / OpenVoice / edge-tts
├─ create_qwen_voice.py    # 创建 Qwen TTS voice
├─ record_voice.py         # 录制参考音频
├─ api.py                  # 本地翻译 API
├─ translator.py           # 本地翻译模型封装
├─ requirements.txt
├─ README.md
├─ models/
│  └─ zipformer/           # 本地 ASR 模型
└─ voice_samples/
   └─ my_voice.wav         # 默认参考音频
```

## How It Works

### ASR

[`orchestrator.py`](/C:/Users/30909/Desktop/document/files/orchestrator.py) 里有两套 ASR：

- 本地 ASR：`sherpa-onnx` + `zipformer`
- 远程 ASR：`Qwen ASR Realtime`

开关：

```env
USE_QWEN_ASR_API=true
```

行为：

- `true`：优先走 Qwen ASR，初始化失败时回退本地 `zipformer`
- `false`：直接走本地 `zipformer`

本地 ASR 会额外配合：

- `RNNoise` 降噪
- `Silero VAD` 断句

### MT

[`orchestrator.py`](/C:/Users/30909/Desktop/document/files/orchestrator.py) 里有两套翻译路径：

- 远程 MT：`DeepSeek API`
- 本地 MT：[`api.py`](/C:/Users/30909/Desktop/document/files/api.py) 提供的 `/translate`

开关：

```env
USE_LOCAL_MT=false
```

行为：

- `false`：走 DeepSeek
- `true`：走本地翻译服务

如果切到本地翻译服务，需要先启动：

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

### TTS

[`main.py`](/C:/Users/30909/Desktop/document/files/main.py) 负责语音合成。

当前 TTS 路由：

- 主路由：`Qwen TTS VC API`
- 回退路由：`XTTS v2`

开关：

```env
USE_QWEN_TTS_API=true
```

行为：

- `true`：优先走 Qwen TTS
- 失败时自动回退本地 XTTS
- `false`：直接走本地后端

## Environment Variables

项目主要依赖 [`.env`](/C:/Users/30909/Desktop/document/files/.env)。

### Qwen ASR

```env
USE_QWEN_ASR_API=true
QWEN_ASR_MODEL=qwen3-asr-flash-realtime-2025-10-27
QWEN_ASR_URL=wss://dashscope.aliyuncs.com/api-ws/v1/realtime
QWEN_ASR_LANGUAGE=zh
```

### MT

远程 DeepSeek：

```env
DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
USE_LOCAL_MT=false
```

本地翻译服务：

```env
USE_LOCAL_MT=true
MT_URL=http://127.0.0.1:8000/translate
MT_SOURCE_LANG=zh
MT_TARGET_LANG=en
MT_TIMEOUT_SEC=8
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

[`main.py`](/C:/Users/30909/Desktop/document/files/main.py) 默认使用：

```python
TTS_BACKEND = "xtts"
VOICE_SAMPLE = "voice_samples/my_voice.wav"
```

## All Tunable Parameters

下面这些是当前代码里最主要、最值得调的参数。

### orchestrator.py

ASR / 音频输入：

- `USE_QWEN_ASR_API`
- `QWEN_ASR_MODEL`
- `QWEN_ASR_URL`
- `QWEN_ASR_LANGUAGE`
- `SAMPLE_RATE`
- `FRAME_SIZE`
- `CHANNELS`
- `VAD_THRESHOLD`
- `MAX_SILENCE_FRAMES`

MT：

- `USE_LOCAL_MT`
- `MT_URL`
- `MT_SOURCE_LANG`
- `MT_TARGET_LANG`
- `MT_TIMEOUT_SEC`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`

说明：

- `SAMPLE_RATE`
  当前是 `16000`，本地 ASR 和 Qwen ASR 都按这个采样率工作。
- `FRAME_SIZE`
  当前是 `512`，控制每次处理的音频块大小。
- `VAD_THRESHOLD`
  越高越严格，越低越容易判定为“有人声”。
- `MAX_SILENCE_FRAMES`
  越大越晚结束一句话，越小越容易提前断句。

### main.py

TTS 总开关：

- `USE_QWEN_TTS_API`
- `QWEN_TTS_MODEL`
- `QWEN_TTS_VOICE`
- `QWEN_TTS_BASE_HTTP_API_URL`
- `DASHSCOPE_API_KEY`

本地 TTS：

- `TTS_BACKEND`
- `VOICE_SAMPLE`
- `PROXY`

XTTS 参数：

- `XTTS_TEMPERATURE`
- `XTTS_SPEED`
- `XTTS_REPETITION_PENALTY`
- `XTTS_LENGTH_PENALTY`
- `XTTS_TOP_K`
- `XTTS_TOP_P`
- `XTTS_STREAM_CHUNK_SIZE`
- `XTTS_STREAMING_MODE`
- `XTTS_STREAM_PREROLL_CHUNKS`

通用：

- `RETRY_COUNT`
- `RETRY_DELAY`

说明：

- `TTS_BACKEND`
  本地后端选择，当前默认是 `xtts`。
- `VOICE_SAMPLE`
  本地 XTTS/OpenVoice 使用的参考音频。
- `XTTS_SPEED`
  控制语速，`1.0` 是标准语速。
- `XTTS_TEMPERATURE`
  越高越自由，越低越稳定。
- `XTTS_STREAMING_MODE`
  可选：`auto` / `on` / `off`

## How To Change Language Direction

### 1. 只改翻译方向

如果你只想改“识别后的文字翻译成什么语言”，改 [`.env`](/C:/Users/30909/Desktop/document/files/.env) 里的：

```env
MT_SOURCE_LANG=zh
MT_TARGET_LANG=en
```

例如：

- 中文 -> 英文

```env
MT_SOURCE_LANG=zh
MT_TARGET_LANG=en
```

- 英文 -> 中文

```env
MT_SOURCE_LANG=en
MT_TARGET_LANG=zh
```

- 日文 -> 英文

```env
MT_SOURCE_LANG=ja
MT_TARGET_LANG=en
```

### 2. 改 ASR 识别语言

如果使用 Qwen ASR，改：

```env
QWEN_ASR_LANGUAGE=zh
```

例如：

- 中文识别：`zh`
- 英文识别：`en`
- 日文识别：`ja`

如果使用本地 `zipformer`，当前模型目录是中文场景优先的 [`models/zipformer`](/C:/Users/30909/Desktop/document/files/models/zipformer)，要换语言通常需要换模型本身。

### 3. 改 TTS 播报语言

当前 TTS 播报语言主要跟 `MT_TARGET_LANG` 走：

- 在 [`orchestrator.py`](/C:/Users/30909/Desktop/document/files/orchestrator.py) 里，翻译完成后会把 `MT_TARGET_LANG` 传给 `tts_speak(...)`

所以一般来说：

- `MT_TARGET_LANG=en` -> 英文播报
- `MT_TARGET_LANG=zh` -> 中文播报

## Recommended Language Combinations

### 中文 -> 英文

```env
QWEN_ASR_LANGUAGE=zh
MT_SOURCE_LANG=zh
MT_TARGET_LANG=en
```

### 英文 -> 中文

```env
QWEN_ASR_LANGUAGE=en
MT_SOURCE_LANG=en
MT_TARGET_LANG=zh
```

### 中文 -> 中文

```env
QWEN_ASR_LANGUAGE=zh
MT_SOURCE_LANG=zh
MT_TARGET_LANG=zh
```

这个更像“转写 + 同语言播报”。

## Create Qwen TTS Voice

如果要使用 Qwen TTS VC，需要先创建 `voice`。

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

把它填回 [`.env`](/C:/Users/30909/Desktop/document/files/.env) 即可。

## Run

启动主流程：

```powershell
python orchestrator.py
```

如果使用本地 MT，还需要先启动：

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

启动成功后通常会看到：

```text
ASR -> MT -> TTS pipeline starting...
System ready - start speaking!
```

## Important Logs

启动时最关键的日志有：

- `[MT model ] provider=deepseek | model=deepseek-chat`
- 或 `[MT model ] provider=local_api | url=...`
- `[ASR model] provider=qwen_api | model=qwen3-asr-flash-realtime-2025-10-27 | language=zh`
- 或 `[ASR model] provider=sherpa-onnx | model_path=...`
- `[ASR route] primary=qwen_api | fallback=zipformer`
- 或 `[ASR route] primary=zipformer`
- `[TTS] INFO    TTS primary model | provider=qwen_api | model=qwen3-tts-vc-2026-01-22 | voice_configured=True`
- `[TTS] INFO    TTS fallback model | provider=xtts | model=xtts_v2 | device=cpu`
- `[TTS] INFO    TTS startup | primary=qwen_api | fallback=xtts`

每句播报时最关键的是：

- `[ASR final  ]: ...`
- `[MT  ]: ...`
- `[TTS] INFO    TTS provider | provider=qwen_api | model=qwen3-tts-vc-2026-01-22`

如果 TTS 回退，会看到：

```text
Qwen TTS API failed, falling back to local backend.
```

## Troubleshooting

### Qwen ASR 没有生效

检查：

- `USE_QWEN_ASR_API=true`
- `DASHSCOPE_API_KEY` 是否存在
- `QWEN_ASR_MODEL` 是否正确
- 网络是否可直连 DashScope

### 本地 MT 没有生效

检查：

- `USE_LOCAL_MT=true`
- [`api.py`](/C:/Users/30909/Desktop/document/files/api.py) 是否已经启动
- `MT_URL` 是否正确

### Qwen TTS 没有生效

检查：

- `USE_QWEN_TTS_API=true`
- `DASHSCOPE_API_KEY` 是否存在
- `QWEN_TTS_VOICE` 是否存在
- `QWEN_TTS_MODEL` 是否和创建 voice 时的 `target_model` 一致

### 上传 GitHub 时

不要提交这些内容：

- `.env`
- `models/`
- `voice_samples/`
- `.venv310/`

## Security Note

请自行确认：

- 参考音频的使用权限
- API Key 的安全保管
- 远程模型服务的计费和使用限制
