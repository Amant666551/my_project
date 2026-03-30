# 实时语音翻译控制台

这是一个实时语音翻译项目，当前主链路为：

`麦克风输入 -> ASR -> MT -> TTS`

当前已经接通并可运行的常见组合是：

`Qwen ASR -> DeepSeek MT -> Qwen TTS`

项目同时保留了本地回退路径：

- ASR：本地 `sherpa-onnx zipformer`
- MT：本地 `/translate` 接口
- TTS：本地 `XTTS`

项目带有一个前端控制台，可用于查看当前链路、修改语言方向、切换 voice，以及启动/停止主管线。

## 当前代码状态

当前版本以 [`orchestrator.py`](/C:/Users/30909/Desktop/document/files/orchestrator.py) 为主控制入口，已经做了以下 ASR 前处理增强：

- 本地高通滤波
- 本地 `RNNoise` 降噪
- 本地 `Silero VAD`
- 平滑后的双阈值 VAD 门控
- pre-roll 缓冲，减少吞首字
- 有界音频队列，避免极端情况下无限堆积

当前主管线仍然是单主循环结构：

1. 持续采集音频
2. 把音频送入 ASR
3. ASR 出句后同步调用 MT
4. MT 成功后将文本送入 TTS 队列

这意味着：

- TTS 已经通过线程队列做了串行播放
- 但 ASR 与 MT 目前仍然串行耦合
- 在 MT 或其他阶段变慢时，可能看到音频队列积压或 `input overflow`

这属于当前版本的已知结构限制，不影响基本功能可用，但会影响高负载下的实时稳定性。

## 项目结构

```text
files/
- orchestrator.py          # 主管线：采音 + ASR + MT 调度 + TTS 投递
- main.py                  # TTS 后端与播放逻辑
- api.py                   # 前端服务、控制接口、本地翻译接口
- translator.py            # 本地翻译模型封装
- record_voice.py          # 录音、创建 voice、激活 voice
- web/
  - index.html             # 前端页面
  - app.js                 # 前端交互逻辑
  - styles.css             # 前端样式
- clients/
  - asr_client.py
  - mt_client.py
  - tts_client.py
- models/
  - zipformer/             # 本地 ASR 模型
  - translate/             # 本地 MT 模型资源
- voice_samples/
  - voice_registry.json    # voice 注册表
  - *.wav                  # voice 样本
- requirements.txt
- .env
- README.md
```

## 当前运行架构

### ASR

- 主路由：`Qwen ASR Realtime`
- 回退：`sherpa-onnx zipformer`
- 前处理：
  - 高通滤波
  - `RNNoise`
  - `Silero VAD`
  - 起说/停说双阈值门控

### MT

- 主路由：`DeepSeek`
- 可切换到本地翻译接口：`/translate`

### TTS

- 主路由：`Qwen TTS VC`
- 回退：`XTTS`

## 运行方式

### 1. 启动前端控制台

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000/
```

前端可用于：

- 查看当前 ASR / MT / TTS 路由
- 查看当前模型和语言方向
- 启动主管线
- 停止主管线
- 查看主管线日志
- 修改语言方向
- 测试本地翻译接口
- 切换当前 voice

### 2. 直接启动主管线

```powershell
python orchestrator.py
```

## 各主要文件职责

### `orchestrator.py`

负责实时主链路：

- 麦克风输入
- 本地前处理
- ASR
- MT
- 投递到 TTS 队列

当前 TTS 已经单独走工作线程，避免一条语音播放时打断另一条语音播放。

### `main.py`

负责 TTS：

- 优先尝试 `Qwen TTS API`
- 失败时回退到本地后端
- 当前本地回退默认是 `XTTS`

### `api.py`

负责控制台和辅助接口：

- `/health`
- `/api/stack`
- `/api/pipeline/start`
- `/api/pipeline/stop`
- `/api/pipeline/status`
- `/api/config/languages`
- `/api/voices`
- `/api/voices/activate`
- `/translate`

注意：

- `/translate` 只用于本地翻译接口测试或本地 MT 路由
- 主管线中的翻译调度仍由 [`orchestrator.py`](/C:/Users/30909/Desktop/document/files/orchestrator.py) 控制

### `record_voice.py`

负责 voice 管理：

- 录音
- 调用 Qwen voice 创建接口
- 将 voice 写入 `voice_registry.json`
- 激活指定 voice 并回写 `.env`

## 当前关键环境变量

### ASR

```env
USE_QWEN_ASR_API=true
QWEN_ASR_MODEL=qwen3-asr-flash-realtime-2025-10-27
QWEN_ASR_URL=wss://dashscope.aliyuncs.com/api-ws/v1/realtime
QWEN_ASR_LANGUAGE=zh
```

### MT

远程 MT：

```env
DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
USE_LOCAL_MT=false
```

本地 MT：

```env
USE_LOCAL_MT=true
MT_URL=http://127.0.0.1:8000/translate
MT_SOURCE_LANG=zh
MT_TARGET_LANG=en
MT_TIMEOUT_SEC=8
```

### TTS

```env
DASHSCOPE_API_KEY=your_key
USE_QWEN_TTS_API=true
QWEN_TTS_MODEL=qwen3-tts-vc-2026-01-22
QWEN_TTS_VOICE=your_voice_id
QWEN_TTS_BASE_HTTP_API_URL=https://dashscope.aliyuncs.com/api/v1
VOICE_SAMPLE=voice_samples/voice_001.wav
```

### 当前 ASR 采音相关默认值

这些值现在由 [`orchestrator.py`](/C:/Users/30909/Desktop/document/files/orchestrator.py) 控制：

```text
SAMPLE_RATE=16000
FRAME_SIZE=512
CHANNELS=1
ASR_INPUT_LATENCY=high
ASR_AUDIO_QUEUE_MAX_CHUNKS=64
```

### 当前 ASR 前处理门控参数

```text
RNNOISE_FRAME_SIZE=160
HPF_ALPHA=0.97
VAD_START_THRESHOLD=0.55
VAD_END_THRESHOLD=0.35
MIN_SPEECH_START_FRAMES=2
MIN_SPEECH_FRAMES=6
ENERGY_THRESHOLD=0.008
ENERGY_RELEASE_RATIO=0.65
PRE_SPEECH_FRAMES=8
VAD_SMOOTHING_FRAMES=5
MAX_SILENCE_FRAMES=30
```

这些参数是当前针对“减少误触发，同时尽量避免吞首字”调过的一组默认值。

## 语言方向设置

当前语言方向主要由以下变量控制：

- `QWEN_ASR_LANGUAGE`
- `MT_SOURCE_LANG`
- `MT_TARGET_LANG`

前端保存语言方向时会同时更新：

- `QWEN_ASR_LANGUAGE`
- `MT_SOURCE_LANG`

目标语言写入：

- `MT_TARGET_LANG`

常见组合：

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

## Voice 管理

推荐统一使用：

[`record_voice.py`](/C:/Users/30909/Desktop/document/files/record_voice.py)

### 录音并创建新的 voice

```powershell
python record_voice.py
```

### 查看已有 voice

```powershell
python record_voice.py --list
```

### 激活指定 voice

```powershell
python record_voice.py --activate 2
```

激活后会同步更新 `.env` 中的：

- `QWEN_TTS_VOICE`
- `QWEN_TTS_VOICE_SAMPLE`
- `VOICE_SAMPLE`

## 启动后建议关注的日志

启动时：

- `[Pipeline config] ...`
- `[MT model ] provider=deepseek | model=deepseek-chat`
- 或 `[MT model ] provider=local_api | url=...`
- `[ASR model] provider=qwen_api | model=... | language=...`
- 或 `[ASR model] provider=sherpa-onnx | model_path=...`
- `[ASR route] primary=qwen_api | fallback=zipformer`
- 或 `[ASR route] primary=zipformer`
- `[TTS] INFO    TTS primary model | provider=qwen_api | model=...`
- `[TTS] INFO    TTS fallback model | provider=xtts | model=xtts_v2 | device=...`

运行中：

- `[ASR final  ]: ...`
- `[MT  ]: ...`
- `[TTS] INFO    TTS provider | provider=qwen_api | model=...`
- `[Audio] input overflow | ...`
- `[Audio] capture queue full, dropping oldest buffered chunk ...`

其中最后两类日志表示当前实时链路存在采音积压。

## 当前已知问题

### 1. 主管线仍然是串行主循环

当前版本下，ASR 出句后 MT 仍在主管线中同步执行，因此在网络抖动、翻译变慢或后续处理变慢时，可能造成：

- `_audio_q` 积压
- 音频块丢弃
- `input overflow`

### 2. Qwen TTS 可能偶发超时

日志中如果出现：

```text
Qwen TTS synthesis failed: ... Read timed out
```

则会自动回退到本地 TTS。

### 3. 当前 README 以现有代码为准

如果后续我们继续把主管线拆成真正的异步流水线，README 也需要同步更新。

## 安全说明

不要提交以下内容：

- `.env`
- `models/`
- `voice_samples/`
- `.venv310/`

另外请确保你对参考音频拥有合法使用权限。
