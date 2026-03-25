# 实时语音翻译控制台

这是一个实时语音翻译项目，主链路是：

`麦克风输入 -> ASR -> MT -> TTS`

当前项目已经支持三段链路都可切换，并且带有一个可视化前端控制台。

当前已经实际跑通的主链路：

`Qwen ASR -> DeepSeek MT -> Qwen TTS`

## 当前能力

- ASR 支持：
  - 远程 `Qwen ASR Realtime`
  - 本地 `sherpa-onnx zipformer`
- MT 支持：
  - 远程 `DeepSeek`
  - 本地翻译接口
- TTS 支持：
  - 远程 `Qwen TTS VC`
  - 本地 `XTTS`
- 前端支持：
  - 展示当前 ASR / MT / TTS 实际路由
  - 展示当前模型和语言方向
  - 启动 / 停止 [`orchestrator.py`](/C:/Users/30909/Desktop/document/files/orchestrator.py)
  - 查看主管线日志
  - 直接修改 [`.env`](/C:/Users/30909/Desktop/document/files/.env) 中的翻译方向

## 项目结构

```text
files/
- orchestrator.py          # 主实时管线：ASR -> MT -> TTS
- main.py                  # TTS 后端与播放逻辑
- api.py                   # 前端服务 + 本地翻译接口 + orchestrator 控制接口
- translator.py            # 本地翻译模型封装
- create_qwen_voice.py     # 兼容入口，推荐改用 record_voice.py
- record_voice.py          # 录音 + 创建 voice + 管理 voice
- web/
  - index.html             # 前端页面
  - styles.css             # 前端样式
  - app.js                 # 前端交互逻辑
- models/
  - zipformer/             # 本地 ASR 模型
- voice_samples/
  - my_voice.wav           # 默认参考音频
- requirements.txt
- README.md
```

## 当前默认架构

从你现在的代码和配置看，常见运行状态是：

- ASR 主路由：`qwen3-asr-flash-realtime-2025-10-27`
- ASR 回退：`sherpa-onnx zipformer`
- MT 主路由：`deepseek-chat`
- MT 可选本地路由：`/translate`
- TTS 主路由：`qwen3-tts-vc-2026-01-22`
- TTS 回退：`XTTS v2`
- VAD：`Silero VAD`
- 降噪：`RNNoise`

## 运行方式

### 1. 启动前端控制台

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000/
```

前端页面现在可以：

- 查看当前路由和模型
- 启动主管线
- 停止主管线
- 查看主管线日志
- 修改语言方向
- 测试本地翻译接口

### 2. 直接启动主管线

如果你不通过前端，也可以直接跑：

```powershell
python orchestrator.py
```

## 前端和后端的职责

### `orchestrator.py`

负责真正的主业务链路：

- 麦克风采集
- ASR
- MT
- TTS

### `api.py`

负责控制台和辅助接口：

- 提供前端页面
- 提供 `/health`
- 提供 `/api/stack`
- 提供 `/api/pipeline/start`
- 提供 `/api/pipeline/stop`
- 提供 `/api/pipeline/status`
- 提供 `/api/config/languages`
- 提供 `/translate`

注意：

- 前端里的 `/translate` 只用于测试本地翻译接口
- 主实时链路中的翻译仍由 [`orchestrator.py`](/C:/Users/30909/Desktop/document/files/orchestrator.py) 控制

## 环境变量

### ASR

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

本地翻译接口：

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
```

### 本地 TTS 回退

```env
TTS_BACKEND=xtts
VOICE_SAMPLE=voice_samples/my_voice.wav
```

## 运行时切换逻辑

### ASR

通过：

```env
USE_QWEN_ASR_API=true
```

控制：

- `true`：优先走 Qwen ASR
- 初始化失败时回退本地 `zipformer`
- `false`：直接走本地 `zipformer`

### MT

通过：

```env
USE_LOCAL_MT=false
```

控制：

- `false`：走 DeepSeek
- `true`：走本地翻译接口

### TTS

通过：

```env
USE_QWEN_TTS_API=true
```

控制：

- `true`：优先走 Qwen TTS
- 失败时回退 XTTS
- `false`：直接走本地 TTS

## 语言方向设置

当前项目里，语言方向主要由这些变量控制：

- `QWEN_ASR_LANGUAGE`
- `MT_SOURCE_LANG`
- `MT_TARGET_LANG`

### 现在的绑定关系

前端保存语言方向时，会自动把下面两项绑定成一致：

- `QWEN_ASR_LANGUAGE`
- `MT_SOURCE_LANG`

也就是说：

- 如果你在前端把源语言设成 `en`
- 保存后 [`.env`](/C:/Users/30909/Desktop/document/files/.env) 会同时更新：
  - `QWEN_ASR_LANGUAGE=en`
  - `MT_SOURCE_LANG=en`

目标语言则写入：

- `MT_TARGET_LANG`

### 常见语言组合

#### 中文 -> 英文

```env
QWEN_ASR_LANGUAGE=zh
MT_SOURCE_LANG=zh
MT_TARGET_LANG=en
```

#### 英文 -> 中文

```env
QWEN_ASR_LANGUAGE=en
MT_SOURCE_LANG=en
MT_TARGET_LANG=zh
```

#### 中文 -> 中文

```env
QWEN_ASR_LANGUAGE=zh
MT_SOURCE_LANG=zh
MT_TARGET_LANG=zh
```

这个更像“转写 + 同语种播报”。

## 所有主要可调参数

### `orchestrator.py`

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
  当前是 `16000`
- `FRAME_SIZE`
  当前是 `512`
- `VAD_THRESHOLD`
  越高越严格
- `MAX_SILENCE_FRAMES`
  越大越晚断句

### `main.py`

远程 TTS：

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

通用重试：

- `RETRY_COUNT`
- `RETRY_DELAY`

### `api.py`

控制台和本地翻译接口相关：

- `MT_SOURCE_LANG`
- `MT_TARGET_LANG`
- `USE_QWEN_ASR_API`
- `USE_LOCAL_MT`
- `USE_QWEN_TTS_API`
- `QWEN_ASR_MODEL`
- `DEEPSEEK_MODEL`
- `QWEN_TTS_MODEL`
- `QWEN_TTS_VOICE`

## 录音、创建 Voice、切换 Voice

现在推荐统一使用：

[`record_voice.py`](/C:/Users/30909/Desktop/document/files/record_voice.py)

### 1. 一步完成录音 + 创建 voice + 激活

直接运行：

```powershell
python record_voice.py
```

它会自动完成：

- 录一段新的参考音频
- 按顺序保存到 [`voice_samples`](/C:/Users/30909/Desktop/document/files/voice_samples)
  目录，例如：
  - `voice_001.wav`
  - `voice_002.wav`
- 调用 Qwen voice 创建接口
- 把结果记录到：
  [`voice_samples/voice_registry.json`](/C:/Users/30909/Desktop/document/files/voice_samples/voice_registry.json)
- 自动更新 [`.env`](/C:/Users/30909/Desktop/document/files/.env) 中的：
  - `QWEN_TTS_VOICE`
  - `QWEN_TTS_VOICE_SAMPLE`
  - `VOICE_SAMPLE`

### 2. 查看已有 voice

```powershell
python record_voice.py --list
```

### 3. 手动切换当前使用的 voice

```powershell
python record_voice.py --activate 2
```

这会把第 2 个已保存 voice 激活，并同步更新 [`.env`](/C:/Users/30909/Desktop/document/files/.env)。

## 重要日志怎么看

启动时建议重点看这些：

- `[MT model ] provider=deepseek | model=deepseek-chat`
- 或 `[MT model ] provider=local_api | url=...`
- `[ASR model] provider=qwen_api | model=qwen3-asr-flash-realtime-2025-10-27 | language=...`
- 或 `[ASR model] provider=sherpa-onnx | model_path=...`
- `[ASR route] primary=qwen_api | fallback=zipformer`
- 或 `[ASR route] primary=zipformer`
- `[TTS] INFO    TTS primary model | provider=qwen_api | model=qwen3-tts-vc-2026-01-22 | voice_configured=True`
- `[TTS] INFO    TTS fallback model | provider=xtts | model=xtts_v2 | device=...`
- `[TTS] INFO    TTS startup | primary=qwen_api | fallback=xtts`

运行中建议看：

- `[ASR final  ]: ...`
- `[MT  ]: ...`
- `[TTS] INFO    TTS provider | provider=qwen_api | model=qwen3-tts-vc-2026-01-22`

如果 Qwen TTS 回退，会看到：

```text
Qwen TTS API failed, falling back to local backend.
```

## 常见问题

### 1. 前端能启动主管线吗？

可以。  
前端现在通过：

- `/api/pipeline/start`
- `/api/pipeline/stop`
- `/api/pipeline/status`

来控制 [`orchestrator.py`](/C:/Users/30909/Desktop/document/files/orchestrator.py)。

### 2. 为什么前端里 `/translate` 和主链路不一样？

因为：

- `/translate` 是本地翻译接口测试台
- 主链路翻译逻辑仍然在 [`orchestrator.py`](/C:/Users/30909/Desktop/document/files/orchestrator.py) 里

### 3. 为什么改了源语言但识别还有时不稳定？

因为 `QWEN_ASR_LANGUAGE` 更像偏好提示，不一定是绝对锁定。  
所以如果设置成英文识别，但实际说中文，ASR 仍然可能识别出中文。

## 故障排查

### Qwen ASR 没有生效

检查：

- `USE_QWEN_ASR_API=true`
- `DASHSCOPE_API_KEY` 是否存在
- `QWEN_ASR_MODEL` 是否正确
- 网络是否能访问 DashScope

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
- `QWEN_TTS_MODEL` 是否与 voice 创建时一致

## 安全说明

不要提交这些内容：

- `.env`
- `models/`
- `voice_samples/`
- `.venv310/`

另外请确认你对参考音频拥有合法使用权限。
