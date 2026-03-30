# 实时语音翻译控制台

这是一个实时语音翻译项目，当前主链路为：

`麦克风输入 -> ASR -> MT -> TTS`

目前常见可运行组合是：

`Qwen ASR -> DeepSeek MT -> Qwen TTS`

同时保留了本地回退路径：

- ASR：`sherpa-onnx zipformer`
- MT：本地 `/translate` 接口
- TTS：`XTTS`

项目带有前端控制台，可查看当前链路、修改语言方向、切换 voice，并启动/停止主管线。

## 当前版本状态

当前版本以 [`orchestrator.py`](/C:/Users/30909/Desktop/document/files/orchestrator.py) 为主管线入口，已经完成了三类关键工作：

### 1. ASR 前处理增强

- 高通滤波
- `RNNoise` 降噪
- `Silero VAD`
- 平滑后的双阈值门控
- pre-roll 缓冲，减少吞首字
- 有界音频队列，避免极端情况下无限堆积

### 2. 串行主管线拆分为并行流水线

当前主管线已拆成独立工作线程：

- 音频输入回调：持续向 `_audio_q` 投递音频
- `ASR worker`：持续消费音频并产出句子
- `MT worker`：持续消费 ASR 文本并翻译
- `TTS worker`：持续消费翻译结果并播报

这意味着：

- MT 或 TTS 变慢时，不会像旧版本那样直接卡死前面的 ASR 消费
- TTS 已经与采音/识别链路解耦
- 系统更接近真正的流式流水线

### 3. AEC-ready 架构骨架已接入

为了后续研究真正的全双工回声消除，本版本已经把 AEC 所需的接线骨架提前接进代码：

- [`aec.py`](/C:/Users/30909/Desktop/document/files/aec.py)
- [`playback_bus.py`](/C:/Users/30909/Desktop/document/files/playback_bus.py)
- [`audio_player.py`](/C:/Users/30909/Desktop/document/files/audio_player.py)

当前状态下：

- TTS 播放优先走受控播放链路
- 播放中的音频会被写入 `playback_bus`
- 麦克风输入会先经过 `aec.py`，再进入当前前处理链路

注意：

- 这一版已经在 [`aec.py`](/C:/Users/30909/Desktop/document/files/aec.py) 中接入了真实 AEC 后端实验分支，当前实验后端是 `pyaec`
- 经当前机器实测，`pyaec` 在外放双讲场景下会误伤近端人声，因此当前推荐默认保持 `ENABLE_AEC=false`
- 这一步的目标是在不改 ASR / MT / TTS 模型选型的前提下，把外放全双工所需的 AEC 接口与播放参考链路先接通并验证

## 已验证结论

### 1. 并行流水线是有效的

在连续说多句、上一句还在播报时继续说下一句的场景下：

- ASR 仍能继续出句
- MT 仍能继续翻译
- TTS 可继续按队列播放

说明当前并行改造方向是正确的。

### 2. 耳机测试基本确认了“回音串音”问题

在外放场景下，边播边说时容易出现明显识别污染；戴耳机后，识别质量明显改善。

这说明：

- 当前系统已经具备并行能力
- 但尚未具备真正稳定的“外放全双工”能力
- 外放场景下的主要问题之一是：TTS 播报回灌到麦克风，污染 ASR

也就是说，后续若要实现真正的全双工外放，需要接入真实的 `AEC` 后端。

### 3. `input overflow` 仍然存在

即使在耳机测试下，日志中仍可能持续看到：

```text
[Audio] input overflow | overflow_count=...
```

这说明当前 ASR 前处理链路仍有实时压力。它不一定立刻导致系统不可用，但会影响稳定性，并可能在高负载或连续说话时带来：

- 句子截断
- 短词误识别
- 结果偶发扭曲

## 项目结构

```text
files/
- orchestrator.py          # 主管线：采音 + ASR + MT + TTS 的并行调度
- main.py                  # TTS 后端与播放逻辑
- aec.py                   # AEC 接口层，已预接 pyaec backend，未启用时安全透传
- playback_bus.py          # 播放参考音频环形缓冲
- audio_player.py          # 受控音频播放，负责把 render reference 写入 playback_bus
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
  - AEC 接口层（当前默认透传）
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
- 播放：
  - 优先走受控播放链路
  - 播放音频可被写入 `playback_bus`
  - 为后续 AEC 提供 render reference

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
- TTS 调度

当前已经包含：

- 独立音频输入队列
- 独立 `ASR worker`
- 独立 `MT worker`
- 独立 `TTS worker`
- Qwen ASR websocket 断线后的自动重连
- AEC 接口层接入点

### `aec.py`

负责 AEC 接口封装：

- 初始化 AEC 接口
- 接收麦克风音频帧
- 从 `playback_bus` 读取 render reference
- 输出处理后的 capture 音频

当前默认后端是透传，不改变音频。

### `playback_bus.py`

负责 render reference 管理：

- 保存近期播放音频
- 维护播放状态
- 为未来真实 AEC 后端提供参考音频

### `audio_player.py`

负责受控播放：

- 尽量使用 `sounddevice + soundfile` 进行程序内可控播放
- 播放时把 render reference 写入 `playback_bus`
- 如受控播放失败，再退回旧的黑盒播放器

### `main.py`

负责 TTS：

- 优先尝试 `Qwen TTS API`
- 失败时回退到本地后端
- 当前本地回退默认是 `XTTS`
- 播放环节已经接到 [`audio_player.py`](/C:/Users/30909/Desktop/document/files/audio_player.py)

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
- 实时主管线中的翻译调度仍由 [`orchestrator.py`](/C:/Users/30909/Desktop/document/files/orchestrator.py) 控制

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

### AEC 配置项

当前版本已经支持读取这些配置，但不要求你现在就设置：

```env
ENABLE_AEC=false
AEC_BACKEND=pyaec
AEC_SAMPLE_RATE=16000
AEC_FRAME_SIZE=160
AEC_FILTER_LENGTH=1600
AEC_DELAY_MS=120
AEC_BYPASS_WHEN_NO_RENDER=true
AEC_PLAYER_CHUNK_SIZE=1024
AEC_PLAYBACK_BUFFER_SEC=6.0
```

说明：

- 当前默认 `ENABLE_AEC=false`
- 当 `ENABLE_AEC=true` 时，会优先尝试 `AEC_BACKEND=pyaec`
- 如果 `pyaec` 未安装或初始化失败，会自动回退到 `passthrough`
- `AEC_FRAME_SIZE` 和 `AEC_FILTER_LENGTH` 是 `pyaec` 后端的内部处理参数，先保留默认值即可
- 目前实测结论是：`pyaec` 已完成接入验证，但不建议在当前项目里默认开启

## 当前主管线关键运行参数

这些值由 [`orchestrator.py`](/C:/Users/30909/Desktop/document/files/orchestrator.py) 控制：

```text
SAMPLE_RATE=16000
FRAME_SIZE=512
CHANNELS=1
ASR_INPUT_LATENCY=high
ASR_AUDIO_QUEUE_MAX_CHUNKS=64
ASR_RESULT_QUEUE_MAX=16
TTS_QUEUE_MAX=16
```

## 当前 ASR 前处理参数

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

这些参数是当前针对以下目标调过的一组默认值：

- 减少键盘/鼠标误触发
- 尽量避免吞首字
- 保持外放前提下的基本可用性

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
- `[AEC] enabled=... | backend=...`
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
- `[ASR] Qwen websocket closed, reconnecting...`
- `[Audio] input overflow | ...`
- `[Audio] capture queue full, dropping oldest buffered chunk ...`
- `[Queue] asr_text queue full, dropping oldest item ...`
- `[Queue] tts queue full, dropping oldest item ...`

## 当前已知问题

### 1. 外放场景尚未具备真正稳定的全双工能力

虽然并行流水线和 AEC 接口骨架已经完成，`aec.py` 里也已经接入了 `pyaec` 实验分支，但当前机器实测表明：开启 `pyaec` 后，双讲时近端人声会被明显抑制；关闭 AEC 反而更容易识别出用户插话。

这说明当前阶段的问题主要不在“接口没接上”，而在“当前 AEC 后端能力不足以支撑稳定的外放双讲”。耳机测试和 AEC on/off 对照测试都支持这一判断。

后续如果要把系统做成真正的外放全双工，需要接入：

- 真实 `AEC` 后端，例如 WebRTC AEC

### 2. ASR 前处理仍然存在实时压力

当前日志中仍可能持续看到：

- `[Audio] input overflow`

这说明当前前处理链路仍有性能压力，后续还可以继续优化。

### 3. Qwen ASR 与 Qwen TTS 都受网络稳定性影响

目前两者都属于远程依赖，因此可能遇到：

- websocket 连接关闭
- 空闲超时
- HTTP 下载超时

当前代码已经对 Qwen ASR 空闲断线做了自动重连；Qwen TTS 失败时会自动回退到本地 XTTS。

### 4. 本地 VAD 初始化仍可能受网络影响

当前 `Silero VAD` 通过 `torch.hub.load(...)` 初始化。在某些环境下，如果本地缓存不可用，初始化阶段仍可能访问网络。

如果后续要继续提升稳定性，可以考虑把这部分彻底本地化。

### 5. 受控播放仍可能因格式或设备问题回退到旧播放器

当前 [`audio_player.py`](/C:/Users/30909/Desktop/document/files/audio_player.py) 会优先尝试受控播放；如果失败，仍会回退到旧的黑盒播放器。

这意味着：

- 大多数 WAV 路径已经能进入 AEC-ready 播放链路
- 若回退发生，则当次播放无法为未来 AEC 提供精确 render reference

## 当前研发结论

截至这个版本，我们已经确认：

- 并行流水线方向正确
- 吞首字问题已明显改善
- 键盘/鼠标误触发已明显下降
- 耳机可显著缓解识别污染
- 当前代码已经具备 AEC-ready 架构
- 真正的外放全双工问题已经聚焦到“接入真实 AEC 后端”

这也是下一阶段的主要研究方向。

## 安全说明

不要提交以下内容：

- `.env`
- `models/`
- `voice_samples/`
- `.venv310/`

另外请确保你对参考音频拥有合法使用权限。
