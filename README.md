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

当前版本以 [`orchestrator.py`](/C:/Users/30909/Desktop/document/files/orchestrator.py) 为主管线入口，已经完成了七类关键工作：

### 1. ASR 前处理增强

- 高通滤波
- `RNNoise` 降噪
- `Silero VAD`
- 平滑后的双阈值门控
- pre-roll 缓冲，减少吞首字
- 自适应能量门限，根据静默噪声底动态调整 energy threshold
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

- [`asr/aec.py`](/C:/Users/30909/Desktop/document/files/asr/aec.py)
- [`asr/playback_bus.py`](/C:/Users/30909/Desktop/document/files/asr/playback_bus.py)
- [`asr/audio_player.py`](/C:/Users/30909/Desktop/document/files/asr/audio_player.py)

当前状态下：

- TTS 播放优先走受控播放链路
- 播放中的音频会被写入 `playback_bus`
- 麦克风输入会先经过 `asr/aec.py`，再进入当前前处理链路

注意：

- 这一版已经在 [`asr/aec.py`](/C:/Users/30909/Desktop/document/files/asr/aec.py) 中接入了真实 AEC 后端实验分支，当前实验后端是 `pyaec`
- 经当前机器实测，`pyaec` 在外放双讲场景下会误伤近端人声，因此当前推荐默认保持 `ENABLE_AEC=false`
- 这一步的目标是在不改 ASR / MT / TTS 模型选型的前提下，把外放全双工所需的 AEC 接口与播放参考链路先接通并验证

### 4. ASR 可观测性已接入

当前版本已经在 ASR 内部加入轻量级运行指标统计，用于辅助调参和回归验证：

- 周期性输出 `[ASR metrics] ...`
- 统计起说次数、final 次数、短句重置次数、平均句长
- 统计 `audio overflow`、队列丢弃、Qwen 重连、ASR worker 错误
- 统计 `avg_vad`、`avg_rms`、`avg_noise_floor`、`avg_dynamic_threshold`

### 5. 第二版 ASR 热词增强已接入

当前版本已经加入第二版热词后处理层：

- 热词库文件：[`asr/hotwords.json`](/C:/Users/30909/Desktop/document/files/asr/hotwords.json)
- 热词管理模块：[`asr/hotword_manager.py`](/C:/Users/30909/Desktop/document/files/asr/hotword_manager.py)
- 接入位置：`ASR final -> 热词后处理 -> MT`

当前设计原则：

- 当前默认关闭，避免把本来正确的句子误改坏
- 仅支持人工维护热词库
- 第一层做保守的 alias -> canonical 替换
- 第二层做保守的中文拼音近似匹配
- 不依赖数据库
- 不做自动联网抓词后直接生效

### 6. 第三版热词候选学习脚手架已接入

当前版本已经加入第三版候选词学习脚手架：

- 来源配置：[`asr/hotword_sources.json`](/C:/Users/30909/Desktop/document/files/asr/hotword_sources.json)
- 候选池：[`asr/hotword_candidates.json`](/C:/Users/30909/Desktop/document/files/asr/hotword_candidates.json)
- 学习脚本：[`asr/hotword_learner.py`](/C:/Users/30909/Desktop/document/files/asr/hotword_learner.py)

当前设计原则：

- 自动从本地文件或指定网页抽取候选词
- 先写入候选池，不直接进入正式热词库
- 记录来源、上下文、分数、拼音和出现次数
- 不依赖数据库

### 7. MT 语境感知提示词工程已接入

当前版本已经把 MT 从“词表增强”路线切换为“语境 prompt 工程”路线：

- 模块位置：[`mt/prompt_context.py`](/C:/Users/30909/Desktop/document/files/mt/prompt_context.py)
- 接入位置：DeepSeek MT 请求前的 `system prompt` 构建
- 当前策略：根据最近几轮对话、口语特征、问句语气和话题领域，为模型补充消歧语境

当前设计原则：

- 不额外维护外部 MT 术语知识库
- 不对 MT 输出做硬编码词表替换
- 优先依赖 DeepSeek 本身的翻译与术语能力
- 通过最近对话上下文提升专名、代词和口语碎片的理解稳定性

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
- asr/
  - __init__.py
  - aec.py                 # AEC 接口层，已预接 pyaec backend，未启用时安全透传
  - playback_bus.py        # 播放参考音频环形缓冲
  - audio_player.py        # 受控音频播放，负责把 render reference 写入 playback_bus
  - hotword_manager.py     # 第二版热词管理与后处理
  - hotwords.json          # 第二版本地热词库
  - hotword_sources.json   # 第三版候选词来源配置
  - hotword_candidates.json# 第三版候选词池
  - hotword_learner.py     # 第三版候选词学习脚本
- mt/
  - __init__.py
  - prompt_context.py      # MT 语境感知 prompt 构建
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

### `asr/aec.py`

负责 AEC 接口封装：

- 初始化 AEC 接口
- 接收麦克风音频帧
- 从 `playback_bus` 读取 render reference
- 输出处理后的 capture 音频

当前默认后端是透传，不改变音频。

### `asr/playback_bus.py`

负责 render reference 管理：

- 保存近期播放音频
- 维护播放状态
- 为未来真实 AEC 后端提供参考音频

### `asr/audio_player.py`

负责受控播放：

- 尽量使用 `sounddevice + soundfile` 进行程序内可控播放
- 播放时把 render reference 写入 `playback_bus`
- 如受控播放失败，再退回旧的黑盒播放器

### `main.py`

负责 TTS：

- 优先尝试 `Qwen TTS API`
- 失败时回退到本地后端
- 当前本地回退默认是 `XTTS`
- 播放环节已经接到 [`asr/audio_player.py`](/C:/Users/30909/Desktop/document/files/asr/audio_player.py)

### `mt/prompt_context.py`

负责 MT 语境感知 prompt 构建：

- 缓存最近几轮对话的源文本与译文
- 根据当前句子判断口语特征、问句语气、可能的话题领域
- 给 DeepSeek MT 注入“只用于消歧”的上下文提示

当前设计原则：

- 不维护外部术语知识库
- 不对 MT 输出做硬编码词表替换
- 优先依赖大模型本身的术语能力
- 通过最近对话语境和提示词工程提升一致性与自然度

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

MT 语境 prompt 配置：

```env
MT_CONTEXT_ENABLED=true
MT_CONTEXT_MAX_TURNS=3
MT_CONTEXT_MAX_CHARS_PER_TURN=80
```

说明：

- `MT_CONTEXT_ENABLED` 控制是否启用语境感知 prompt
- `MT_CONTEXT_MAX_TURNS` 控制最多带入多少轮近期上下文
- `MT_CONTEXT_MAX_CHARS_PER_TURN` 控制每轮上下文写入 prompt 前的截断长度
- 这套方案的目标不是“词表纠错”，而是“利用对话语境帮助模型消歧”

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

### ASR 可观测性配置项

```env
ASR_METRICS_ENABLED=true
ASR_METRICS_LOG_INTERVAL_SEC=30
```

说明：

- `ASR_METRICS_ENABLED` 控制是否生成 ASR 统计日志
- `ASR_METRICS_LOG_INTERVAL_SEC` 控制统计窗口长度
- 当前默认 `LOG_FILE_MODE=concise` 时，周期性 metrics 不会写入文件
- 若要查看完整 metrics，建议临时切到 `LOG_FILE_MODE=verbose`，或把控制台切到 `LOG_CONSOLE_MODE=verbose`

### 日志配置项

```env
LOG_LEVEL=INFO
LOG_CONSOLE_ENABLED=true
LOG_CONSOLE_LEVEL=INFO
LOG_CONSOLE_MODE=minimal
LOG_FILE_ENABLED=true
LOG_FILE_LEVEL=INFO
LOG_FILE_MODE=concise
LOG_DIR=logs
LOG_FILE_NAME=pipeline.log
LOG_MAX_BYTES=5242880
LOG_BACKUP_COUNT=3
```

说明：

- 当前默认是“终端精简 + 文件精简”
- `LOG_CONSOLE_MODE=minimal` 时，终端只保留三类核心结果：
  - `[ASR final  ]: ...`
  - `[MT  ]: ...`
  - `HH:MM:SS [TTS] TTS provider | ...`
- `LOG_CONSOLE_MODE=verbose` 时，终端会恢复完整运行日志
- `LOG_FILE_MODE=concise` 时，[`logs/pipeline.log`](/C:/Users/30909/Desktop/document/files/logs/pipeline.log) 只保留启动、就绪、结果链路以及所有 warning/error
- `LOG_FILE_MODE=verbose` 时，文件会保留完整 INFO 日志，包括启动细节、路由、AEC、metrics 等
- `LOG_MAX_BYTES` 和 `LOG_BACKUP_COUNT` 控制滚动日志大小与保留份数

### ASR 自适应门限配置项

```env
ADAPTIVE_ENERGY_ENABLED=true
ADAPTIVE_ENERGY_NOISE_FLOOR_ALPHA=0.02
ADAPTIVE_ENERGY_MIN_FACTOR=1.8
ADAPTIVE_ENERGY_MAX_FACTOR=3.0
ADAPTIVE_ENERGY_VAD_CEILING=0.12
```

说明：

- 当前版本默认启用自适应门限
- 只有在“当前不在说话”且 `vad_prob` 足够低时，系统才会更新噪声底估计
- 动态门限不会低于基础 `ENERGY_THRESHOLD`

### ASR 热词增强配置项

```env
HOTWORD_REWRITE_ENABLED=false
HOTWORD_MAX_REPLACEMENTS=2
HOTWORD_PINYIN_ENABLED=false
HOTWORD_PINYIN_MIN_SCORE=0.88
HOTWORD_PINYIN_MAX_REPLACEMENTS=1
```

说明：

- `HOTWORD_REWRITE_ENABLED` 控制是否启用热词后处理，当前推荐默认保持 `false`
- `HOTWORD_MAX_REPLACEMENTS` 控制单句最多替换次数，默认保守限制为 `2`
- `HOTWORD_PINYIN_ENABLED` 控制是否启用第二版拼音近似匹配，当前推荐默认保持 `false`
- `HOTWORD_PINYIN_MIN_SCORE` 控制拼音近似匹配阈值
- `HOTWORD_PINYIN_MAX_REPLACEMENTS` 控制单句最多执行几次拼音近似替换

### ASR 候选词学习配置项

```env
HOTWORD_SOURCE_TIMEOUT_SEC=12
HOTWORD_CANDIDATE_CONTEXT_LIMIT=3
HOTWORD_AUTO_PROMOTE_THRESHOLD=0.93
```

说明：

- `HOTWORD_SOURCE_TIMEOUT_SEC` 控制网页抓取超时时间
- `HOTWORD_CANDIDATE_CONTEXT_LIMIT` 控制每个候选词保留多少条上下文
- `HOTWORD_AUTO_PROMOTE_THRESHOLD` 当前只用于标记 `suggest_promote`，不会直接自动写入正式热词库

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

## 当前日志行为

### 终端默认输出

当前默认 `LOG_CONSOLE_MODE=minimal`，终端只保留三类核心结果：

- `[ASR final  ]: ...`
- `[MT  ]: ...`
- `HH:MM:SS [TTS] TTS provider | ...`

这适合日常测试时直接盯终端看识别、翻译和播报结果。

### 文件日志默认输出

当前默认 `LOG_FILE_MODE=concise`，详细日志写入：

- [`logs/pipeline.log`](/C:/Users/30909/Desktop/document/files/logs/pipeline.log)

默认会保留：

- 主管线启动
- `System ready - start speaking!`
- `ASR final`
- `MT result`
- `TTS provider`
- 所有 warning / error

默认不会保留：

- 启动阶段的大量细节 INFO
- AEC / route / model 说明
- 周期性 `[ASR metrics]`

### 需要完整日志时

如果你想临时查看完整日志：

- 终端完整输出：`LOG_CONSOLE_MODE=verbose`
- 文件完整输出：`LOG_FILE_MODE=verbose`

这样会恢复启动细节、路由、模型、AEC、metrics 等完整 INFO 日志。

## 当前已知问题

### 1. 外放场景尚未具备真正稳定的全双工能力

虽然并行流水线和 AEC 接口骨架已经完成，`asr/aec.py` 里也已经接入了 `pyaec` 实验分支，但当前机器实测表明：开启 `pyaec` 后，双讲时近端人声会被明显抑制；关闭 AEC 反而更容易识别出用户插话。

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

### 4. 第二版热词增强仍是保守规则层

当前热词增强不是模型级 biasing，而是 `ASR final` 后的保守重写层。这意味着：

- 适合修正少量稳定专名、术语和项目词
- 适合对中文专名做有限的拼音近似纠偏
- 不适合替代真正的模型级热词注入
- 当前更适合作为轻量增强，而不是大规模自动学习系统

### 5. 第三版候选词学习当前只做“发现”，不做“自动生效”

当前 [`asr/hotword_learner.py`](/C:/Users/30909/Desktop/document/files/asr/hotword_learner.py) 会从：

- 本地文本文件
- 指定网页

中抽取候选专名、术语和机构名，并写入 [`asr/hotword_candidates.json`](/C:/Users/30909/Desktop/document/files/asr/hotword_candidates.json)。

当前阶段它不会自动改线上热词库，目的是先让系统具备“自动发现词汇”的能力，同时避免误学后直接污染识别。

### 6. 如何运行第三版候选词学习

在项目根目录执行：

```powershell
python -m asr.hotword_learner
```

默认行为：

- 读取 [`asr/hotword_sources.json`](/C:/Users/30909/Desktop/document/files/asr/hotword_sources.json)
- 跳过已经存在于 [`asr/hotwords.json`](/C:/Users/30909/Desktop/document/files/asr/hotwords.json) 里的正式热词
- 将新发现的候选项写入 [`asr/hotword_candidates.json`](/C:/Users/30909/Desktop/document/files/asr/hotword_candidates.json)

### 7. 本地 VAD 初始化仍可能受网络影响

当前 `Silero VAD` 通过 `torch.hub.load(...)` 初始化。在某些环境下，如果本地缓存不可用，初始化阶段仍可能访问网络。

如果后续要继续提升稳定性，可以考虑把这部分彻底本地化。

### 8. 受控播放仍可能因格式或设备问题回退到旧播放器

当前 [`asr/audio_player.py`](/C:/Users/30909/Desktop/document/files/asr/audio_player.py) 会优先尝试受控播放；如果失败，仍会回退到旧的黑盒播放器。

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
