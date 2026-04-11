# 实时语音翻译项目

当前主链路：

`麦克风输入 -> ASR -> MT -> TTS`

当前默认组合：

- ASR: Qwen Realtime ASR
- MT: DeepSeek
- TTS: Qwen Realtime VC TTS
- Speaker Matching: SpeechBrain ECAPA

当前主入口：

- `orchestrator.py`：实时主流程
- `main.py`：TTS 后端与播放
- `record_voice.py`：录音、注册、激活、删除 voice
- `asr/speaker_matcher.py`：SpeechBrain 说话人匹配
- `app_logging.py`：日志过滤与文件清理

## 当前状态

项目现在已经具备这些能力：

- Qwen 实时 ASR
- DeepSeek 翻译
- Qwen 实时 VC TTS
- 按句串行播放 TTS，避免互相打断
- SpeechBrain 说话人匹配
- 说话人到 voice 的动态路由
- `TURN` 汇总日志
- 本地 voice 注册库
- 单文件日志 `logs/pipeline.log`
- 启动时清空日志，并按配置周期清空同一个日志文件

## 目录结构

```text
files/
  orchestrator.py
  main.py
  api.py
  app_logging.py
  record_voice.py
  translator.py
  README.md
  requirements.txt
  .env
  asr/
    speaker_matcher.py
    aec.py
    audio_player.py
    hotword_manager.py
    playback_bus.py
  mt/
    prompt_context.py
    scene_analyzer.py
  voice_samples/
    voice_registry.json
    voice_XXX.wav
  models/
    speechbrain/
    torch_hub/
  logs/
    pipeline.log
```

## 运行方式

### 1. 安装依赖

如果你使用当前仓库里的虚拟环境：

```powershell
.\.venv310\Scripts\pip.exe install -r requirements.txt
```

### 2. 启动主流程

```powershell
python orchestrator.py
```

### 3. 启动前端控制台

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000/
```

## 当前模型

### ASR

- 模型：`qwen3-asr-flash-realtime-2026-02-10`
- 地址：`wss://dashscope.aliyuncs.com/api-ws/v1/realtime`
- 语言：`zh`

### MT

- 提供方：DeepSeek
- 模型：`deepseek-chat`

### TTS

- 模型：`qwen3-tts-vc-realtime-2026-01-15`
- 地址：`wss://dashscope.aliyuncs.com/api-ws/v1/realtime`
- 模式：`commit`
- 支持按句传入 `voice`，每句重新建立 realtime session，避免旧 session 复用问题

### Speaker Matching

- 后端：SpeechBrain ECAPA
- 模型源：`speechbrain/spkrec-ecapa-voxceleb`
- 本地缓存目录：`models/speechbrain/spkrec-ecapa-voxceleb`
- 默认设备：CPU，如果可用也可切到 CUDA

说明：

- 当前匹配流程是“按句提 embedding -> 会话内聚类 -> 注册库匹配 -> TTS 路由”
- 优先使用本地缓存模型
- 已静音 `huggingface_hub` 的常见启动 warning，不改变实际功能

## API_ONLY 模式

项目支持全局开关：

```env
API_ONLY=true
```

当 `API_ONLY=true` 时：

- ASR 强制使用 Qwen API
- MT 强制使用 DeepSeek
- TTS 强制使用 Qwen API
- 不初始化本地 zipformer fallback
- 不初始化本地 XTTS / OpenVoice / Edge fallback

推荐配置：

```env
API_ONLY=true
USE_QWEN_ASR_API=true
USE_QWEN_TTS_API=true
USE_LOCAL_MT=false
```

## 关键环境变量

### ASR

```env
USE_QWEN_ASR_API=true
ASR_MODE=api_only
QWEN_ASR_MODEL=qwen3-asr-flash-realtime-2026-02-10
QWEN_ASR_URL=wss://dashscope.aliyuncs.com/api-ws/v1/realtime
QWEN_ASR_LANGUAGE=zh
```

### MT

```env
DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
USE_LOCAL_MT=false
MT_URL=http://127.0.0.1:8000/translate
MT_SOURCE_LANG=zh
MT_TARGET_LANG=en
MT_TIMEOUT_SEC=8
MT_CONTEXT_ENABLED=true
MT_CONTEXT_MAX_TURNS=3
MT_CONTEXT_MAX_CHARS_PER_TURN=80
MT_SCENE_ANALYZER_ENABLED=true
MT_SCENE_ANALYZER_MODEL=deepseek-chat
MT_SCENE_ANALYZER_TIMEOUT_SEC=8
MT_SCENE_ANALYZER_REFRESH_TURNS=10
```

### TTS

```env
USE_QWEN_TTS_API=true
TTS_MODE=api_only
QWEN_TTS_MODEL=qwen3-tts-vc-realtime-2026-01-15
QWEN_TTS_VOICE=your_voice_id
QWEN_TTS_URL=wss://dashscope.aliyuncs.com/api-ws/v1/realtime
QWEN_TTS_SESSION_MODE=commit
QWEN_TTS_VOICE_SAMPLE=voice_samples/voice_006.wav
VOICE_SAMPLE=voice_samples/voice_006.wav
```

### Speaker Matching

当前代码中会读取这些配置：

```env
SPEAKER_MATCHING_ENABLED=true
SPEAKER_REGISTRY_PATH=voice_samples/voice_registry.json
SPEAKER_EMBEDDING_SAMPLE_RATE=16000
SPEAKER_CLUSTER_THRESHOLD=0.50
SPEAKER_REGISTRY_THRESHOLD=0.20
SPEAKER_REGISTRY_MARGIN=0.05
SPEAKER_MIN_DURATION_SEC=0.5
SPEAKER_ACTIVE_MODEL_ONLY=true
```

说明：

- `SPEAKER_CLUSTER_THRESHOLD`
  控制会话内是否归到已有 `speaker_x`
- `SPEAKER_REGISTRY_THRESHOLD`
  控制是否命中注册库里的某个 `voice_00x`
- `SPEAKER_REGISTRY_MARGIN`
  控制 `top1` 和 `top2` 至少要拉开多少差距
- `SPEAKER_ACTIVE_MODEL_ONLY=true`
  只加载和当前 `QWEN_TTS_MODEL` 一致的 voice

当前这组阈值更偏向封闭集场景，也就是默认发言人主要来自本地注册库。

## 声音管理

### 录制并创建新 voice

```powershell
python record_voice.py
python record_voice.py --duration 25
```

行为：

- 录制新的样本到 `voice_samples/voice_XXX.wav`
- 调用 Qwen 声音注册接口
- 写入 `voice_samples/voice_registry.json`
- 自动同步 `.env` 中当前使用的 TTS voice

### 查看已保存的 voice

```powershell
python record_voice.py --list
```

### 激活某个 voice

```powershell
python record_voice.py --activate 2
```

### 删除本地 voice

```powershell
python record_voice.py --delete 1
python record_voice.py --delete 1 3 5
```

删除行为：

- 删除本地注册记录
- 删除对应的本地样本 wav
- 如果删掉的是当前 active voice，会自动切到剩余的第一个 voice

注意：

- 当前 `--delete` 是本地删除
- 不会删除阿里云侧已经创建好的远程 `voice_id`

## 当前本地注册库

程序实际读取的是：

- `voice_samples/voice_registry.json`

如果你想确认当前本地有哪些 voice，直接执行：

```powershell
python record_voice.py --list
```

目前代码并不会把 README 里的示例当成真实数据源，真实路由完全以本地注册库为准。

## 说话人匹配说明

当前流程：

1. 前端切句
2. 每句音频送入 SpeechBrain
3. 计算 speaker embedding
4. 先做会话内 `speaker_x` 聚类
5. 再和注册库做匹配
6. 命中后将该句 TTS 路由到对应 `voice_id`

说话人日志仍然保留这种映射关系：

```text
speaker_x -> voice_00x
```

例如：

```text
speaker_route | session=speaker_1 -> registry=voice_004 -> voice=qwen-tts-vc-...
```

## 推荐看哪几种日志

当前最有用的是这几类：

- `ASR final`
- `MT result`
- `TURN`
- `TTS provider`
- `LATENCY trace`
- warning / error

其中 `TURN` 是现在最适合人工排查的一行汇总日志，例如：

```text
turn | id=3 | speaker=speaker_2 | route=voice_006 | voice=qwen-tts-vc-myvoice-... | top1=voice_006:0.436 | top2=voice_004:0.081 | asr=下课了，还得去开会，真不行了他。 | mt=Class is over, and I still have to go to a meeting...
```

含义：

- `id`：当前句子的 trace id
- `speaker`：会话内 speaker 编号
- `route`：最终路由到哪个注册 voice
- `voice`：TTS 实际使用的远程 `voice_id`
- `top1`：当前句子最像谁
- `top2`：第二像谁
- `asr`：原始识别文本
- `mt`：翻译结果

如果只想快速看整体是否跑通，优先看：

1. `ASR final`
2. `MT result`
3. `TURN`
4. `TTS provider`

## 日志文件策略

当前默认日志文件：

- `logs/pipeline.log`

当前行为：

- 程序每次启动时先清空一次 `pipeline.log`
- 运行过程中每隔 `LOG_ROTATE_MINUTES` 分钟清空同一个文件
- 不再按天生成新日志文件

对应配置：

```env
LOG_FILE_NAME=pipeline.log
LOG_DAILY_ONLY=false
LOG_ROTATE_MINUTES=30
```

这里的“清空”是清掉同一个文件内容，不是生成新的滚动文件名。

## Latency 日志

格式示例：

```text
trace | id=12 | asr_latency_ms=... | asr_first_partial_ms=... | asr_final_tail_ms=... | mt_scene_analyzer_latency_ms=... | mt_translator_latency_ms=... | tts_latency_ms=... | end_to_end_latency_ms=... | scene_cache_hit=False | partials=...
```

主要含义：

- `asr_latency_ms`
  从开始说话到前端判定结束
- `asr_first_partial_ms`
  从开始说话到第一条 partial
- `asr_final_tail_ms`
  从前端结束到最终 final 返回
- `mt_translator_latency_ms`
  翻译耗时
- `tts_latency_ms`
  TTS 从请求到首音频可播放耗时
- `end_to_end_latency_ms`
  从开始说话到 TTS 首音频可播放的总耗时

## 依赖说明

当前 `requirements.txt` 里和本项目强相关的核心依赖包括：

- `speechbrain`
- `torch`
- `torchaudio`
- `soundfile`
- `sounddevice`
- `dashscope`
- `requests`
- `python-dotenv`

其中 Speaker Matching 相关特别依赖：

- `speechbrain`
- `torchaudio`
- `soundfile`

本地缓存目录主要包括：

- `models/speechbrain/spkrec-ecapa-voxceleb`
- `models/torch_hub/snakers4_silero-vad_master`

## 已知边界

### 1. Speaker 分数不要按绝对值机械理解

在当前 SpeechBrain + 当前样本条件下，更建议看：

- `top1` 是谁
- `top1` 和 `top2` 的差距
- 多句是否稳定命中同一个 voice

### 2. 短句更容易波动

像“你好啊”“可以听到吗”这种很短的句子，speaker embedding 稳定性通常更差。

更适合测试匹配效果的句子：

- 2 到 8 秒
- 内容稍长
- 发音自然

### 3. Realtime 服务仍然受网络影响

Qwen ASR / Qwen TTS 仍可能遇到：

- websocket 关闭
- 空闲超时
- 网络抖动

## 安全说明

不要提交这些内容：

- `.env`
- `models/`
- `voice_samples/`
- `.venv310/`
- `logs/`
