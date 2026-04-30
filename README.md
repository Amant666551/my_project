# 实时语音翻译项目

这是一个实时语音翻译系统，主链路为：

`麦克风输入 -> ASR -> MT -> TTS`

当前默认组合：

- ASR：Qwen Realtime ASR
- MT：DeepSeek
- TTS：Qwen Realtime VC TTS
- Speaker Matching：SpeechBrain ECAPA

## 项目入口

- `orchestrator.py`：实时主流程
- `main.py`：TTS 后端与播放
- `record_voice.py`：录音、注册、激活、删除 voice
- `asr/speaker_matcher.py`：说话人匹配
- `app_logging.py`：日志配置

## 当前能力

- 实时 ASR、翻译、TTS 串联
- 按句播放，避免 TTS 互相打断
- Speaker Matching + 目标 voice 路由
- 本地 voice 注册库
- `TURN` 汇总日志
- 端到端时延统计
- 评测预测导出

## 运行方式

### 安装依赖

```powershell
.\.venv310\Scripts\pip.exe install -r requirements.txt
```

### 启动主流程

```powershell
python orchestrator.py
```

### 启动网页前端

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000/
```

### 启动桌面版

```powershell
python desktop\desktop_app.py
```

### 打包桌面版

```powershell
.\desktop\build_desktop.ps1
```

## 运行配置

常用的最小配置：

```env
API_ONLY=true
USE_QWEN_ASR_API=true
USE_QWEN_TTS_API=true
USE_LOCAL_MT=false
```

常见可调项：

```env
QWEN_ASR_MODEL=qwen3-asr-flash-realtime-2026-02-10
DEEPSEEK_MODEL=deepseek-chat
QWEN_TTS_MODEL=qwen3-tts-vc-realtime-2026-01-15
ASR_FINAL_ALIGNMENT_WAIT_MS=120
ASR_FINAL_ALIGNMENT_MAX_DEFERRED=8
SPEAKER_MATCHING_ENABLED=true
SPEAKER_REGISTRY_PATH=voice_samples/voice_registry.json
SPEAKER_CLUSTER_THRESHOLD=0.72
SPEAKER_REGISTRY_THRESHOLD=0.70
SPEAKER_REGISTRY_MARGIN=0.05
```

说明：

- 主流程实际使用的模型以 `.env` 为准
- `USE_LOCAL_MT=true` 时会走本地翻译服务
- `ASR_FINAL_ALIGNMENT_*` 控制远端 final 与本地切句的对齐等待窗口和积压告警阈值
- `SPEAKER_*` 主要影响会话聚类和注册库路由
- 当前系统面向闭集说话人场景

## 说话人匹配

当前流程是：

1. 前端切句
2. 提取 speaker embedding
3. 会话内聚类
4. 再和注册库匹配
5. 命中后路由到对应 `voice_id`

如果分数不够，系统会退回默认 voice。

## 日志

当前日志保留：

- `ASR final`
- `MT result`
- `TURN`
- `TTS provider`
- `LATENCY trace`
- `warning/error`

`LATENCY` 关键字段：

- `end_to_end_latency_ms`：从 ASR 开始到 TTS 首个可播放音频就绪（ready 口径）
- `end_to_end_net_latency_ms`：ready 时延减去 `tts_queue_wait_ms`
- `end_to_end_done_latency_ms`：从 ASR 开始到 TTS 播放结束（done 口径）
- `asr_to_mt_gap_ms`：ASR final 到 MT 开始间隔
- `tts_queue_wait_ms`：进入 TTS 队列到开始合成的等待时长
- `tts_done_latency_ms`：TTS 开始到播放完成

`trace_id` 说明：

- `TURN` 和 `LATENCY` 里的 `id` 都来自同一个 `trace_id`
- 正常情况下，每个 final utterance 会分配唯一 `trace_id`
- 当远端 final 和本地切句时序错位时，系统会先短暂等待对齐，再进入 fallback
- fallback 分支会创建新的 trace（不复用旧 trace），避免多个句子共享同一个 `id`
- 可通过日志关键词 `speaker_resolution_waiting` / `trace_fallback_without_pending_speaker` 观察对齐抖动

不再重点打印：

- `partial`
- `deepseek_call`
- `deepseek_done`
- `scene_result`

默认日志文件：

- `logs/pipeline.log`

## 评测

如果要导出评测预测，设置：

```env
EVAL_PREDICTIONS_PATH=eval/preds/run01.jsonl
```

评分说明见 `eval/README.md`。

当前主评测口径仍以 `end_to_end_latency_ms`（ready）为主；`done` 相关字段用于补充分析播放尾时延。

## 依赖

核心依赖包括：

- `speechbrain`
- `torch`
- `torchaudio`
- `soundfile`
- `sounddevice`
- `dashscope`
- `requests`
- `python-dotenv`

## 已知边界

- 短句更容易导致 speaker 波动
- 说话人很相近时路由更容易抖动
- Qwen ASR / Qwen TTS 仍受网络影响

## 安全说明

不要提交：

- `.env`
- `models/`
- `voice_samples/`
- `.venv310/`
- `logs/`
