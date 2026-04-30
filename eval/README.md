# 评测骨架

这个目录用于固定三件事：

- 自建评测集格式
- baseline 定义
- 指标脚本输入输出格式

## 1. 参考集格式

建议使用 `jsonl`，每行一个 turn。

最小字段：

```json
{"utt_index": 1, "speaker_ref": "spk_a", "route_ref": "001", "zh_ref": "你好，可以听到我说话吗？", "en_ref": "Hello, can you hear me?"}
```

推荐字段：

```json
{
  "utt_id": "session01_001",
  "utt_index": 1,
  "speaker_ref": "spk_a",
  "route_ref": "001",
  "zh_ref": "你好，可以听到我说话吗？",
  "en_ref": "Hello, can you hear me?",
  "duration_ms": 2100,
  "notes": "short_utterance"
}
```

约定：

- `speaker_ref`：真实说话人标签，按人标，不按系统 speaker 编号标
- `route_ref`：这个真实说话人理论上应该路由到的目标 voice 编号
- `zh_ref`：源语文本参考
- `en_ref`：目标语文本参考

评分时，`score_eval.py` 会把 `voice_001` 与 `001` 视为同一个 route。
所以参考集中可以统一写成：

- `001`
- `002`
- `003`
- `004`

## 2. 预测文件格式

主流程现在支持通过环境变量导出结构化预测：

```env
EVAL_PREDICTIONS_PATH=eval/preds/run01.jsonl
```

每个成功翻译并进入 TTS 的 turn，会自动写一行 JSON。字段包括：

- `utt_index`
- `trace_id`
- `source_text`
- `translated_text`
- `speaker_pred`
- `route_pred`
- `voice_pred`
- `top1_label`
- `top1_score`
- `top2_label`
- `top2_score`
- `session_score`
- `registry_score`
- `registry_margin`
- `asr_latency_ms`
- `mt_latency_ms`
- `tts_latency_ms`
- `end_to_end_latency_ms`

如果参考集也带了 `utt_id` / `utt_index`，评分脚本会优先按键对齐；否则按顺序对齐。

## 3. baseline 定义

建议固定四个 baseline：

- `B0_no_routing`
  所有句子都走默认 voice，不做 speaker 匹配
- `B1_registry_only`
  每句只和注册 voice 原型比，不使用 session 历史
- `B2_current_system`
  当前系统：registry + session + threshold + margin
- `B3_proposed_method`
  在当前系统上增加 `pending`、切换惩罚、文本/时序辅助、低置信度延迟确认

论文主表至少比较这四个版本。

## 4. 指标定义

当前 `score_eval.py` 会输出：

- `route_accuracy`
- `unknown_rate`
- `speaker_consistency`
- `false_switch_rate`
- `switch_precision`
- `switch_recall`
- `switch_f1`
- `translation_exact_match`
- `translation_token_f1`
- `translation_bleu`
- `end_to_end_latency_ms_mean`
- `end_to_end_latency_ms_p95`

含义：

- `route_accuracy`
  预测 route 是否等于 `route_ref`
- `unknown_rate`
  `route_pred=none` 或 `speaker_pred=unknown` 的比例
- `speaker_consistency`
  同一真实说话人的预测 route 一致性
- `false_switch_rate`
  真实未切换时，系统错误切换 route 的比例
- `switch_precision / recall / f1`
  相邻 turn 的切换点检测表现
- `translation_*`
  翻译文本质量
- `latency_*`
  端到端时延

## 5. 使用方式

先按最小流程跑通一次：

1. 在 `.env` 里加：

```env
EVAL_PREDICTIONS_PATH=eval/preds/run01.jsonl
```

2. 正常运行系统，说 3 到 5 句测试语音
3. 确认预测文件已经生成：

```text
eval/preds/run01.jsonl
```

4. 按照同样顺序，把这些句子的真实答案填到一个参考文件里
5. 再运行评分脚本

这个最小流程的核心是：

- `preds` 文件由系统自动生成
- `refs` 文件由你手工标注真实答案
- `score_eval.py` 负责把两者对齐后算指标

下面是具体命令。

先准备参考集，例如：

```text
eval/reference_set_4spk_v1.jsonl
```

运行主流程时打开导出：

```env
EVAL_PREDICTIONS_PATH=eval/preds/run01.jsonl
```

评分：

```powershell
python eval\score_eval.py --refs eval\reference_set_4spk_v1.jsonl --preds eval\preds\run01.jsonl --out eval\results\run01_metrics.json
```

指标会打印到终端，同时写入：

```text
eval/results/run01_metrics.json
```

## 6. 先做什么数据

建议先做一个小规模闭集评测集：

- 4 个已注册说话人
- 单通道录音
- 多轮对话
- 含短句、长句、切换、相似声线
- 每人固定对应一个 `route_ref`，例如 `001 / 002 / 003 / 004`

这个自建集最适合验证你的论文点。
