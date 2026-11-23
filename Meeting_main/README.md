# Meeting_main — 会议转写系统（工程手册）

> 想看"这是什么、解决什么问题"，去仓库根目录的 [README](../README.md)。
> 这一份只讲：怎么跑、参数什么意思、落盘契约是什么、出问题查哪。

一句话链路：16 kHz 单声道 WAV → 流式草稿 → 句尾固化 → 异步纠错 → 会后说话人回填 → 离线二遍终稿。

---

## 环境

| 项 | 值 |
|---|---|
| Python | 3.12（conda env `funasr`） |
| 输入 | 16 kHz 单声道 WAV；多声道只取 **ch0**，不下混 |
| 依赖 | 沿用 `../diarization_cl-main/requirements.txt`，不额外引入第三方包 |
| 系统依赖 | `ffmpeg`（非 16k/单声道时转码） |
| 模型 | VAD / CAM++ / Paraformer-Streaming / SEACO，路径见 `web_server.py` 顶部常量 |

```powershell
$env:PYTHONIOENCODING="utf-8"        # 必须；否则中文输出按 GBK 解析会乱码
$env:DEEPSEEK_API_KEY="sk-..."       # 可选；不配则跳过润色，raw_text 照常保留
```

---

## 运行

### CLI 转写

```bash
# 全速跑（离线批处理）
python run_offline.py --wav meeting.wav --fast --no-llm

# 实时节奏（按音频墙钟送流，用来量延迟）
python run_offline.py --wav meeting.wav
```

成功时打印 JSON 摘要：

```json
{
  "session_dir": "output\\sessions\\s_smoke_180",
  "status": "diarized",
  "utterances": 33,
  "speakers": ["speaker_0", "speaker_1"],
  "duration_ms": 180000,
  "elapsed_s": 183.6
}
```

### HTTP 服务（8736）

```bash
python stream_service.py          # http://127.0.0.1:8736
```

| 路由 | 作用 |
|---|---|
| `GET /` | 演示页（注入 `meeting_bridge.js`） |
| `GET /api/health` | `{ok, busy, resetting}` |
| `GET /api/sessions` | 历史会话列表 |
| `GET /api/sessions/<id>` | 单场 `session.json` |
| `POST /api/stream_meeting?name=&fast=&silence=` | 上传 WAV，返回 NDJSON 事件流 |
| `POST /api/stream_meeting/cancel` | 结束当前会议并进入会后阶段 |
| `POST /api/stream_meeting/reset` | 丢弃当前会议 |

只监听 `127.0.0.1`，不对外。

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--wav` | 必填 | 16 kHz；多声道取 ch0 |
| `--session-dir` | 自动 | 指定输出目录 |
| `--session-root` | `output/sessions` | 自动建 `s_YYYYMMDD_HHMMSS` |
| `--chunk-ms` | `600` | **送流**粒度，不影响 ASR 分块 |
| `--silence-ms` | `1000` | VAD 句尾静音阈值。降到 800 定稿快 ~0.34s，但句数 ×1.5 且容易腰斩 |
| `--max-seconds` | `0` | 只处理前 N 秒，调试用 |
| `--fast` | 关 | 全速推流，不按墙钟节流 |
| `--no-llm` | 关 | 跳过 DeepSeek 润色 |
| `--no-diar` | 关 | 跳过会后说话人回填 |
| `--no-final-asr` | 关 | 跳过离线二遍，终稿回退直播文本 |
| `--no-final-polish` | 关 | 跳过会后按 turn 的精修 |
| `--strict-polish` | 关 | 只允许同音字原位替换，测 CER 用 |

> `--chunk-ms` 只决定"一次喂多少"，引擎内部仍按 600 ms 的 stride 切块。
> 想改 ASR 分块要动 `streaming_tuning.tune()` 的 `chunk`，不是这个参数。

---

## 数据契约

### 事件流（NDJSON）

```json
{"t":"asr_partial","utt_id":"utt_0001","start_ms":0,"current_ms":1200,"text":"喂"}
{"t":"asr_final","utt_id":"utt_0001","start_ms":0,"end_ms":600,"raw_text":"喂你"}
{"t":"llm_confirmed","utt_id":"utt_0001","start_ms":0,"end_ms":600,
 "raw_text":"喂你","polished_text":"喂你。","corrected":false,"bypassed":true,"error":null}
```

顺序保证：同句 `asr_final` 早于 `llm_confirmed`；跨句按 `asr_final` 提交顺序（单消费者 FIFO）。

**时间戳铁律**：`start_ms` / `end_ms` 只由累计采样点数算出（`samples * 1000 // 16000`），
禁止使用 `time.time()`，否则与 WAV 文件的时间轴必然漂移。

### session 目录

| 文件 | 何时生成 | 说明 |
|---|---|---|
| `session_audio.wav` | 边收边写 | 16k 单声道，采样点时钟的唯一真相 |
| `utterances.jsonl` | 每句 `asr_final` | 追加写 |
| `events.ndjson` | 每个事件 | 追加写 |
| `transcript.json` | 转写结束 | T+0 快照 |
| `polished.json` | 每句润色完成 | `--no-llm` 时不存在 |
| `diarization.json` | 回填完成 | `--no-diar` 时不存在 |
| `final_text.json` | 离线二遍完成 | `--no-final-asr` 时不存在 |
| `turns_polished.json` | 会后精修完成 | 按 turn 覆盖 |
| `session.json` | 每次合并 | 幂等，可多次重写 |

所有 `*.json` 走「临时文件 + `os.replace`」原子替换；`*.jsonl` / `*.ndjson` 逐行追加并加锁。

### session.json

```json
{
  "schema_version": 1,
  "status": "diarized",
  "duration_ms": 180000,
  "utterances": [
    {"utt_id":"utt_0001","start_ms":0,"end_ms":600,"raw_text":"喂你",
     "polished_text":"喂你。","corrected":false,"bypassed":true,"error":null,
     "spk":0,"spk_label":"speaker_0","overlap_spk":null,"final_text":null}
  ],
  "turns": [{"turn_id":"turn_0001","start_ms":0,"end_ms":12000,"text":"…","spk":0,"spk_label":"speaker_0"}]
}
```

- `status`：有 `diarization.json` 为 `diarized`，否则 `transcript`
- `spk` 是 **0-based 整数**，`spk_label` 恒为 `speaker_{spk}`
- `turns[].text` 优先级：`turns_polished` > `final_text` > `polished_text` > `raw_text`
- `turns` 把相邻同说话人、间隔 8 秒内的句子合并成一段

---

## 实测数字

| 场景 | 数值 | 口径 |
|---|---|---|
| 39.4 分钟长会议 | 572.5 s，596 句 | `--fast --no-llm --no-diar --no-final-asr`，RTF 0.242 |
| 首字上屏 | 中位 918 ms | 实时模式，chunk 600 ms |
| 定稿延迟 | 中位 1254 ms | 含 1000 ms 静音判定 |
| 流式 CER | 38.32% | AISHELL-4，8ch 取 ch0 |

详细对拍（chunk 四档、静音三档、长跑）见 [`reports/`](reports/)。

---

## 测试

```bash
python -m pytest -q                            # 快测，默认排除 slow
python -m pytest -m slow -k equivalence -v     # 事件级等价夹具（加载模型，约 30 s）
```

`slow` 用例依赖 `D:\Deepseek\yinp\smoke_clip_180s.wav`，缺失自动 skip。

**已知基线红**：`test_turn_without_long_window_coverage_keeps_fallback_identity`
（说话人 turn 平滑），动工前就失败，与本仓流水线改动无关；验收口径是"不新增失败"。

---

## 排障

| 症状 | 原因 |
|---|---|
| 中文输出乱码 | 没设 `PYTHONIOENCODING=utf-8` |
| 进程卡死在启动 | Python 3.12 的 WMI 查询挂起，需保留 `platform._wmi_query` 补丁（`web_server.py` 已有） |
| 87xx 端口占用 | 服务单实例，先确认没有残留进程 |
| 转写全是空句 | 模型路径不对；改 `web_server.py` 顶部的常量 |
| 多声道结果变差 | 用了 `-ac 1` 下混；必须取 ch0 |

---

## 已知问题

- **模型路径未配置化**：写死在 `diarization_cl-main/streaming_asr/web_server.py`。
- **模型跨调用会漂**：同一份模型连续 `generate`，同一段音频可能解出不同的字（换新对象即恢复）。
  等价性夹具因此每边都重载模型，见 `tests/test_tuning_equivalence.py`。
- **`--silence-ms` 无上下界**：传 0 或超大值会让 VAD 行为异常。
