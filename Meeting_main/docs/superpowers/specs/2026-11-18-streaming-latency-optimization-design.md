# 流式 ASR 延迟优化与 Partial 稳定设计

> 状态：**已实施，部分结论被实测推翻**（2026-11-18）
> 实测报告见 `reports/streaming_latency_2026-11-18.md`：
> §4.3 的 Partial 稳定器经证明在该引擎上是恒等变换（输出严格单调追加，79/79 句零回改），
> 已按报告砍掉；§4.2 的自适应端点降级为可调参数 `--silence-ms`，默认仍 1000ms。
> 其余（参数层、评测层、长跑）按本设计实施。
> 项目：`D:\Deepseek\Meeting_main`
> 目标：在不修改上游 `diarization_cl-main` 的前提下，把流式链路做出
> 「可量化的延迟下降 + 可量化的草稿稳定性提升」，并把评测沉淀成三份报告。

## 1. 背景

现有链路已经能跑通三态（`asr_partial` 灰斜体 → `asr_final` → `llm_confirmed` 黑正体），
但三个关键参数是硬编码的、草稿是无稳定机制的、延迟从未被测量过：

| 事实 | 位置 | 后果 |
|---|---|---|
| `chunk_size_asr=[0,10,5]` 硬编码 600ms | `.zhang/streaming_asr_engine.py:82` | 无法做 chunk 对比 |
| `chunk_stride_samples=9600` 与上面的 chunk 强耦合 | 同上 `:86` | 改 chunk 必须同时改 stride |
| `--chunk-ms` 只改喂流节奏，内部仍切 9600 | `run_offline.py:180` | **拿它做实验等于没做** |
| VAD 每 600ms 才被调用一次 | `streaming_asr_engine.py:165` | 端点分辨率被 stride 钉死 |
| `asr_partial` 每帧整段覆盖，无稳定/波动之分 | `diar_web.html:544` | 草稿反复改写 |
| 上游 `.zhang/` 被指纹锁死 | `bootstrap.py:39` | 改上游 = 破红线 + 破可复现 |

## 2. 范围与非目标

**范围**

- 把 ASR chunk / VAD chunk / look_back / 端点阈值参数化，可在线扫描。
- 新增 Partial 假设稳定机制（稳定前缀 + 波动尾巴），并打通到前端两段渲染。
- 新增自适应 VAD 端点策略，与固定阈值做 A/B。
- 新增延迟探针与评测台，产出 chunk 扫描 / 稳定性 / 长跑三份报告。
- 30–60 分钟连续录音长跑验证。

**非目标**

- **不做热词**（流式模型 `paraformer_streaming` 的 `model.py` 零处 `hotword` 支持；
  只有 `contextual_paraformer` / `seaco_paraformer` / `fun_asr_nano` 支持，
  且 `generate` 对未知参数是 `kwargs.get` 静默忽略——传了不报错也不生效）。
  本次不碰热词，也不动 `offline_asr.py` 里那个一直没人传的 `hotwords` 形参。
- 不改 `diarization_cl-main` 下任何文件，不刷新 `bootstrap.BASELINE`。
- 不做真流式上行（麦克风边说边传）。现有「整段上传 + 服务端墙钟回放」足以测量
  **模型侧**延迟；麦克风采集与上行是两段常数，横比时自动抵消。
- 不改会后离线二遍 ASR、说话人回填、LLM 润色。
- 不改物理时间戳算法：`start_ms` / `end_ms` 仍只由 `samples * 1000 // 16000` 得出。

## 3. 延迟账本（本设计的纲）

Paraformer 流式模型的 `chunk_size=[0, c, r]` 中每一项的单位是 **60ms 帧**（960 样本）。
`c` 是每次送入的音频块长，`r` 是右上下文前瞻——**前瞻也是延迟**。

| 环节 | 现状 (600ms) | 可动范围 | 单位延迟 |
|---|---|---|---|
| 攒块（stride） | 600ms | 300 / 480 / 600 / 960 | `c * 60` |
| 右上下文前瞻 | 300ms | 120 / 240 / 300 / 480 | `r * 60` |
| 推理（CPU） | 不可控 | — | 实测 RTF |
| VAD 端点（定稿才走） | 1000ms 静音 + 600ms 网格量化 | 自适应 400~1200 | 端点值 |
| 事件→渲染 | <50ms | — | — |

推得两条硬结论：

1. **首字上屏下界 = `(c + r) * 60` + 推理**。600ms 档是 900ms + 推理 ≈ 1.2s；
   300ms 档是 420ms + 推理 ≈ 0.7s。**砍 chunk 是唯一能真正砍首字延迟的旋钮。**
2. **VAD 端点只在定稿路径上**，与首字延迟无关。它和 chunk 共用同一个 stride，
   所以缩小 chunk 会顺带把端点量化误差从 600ms 降到 300ms——**一根绳上的两个收益**。

## 4. 架构：一个子类承载三层

上游 `StreamingSession` 的五个关键量**全是实例属性**，不是常量：

```
feed_pcm_chunk  → self.chunk_stride_samples
_step_chunk     → self.chunk_size_vad / self.chunk_size_asr
                  self.encoder_chunk_look_back / self.decoder_chunk_look_back
                  self.max_end_silence_time      （每次调用都重新读取）
```

因此只需子类化 + 覆写属性 + 覆写 `feed_pcm_chunk` 做事件后处理，
**上游一个字节都不用改**，`bootstrap.upstream_fingerprint()` 保持 `{}`。

```
                     run_offline.run(session_factory=TunedSession...)
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
          StreamingSession（上游，只读）      TunedSession（新增）
                    │                              │
                    │  ① 参数层：__init__ 后覆写 5 个实例属性
                    │  ② 端点层：每次 feed 前按句长/能量重设 max_end_silence_time
                    │  ③ 输出层：super().feed_pcm_chunk() 返回后改写 asr_partial
                    ▼                              ▼
             asr_partial(原始)            asr_partial(+stable/volatile)
```

`run_offline.run()` 已有 `session_factory` 形参，**编排层零改动**，只加 CLI 参数。

### 4.1 参数层：`streaming_tuning.py`

```python
class TunedSession(StreamingSession):
    def __init__(self, *a, asr_chunk=(0, 10, 5), look_back=(4, 1),
                 vad_chunk_ms=600, endpoint=None, stabilize=None, **kw):
        super().__init__(*a, **kw)
        self.chunk_size_asr = [0, asr_chunk[1], asr_chunk[2]]
        self.chunk_stride_samples = asr_chunk[1] * 960
        self.chunk_size_vad = vad_chunk_ms
        self.encoder_chunk_look_back, self.decoder_chunk_look_back = look_back
        ...
```

- 允许档位：`(0,5,2)` 300ms / `(0,8,4)` 480ms / `(0,10,5)` 600ms / `(0,16,8)` 960ms。
  其他值直接 `ValueError`，避免跑出一个没人验证过的档。
- 默认值必须与上游完全一致（`(0,10,5)` / `(4,1)` / `600`），保证行为不回归。

### 4.2 端点层：自适应 VAD

`max_end_silence_time` 每帧被重新读取，所以可以逐块调整。策略按当前句长与尾部能量分档：

| 条件 | 阈值 | 理由 |
|---|---|---|
| 句长 < 1.5s | 600ms | 短句易被切碎粘到下一句，但仍要早点断 |
| 1.5s ≤ 句长 ≤ 8s | 1000ms | 与现状一致，不动 |
| 句长 > 8s 且本块 RMS 明显下降 | 700ms | 长句换气点，早断可降定稿延迟 |
| 其余 | 1000ms | 兜底 |

**已知风险**：fsmn-vad 是否支持在流中途改变 `max_end_silence_time` 未经实验证实。
若实测无效，退化为「会话开始时按音频整体电平一次性设定」并在报告里写明。

### 4.3 输出层：`partial_stabilizer.py`

纯 Python、无模型、无依赖，可在毫秒级跑完。

**输入**：累计文本（上游 `raw_text` 是逐块追加的，天然单调增长）。
**状态**：最近 `N` 帧文本的环形缓冲、已提交前缀 `committed`、回退计数。
**算法**：

1. `candidate = LCP(history[-N:])` —— 最近 N 帧的最长公共前缀。
2. `committed` 只增不减：若 `candidate` 以 `committed` 开头，则延伸到 `candidate` 与
   当前 `committed` 的公共前缀；否则 `committed` 不动、`regressions += 1`。
3. `volatile = current_text[len(committed):]`。
4. 输出 `stable_text = committed`、`volatile_text = volatile`。

**为什么 committed 只增不减**：已上屏的字一旦被回收，用户视觉上就是「跳字回退」，
比晚一点稳定更糟。真正的纠正留给 `asr_final` 整段替换——那一帧本来就是整体重绘。

**迟滞参数 N**：默认 `N=2`（连续两帧仍然存在才算稳定）。N=1 几乎不滤抖，
N=3 会让稳定字明显滞后，两者都进实验矩阵。

**状态生命周期**：稳定器按 `utt_id` 持有状态，出现新的 `utt_id` 时整块重置；
同一 `utt_id` 内不掉帧。`asr_final` 到达时先快照当时的 `stable_text` 记为
`committed_end`（供 §5 的「句尾一致率」使用），再清掉该句状态。
`max_utt_duration_ms=12000` 强切会产生新 `utt_id`，因此也被这条规则覆盖。

### 4.4 事件契约（向后兼容）

`asr_partial` 增加三个**可选**字段，老前端不读就不受影响：

```json
{"t":"asr_partial","utt_id":"utt_0001","start_ms":0,"current_ms":1200,
 "text":"那我们这个","stable_text":"那我们","volatile_text":"这个","regressions":0}
```

- `text` 语义不变（仍是完整累计文本），保证 `events.ndjson` 的旧消费者照常工作。
- 未开启稳定器时三个字段都不出现。
- 前端 `diar_web.html`：有 `stable_text` 时两段渲染（稳定段深灰正体、波动段浅灰斜体），
  没有则维持现在的整段灰色斜体。`meeting_bridge.js` 只负责透传，不做文本改写。

### 4.5 评测层：`latency_probe.py` + `chunk_sweep.py`

延迟探针不碰引擎，只挂 `on_event`：

```python
run(wav, dir, fast=False, session_factory=tuned_factory,
    on_event=lambda ev: log.append((time.monotonic(), fed_samples/16.0, ev)))
```

因为 `fast=False` 时 `run()` 用 `time.sleep` 把送流节流到墙钟，事件时刻可以直接和
音频时间轴对齐。记 `T0` 为第一块音频开始送入的 `time.monotonic()`，
音频位置 `t`（毫秒）在墙钟上的对应时刻定义为 `W(t) = T0 + t/1000`。则：

- `TTFT` = 首个 `asr_partial` 时刻 − `W(该句 start_ms)`
- `TTSC` = 首个带 `stable_text` 的 `asr_partial` 时刻 − `W(该句 start_ms)`
- `FinalLatency` = `asr_final` 时刻 − `W(该句 end_ms)`
- `RTF` = 总推理耗时 / 音频秒数（只在 `fast=True` 下有意义）

`W(t)` 成立的前提是送流严格按墙钟节流且无掉块；`fast=True` 时该等式不成立，
所有基于 `W` 的指标一律作废——这是 §9 里「两套模式不许混用数字」的具体含义。

## 5. 指标定义（口径必须唯一）

| 指标 | 定义 | 用在哪 |
|---|---|---|
| TTFT | 见 4.5 | E1/E3 |
| TTSC | 见 4.5 | E2 |
| FinalLatency | 见 4.5 | E3 |
| 抖动率 J | `累计被改写的已上屏字符数 / 累计上屏字符数` | E2 |
| 回退次数 R | `regressions` 总和 | E2 |
| 句尾一致率 | `1 - edit(committed_end, raw_text_final) / len(raw_text_final)` | E2 |
| CER / Sub / Del / Ins | 复用 `cer_eval.py` 现有口径 | E1/E4 |
| 切碎率 | 时长 < 500ms 的句占比 | E3 |
| RTF | 见 4.5 | E4 |

**J 的算法**：每帧把上一帧的 `stable_text` 记为已上屏，本帧 `stable_text` 若不以它为前缀，
差额字符数计入被改写量。这个口径能被单测覆盖，不依赖人工判断。

## 6. 实验矩阵

| 编号 | 配置 | 音频 | 模式 | 回答什么问题 |
|---|---|---|---|---|
| E1 | chunk ∈ {300,480,600,960}，其余固定 | 10min 真实会议 | 实时 | 延迟降多少、CER 涨多少 |
| E2 | E1 最优 chunk × N ∈ {1,2,3} | 同上 | 实时 | 稳定器的迟滞怎么选 |
| E3 | 固定 1000ms vs 自适应端点 | 同上 | 实时 | 定稿延迟 vs 切碎率 |
| E4 | 最终配置 | AISHELL-4 `L_R003S01C02` 39.4min（8ch 只取 ch0） | fast | 长跑吞吐 / 内存 / CER 是否漂移 |
| E5 | 最终配置 | 10min 真实会议 | 实时 | 端到端三态演示 |

E1 每个配置跑两遍：一遍 `--no-llm` 单独量引擎侧，一遍带润色确认端到端不回归。

## 7. 测试计划

新增四个**不加载模型**的测试文件：

1. `tests/test_tuning_params.py`
   - 四档 chunk 都能正确推出 stride（4800/7680/9600/15360）。
   - 非法 chunk 抛 `ValueError`。
   - 默认构造出的五个属性与上游常量逐项相等。
2. `tests/test_partial_stabilizer.py`
   - LCP 基本行为；N=1/2/3 的差异。
   - 单调增长时 `committed` 只增不减。
   - 输入回改（不以 committed 为前缀）时 `regressions` 自增且 committed 不变。
   - 空串、单帧、超长文本不崩。
3. `tests/test_adaptive_endpoint.py`
   - 各句长/能量分档的返回值；边界值（1.5s、8s）。
4. `tests/test_event_contract.py`
   - 关闭稳定器时 `asr_partial` **不含**新字段（严格相等，防误加）。
   - 开启时字段齐全且 `stable_text + volatile_text == text`。

外加一个 slow 测试：

5. `tests/test_tuning_equivalence.py`（`-m slow`）
   - 同一段 30s 音频，`StreamingSession` 与 `TunedSession(默认参数)` 的
     **事件序列逐条相等**。这是「参数化没有改变默认行为」的唯一硬证据。

**现有快速测试不是全绿，基线要先钉住再动工**（2026-11-18 实测）：

```
1 failed, 190 passed, 3 deselected in 10.53s
FAILED tests/test_offline_diarization_updates.py::test_turn_without_long_window_coverage_keeps_fallback_identity
```

该失败项属说话人 turn 平滑（`_smooth_turn_speakers` 的迟滞判定），与本设计无关，
本次**不修**，作为已知基线红记录。因此验收口径是「**不新增失败**」，
而不是「全绿」。动工前先保存这份测试清单，收尾时逐项比对。

`bootstrap.upstream_fingerprint()` 同日实测返回 `{}`（上游与 `BASELINE` 一致），
设计全程必须保持这个结果。

## 8. 验收标准

**P0（参数化 + chunk 扫描）**

- `TunedSession` 默认参数与基线逐事件等价（slow 等价性测试通过）。
- `bootstrap.upstream_fingerprint()` 仍返回 `{}`。
- 快速测试**不新增失败**（基线 1 failed / 190 passed，见 §7）。
- E1 产出 `reports/chunk_sweep.md`：四档 × {TTFT, FinalLatency, CER, RTF, 切碎率} 同表。
- 明确给出「延迟最优档」与「延迟/精度折中档」两个推荐值。

**P1（稳定器 + 自适应端点）**

- 抖动率 J 相对基线下降 ≥ 50%，同时 CER 劣化 ≤ 1pp；达不到就如实报告，不调指标。
- 前端两段渲染在 8736 页面上目视可辨，`asr_final` 时整段转黑无闪烁。
- E2/E3 产出 `reports/stability.md`。

**P2（长跑）**

- 39.4min 音频 `--fast` 全流程跑完，RTF ≤ 0.5，峰值内存不超 16GB 机器可承受范围。
- 10min 实时模式跑完，产出 E5 的端到端延迟表。
- 产出 `reports/longrun.md`，并更新 `HANDOFF.md` 的实测耗时表。

## 9. 风险与限制

- **fsmn-vad 中途改阈值可能无效**：见 4.2，有退化方案。
- **`--fast` 不能测量延迟**：两套模式必须分开跑，报告里不许混用数字。
- **AISHELL-4 是 8ch**：必须只取 ch0（`pan=mono|c0=c0` 或 `run_offline` 现有取列逻辑），
  严禁 `-ac 1` 下混。
- **39min 实时跑要 39 分钟**：不进 CI，只做一次性实验，结果存盘。
- **两段渲染可能引起重排跳动**：前端用固定高度的行容器；必要时退化为「稳定段加粗」而非变色。
- **稳定器只对单调增长的文本有效**：`per_utt` 模式下 cache 每句清空，句内不会回改，
  但 `max_utt_duration_ms=12000` 强切时仍可能触发回退路径，需在 E2 里观察。
- **CER 与延迟同时动**：任何一档都必须带 CER，否则「延迟降了但字全错了」的数据不可用。

## 10. 文件清单

**新增**

- `streaming_tuning.py` — `TunedSession`（三层组装 + 参数校验）
- `partial_stabilizer.py` — 稳定器纯逻辑
- `latency_probe.py` — 单次运行的延迟事件采集
- `chunk_sweep.py` — 实验矩阵驱动与报告生成
- `tests/test_tuning_params.py` / `test_partial_stabilizer.py` /
  `test_adaptive_endpoint.py` / `test_event_contract.py` / `test_tuning_equivalence.py`
- `reports/chunk_sweep.md` / `reports/stability.md` / `reports/longrun.md`

**修改**

- `run_offline.py` — 新增 `--asr-chunk` / `--vad-chunk-ms` / `--look-back` /
  `--stabilize N` / `--endpoint fixed|adaptive`；默认值保持现状。
- `diar_web.html` — `asr_partial` 两段渲染（有 `stable_text` 才生效）。
- `meeting_bridge.js` — 透传新字段，不做文本改写。
- `HANDOFF.md` — 补实测数据与操作说明。

**不动**

- `diarization_cl-main` 全目录（只读）。
- `bootstrap.py` 的 `BASELINE` 指纹。
- `session_store.py` 六文件契约（`asr_partial` 新字段不落新文件，仍进 `events.ndjson`）。
