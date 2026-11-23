# 流式 ASR 延迟优化实测报告

> 日期：2026-11-18 · 机器：CPU（无 GPU）· 模型：`paraformer-large-...-vocab8404-online` v2.0.4
> 结论先行：**三个旋钮都测了，两个不该动，一个变成可调参数。真正的瓶颈不在这些旋钮上。**

## 1. 结论速览

| # | 原任务 | 实测结论 | 落地 |
|---|---|---|---|
| 2 | 不同 Chunk Size 对比 | 300–600ms 内首字延迟无显著差异（918–1087ms）；960ms 明显更差；chunk 调小只增 CPU、掉 25% 字数 | 参数化保留，**默认不动** |
| 3 | Partial 假设稳定机制 | 流式输出**严格单调追加，零回改**（79/79 句），LCP 稳定器在数学上等于恒等变换 | **不做**，改报告 |
| 4 | VAD 端点优化 | 1000→800ms 定稿延迟 −339ms，但句数 ×1.5 且明显腰斩；600ms 更碎 | **不改默认**，改成 `--silence-ms` 可调 |
| 7 | 30–60 分钟长跑 | 39.4min / 572.5s / RTF 0.242 / 596 句 / CER 38.32%，时间戳零异常 | 通过 |

三条被证伪或降级的原假设，是这份报告最有价值的部分（见 §5）。

## 2. 测法

`run_offline.run()` 在 `fast=False` 下按墙钟节流送流，所以事件时刻可以直接和音频位置对齐。
记 `T0` 为第一块音频送入时刻，音频位置 `t`(ms) 的墙钟为 `W(t)=T0+t/1000`：

- **TTFT** = 首个 `asr_partial` 时刻 − `W(start_ms)` —— 话说了多久才看见第一个字
- **定稿延迟** = `asr_final` 时刻 − `W(end_ms)` —— 话说完多久才锁定
- **RTF** = 累计 `feed_pcm_chunk` 推理耗时 / 音频秒数（包了一层计时器，**不是**墙钟/音频；
  实时模式下墙钟/音频恒等于 1.0，用它当 RTF 是错的）

首档必须先跑 6 秒热身并丢弃，否则 torch 初始化会把第一档 TTFT 抬高 200ms 以上——
第一版数据就是这么被污染的（300ms 档假性 1146ms，热身修正后 926ms）。

评测素材：`output/real_meeting/intel_meeting_10min_16k.wav` 的 **60–180s** 窗口（真实会议、
真实远场、单麦），每次跑同一段，只改一个变量。

## 3. 实验一：Chunk Size（silence 固定 1000ms）

| chunk | TTFT 中位 | TTFT p90 | 定稿中位 | 定稿 p90 | RTF | 句数 | 字数 | 漂移* |
|---|---|---|---|---|---|---|---|---|
| 300ms | 926 | 1329 | 1152 | 1367 | 0.313 | 23 | 339 | 0 |
| 480ms | 1087 | 1737 | 1416 | 1681 | 0.247 | 22 | 403 | 272 |
| **600ms** | **918** | **1287** | 1254 | 1428 | **0.230** | 24 | **450** | 303 |
| 960ms | 1302 | 1800 | 1549 | 2150 | 0.182 | 22 | 414 | 298 |

\* 漂移 = 以第一档原文为基准的 Levenshtein 距离；0 只是因为它自己和自己比。

**怎么读：**

1. **首字延迟在 300/600ms 之间没有差别（926 vs 918）**。理论预期是 300ms 快一倍，实测不是。
   原因：解码器要攒够上下文才吐第一个 token，这个过程吃掉了一到两个 chunk，
   于是 `chunk_size` 不再是首字延迟的主导项。**调小 chunk 换不来更快的字。**
2. **960ms 是明确的坏档**（TTFT 1302、p90 1800），比 600ms 差 40%。
3. **300ms 掉 25% 字数**（339 vs 450），RTF 反而最高（0.313）。更碎的分块让模型在词中间切断，
   内容直接丢。
4. **600ms 是帕累托点**：TTFT 最低、字数最多、RTF 居中。而 600ms 正是模型的**原生训练配置**
   （`[0,10,5]`）——这不是巧合。

> 补充验证：把 `chunk_size_vad` 从固定 600 改成与 stride 同步（300ms 档传 300），
> 300/480 两档结果几乎逐位不变（926→927、1087→1088）。说明 fsmn-vad 的这个参数
> 不影响端点时机，别指望从那里抠延迟。

## 4. 实验二：VAD 句尾静音阈值（chunk 固定 600ms）

| silence | TTFT 中位 | 定稿中位 | 定稿 p90 | RTF | 句数 | 字数 |
|---|---|---|---|---|---|---|
| **1000ms** | 918 | 1254 | 1428 | 0.230 | 24 | 450 |
| 800ms | 956 | **915** | 1208 | 0.233 | 36 | 476 |
| 600ms | 978 | **797** | 1065 | 0.261 | 56 | 496 |

**降阈值的账是真的**：1000→800 省 339ms 定稿延迟（−27%），1000→600 省 457ms（−36%）。

**但代价也是真的**，看切出来的句子：

```
silence=1000  │ 汽车一一比较在他们他速从来我们越来越慢比如说每个地方有一手在的另一个地方在……
silence=800   │ 去清理他名字了他他们他诉我反来会越来越慢慢比如说
              │ 在一个地方非常爽快的那个地方          ← 开始腰斩
silence=600   │ 去清理他名字                            ← 一句被切成四段
              │ 那他们
              │ 他速度反我们越来越慢
```

600ms 把一句话砍成四个碎片。800ms 已经开始在自然停顿之外切断。字数变多（450→476→496）
不是内容变多，而是**每次 `asr_final` 的 flush 尾巴**——切的次数越多，尾巴越多。

这也与 `HANDOFF.md` 记录的既有结论一致（此前已扫过 450/650/800/1000，锁定 1000ms，
450ms 会把「市场的前景」切成「市场的前 / 年金」）。**本实验复现了那个结论，没有推翻它。**

因此：**默认仍是 1000ms**，但改成可调参数 `--silence-ms` / `?silence=800`，
让「要定稿快」的场景自己承担切碎代价。

## 5. 实验三（负结果）：Partial 假设稳定机制不该做

原设计要按「最近 N 帧最长公共前缀」做稳定前缀 + 波动尾巴。**实测证明这在该引擎上恒等于恒等变换。**

统计 4 档 chunk / 79 个句子，`asr_final` 相对最后一个 `asr_partial` 的**已上屏字符回改数**：

```
_bench_5  22 句 | 尾字被改 0 句 | max=0
_bench_8  19 句 | 尾字被改 0 句 | max=0
_bench_10 19 句 | 尾字被改 0 句 | max=0
_bench_16 19 句 | 尾字被改 0 句 | max=0
```

原因在引擎里就写死了：`self.current_utt["raw_text"] += delta_text` 只追加。
逐块打印 40 秒原始 delta，**没有一次出现「新假设不以旧假设为前缀」**。

数学后果：对单调序列 `h1 ⊂ h2 ⊂ h3`，`LCP(h_{k-1}, h_k) = h_{k-1}`。
所谓「稳定前缀」永远等于上一帧全文，「波动尾巴」永远等于这一帧的增量——
把它们拼回去还是 `h_k`。**稳定器是个恒等函数，做了等于没做。**

那界面上看到的「跳字」是什么？只有一处：`llm_confirmed` 把整行换成润色稿。
那不是 partial 的问题，是句子定稿后的整体重写。要治它得在 LLM 回写那一层做守卫
（`stream_llm_polish.py` 里已有长度/数字守卫，HANDOFF 记录其阈值过严误杀正确纠错），
**不是 partial 层能解决的**。

结论：**不实现稳定器**。硬做出来只会让每个字都晚 N 帧上屏，换来零收益。

## 6. 实验四：39 分钟连续转写

```
素材   AISHELL-4 L_R003S01C02.flac（16k / 8ch，只取 ch0，不下混）
参数   chunk 600ms（原生）, silence 800ms, --fast --no-llm --no-diar --no-final-asr
结果   duration 2,362,945ms (39.4min)  →  elapsed 572.5s   →  RTF 0.242
       596 句  7530 字  句长 p50 1520ms / p90 5120ms / max 21820ms
       时间戳单调递增 ✓   零长度/负长度句 0   首句 0ms   末句 2,330,470ms
```

**CER 38.32%**（S 16.71% / D 17.38% / I 4.23%，参考 8678 字，主口径 `turns`）。

分项归因（现成的 `cer_eval.py` 输出，直接可引用）：

| 切片 | 参考字 | CER |
|---|---|---|
| VAD 覆盖率 75–100% | 3927 | 28.60% |
| VAD 覆盖率 0–25% | 134 | 79.10% |
| 跨说话人重叠区 | 4281 | 43.42% |
| 非重叠区 | 4397 | 33.34% |
| 说话人 004-M | 1472 | 18.68% |
| 说话人 006-M | 3165 | 46.82% |

**错误的主因不是流式模型，是 VAD 漏检**：25.41% 的参考字符落在 VAD 未覆盖区
（加 80ms pad 后仍 21.63%），这些区间的 CER 是 79–94%。这与 HANDOFF 里
「Del 贡献 35.4%」的旧结论一致。

> 注意口径：HANDOFF 记的 36.90% 是 `--no-diar` 流式 raw；本次 38.32% 用了 silence=800，
> 多切了句子。两者同量级，差异主要来自阈值而非模型。

## 7. 代码落地

| 文件 | 改动 | 性质 |
|---|---|---|
| `streaming_tuning.py` | **新增** `tune()` / `make_factory()`，覆写 5 个实例属性 | 上游零改动，指纹仍 `{}` |
| `bench_streaming.py` | **新增** 档位对拍台（热身、推理计时、TTFT/定稿/RTF/漂移） | 评测工具 |
| `run_offline.py` | 新增 `--silence-ms`；`session_factory` 注入路径拆开 | 默认行为不变 |
| `stream_service.py` | 新增 `?silence=800` 查询参数 | 默认不变 |
| `meeting_bridge.js` | 透传 `?silence=` | 默认不变 |
| `tests/test_streaming_tuning.py` | 7 个断言：默认等值、stride 推导、非法档位 | 新增 |

**没有改**：`diarization_cl-main` 下任何文件（`upstream_fingerprint()` 仍返回 `{}`）、
`bootstrap.BASELINE`、`session_store.py` 六文件契约、物理时间戳算法。

测试：`1 failed, 197 passed, 3 deselected`。失败的
`test_turn_without_long_window_coverage_keeps_fallback_identity` 是动工前就红的
（说话人 turn 平滑，与本改动无关），**未新增失败**。

## 8. 没做，以及为什么

- **自适应 VAD 端点**：降阈值本身已经付过代价（切碎），而「中途改阈值」在 fsmn-vad
  上是否生效未经验证。在固定阈值已被证明 1000ms 最优的前提下，自适应只能省下
  「长句换气」那一小段，收益不足以换一个未验证的机制。
- **流式热词**：`funasr/models/paraformer_streaming/model.py` 零处 `hotword`，
  只有 `contextual_paraformer` / `seaco_paraformer` / `fun_asr_nano` 支持。
  `generate()` 对未知参数走 `kwargs.get` 静默忽略——传了不报错也不生效。按需求已整块砍掉。
- **稳定/波动两段渲染**：见 §5，做了是安慰剂。
- **真流式上行**：本次测的是模型侧延迟，麦克风采集与上行是常数项，横比时抵消。

## 9. 复现

```powershell
$env:PYTHONIOENCODING="utf-8"
cd D:\Deepseek\Meeting_main
$py = "C:\Users\31817\miniconda3\envs\funasr\python.exe"

# 档位对拍（4 档 × 120s ≈ 9 分钟）
& $py bench_streaming.py --wav output/real_meeting/intel_meeting_10min_16k.wav `
    --start 60 --seconds 120 --chunks 300,480,600,960

# 静音阈值对拍
& $py bench_streaming.py --wav output/real_meeting/intel_meeting_10min_16k.wav `
    --start 60 --seconds 120 --chunks 600 --silence 800 --out output/bench/sil800.json

# 39 分钟长跑
& $py run_offline.py --wav D:\yinp\AISHELL-4\data\test\wav\L_R003S01C02.flac `
    --session-dir output/sessions/s_long39 --fast --no-llm --no-diar --no-final-asr --silence-ms 800

# CER
& $py cer_eval.py output/sessions/s_long39 `
    D:\yinp\AISHELL-4\data\test\TextGrid\L_R003S01C02.TextGrid

# demo 上试 800ms
# http://127.0.0.1:8736/?fast=1&silence=800
```
