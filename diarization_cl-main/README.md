# 长会议说话人分离 · Diarization

把一段 36–40 分钟的会议录音，切成「**谁 · 在哪一秒 · 说了什么**」的时间轴。

单声道、纯 CPU、全本地运行，音频不出机器。

`Python 3.12` · `CPU only` · `离线` · `Windows / Linux`

> 本仓库是**说话人分离引擎**（回答"这句是谁说的"）。
> 实时转写、LLM 纠错与会议前端的完整链路在配套仓库，见 [配套系统](#配套系统)。

---

## 它解决什么问题

一场 40 分钟的会议录音，转写成文字之后仍然没法用——你不知道每句话是谁说的。
要交付一份会议纪要，需要的是这样的东西：

```
00:01:35  speaker_2   这个季度的成本我们压了 8 个点
00:01:41  speaker_0   压在哪
00:01:43  speaker_2   主要是物流和仓储
```

这件事的难点不在"识别声音"，在于：

- **没有人告诉你有几个说话人**，得自己估；
- **轮次极碎**，大量发言不到 5 秒，切点偏一点就整句判错；
- **长音频会累积误差**，40 分钟跑下来聚类中心会漂。

本仓库解决的就是这三件事。

---

## 效果

```
输入   40 分钟会议录音（16 kHz 单声道）· 6 人 · 远场
输出   带 speaker 标签与时间戳的句子序列（可直接对齐 ASR 文本）
```

| 指标 | 数值 | 口径（重要） |
|---|---|---|
| 句级归属率 | **61.01%** | AliMeeting 8 场句级均值，人工标注句边界 |
| 句级归属率（VAD 口径） | 89.42% | 按 VAD 切出的句子算，**不可与上一行混用** |
| 可判句归属率 | 96.2% | AISHELL-4 39 分钟 · 6 人，人数估计精确命中 |

> ⚠️ 上表三行**是三把不同的尺子**，不能互相换算：
> 61.01% 按人工标注的句子算，89.42% 按 VAD 切出的句子算（差约 28 个点）；
> 96.2% 是"能判定归属的句子里判对的比例"，分母更小。
> 另有一层：DER 还分"参考是否截断到分析窗口"两条口径，两者差 69 个点——
> 那是帧级的第四把尺子，与上面三行无关，引用时一并标明。

---

## 核心特性

- **两级时间窗**：2.0 秒长窗定身份，0.6 秒短窗只找"换人边界"。长窗保证声纹纯度，
  短窗保证边界精度，两者分开跑。
- **自动人数**：k-NN 稀疏化 + eigengap 估计 K，不需要事先告诉它有几个人。
- **句级输出**：与 ASR 的句子边界直接对齐，不是纯帧级的 `[start, end, spk]` 三元组。
- **可解释的诊断台**：网页上并排多条泳道，拧参数秒级出结果，同时看到真值对照与
  句级四类分解（对 / 弃权 / 未映射 / 错答）。
- **全离线 CPU**：VAD、声纹、聚类都在本机跑完，不需要 GPU、不调用云服务。
- **评测闭环内置**：帧级 DER 与句级归属率两套评分器都在仓库里，改完能立刻验。

---

## 架构

```
                16 kHz 单声道 WAV
                        │
                        ▼
            ┌───────────────────────┐
            │  FSMN-VAD             │  切出语音区间（本地调过阈值的副本）
            └───────────┬───────────┘
                        ▼
            ┌───────────────────────┐
            │  CAM++ 声纹嵌入        │  192 维，按 2.0s / 0.6s 两种窗提取
            └───────────┬───────────┘
                        ▼
   ┌────────────────────────────────────────────┐
   │  聚类（v11 组件）                            │
   │  k-NN 稀疏化 → eigengap 自动定 K → 质心      │
   └────────────────────┬───────────────────────┘
                        ▼
            ┌───────────────────────┐
            │  短窗对照质心找换人边界 │  余弦边距 + 迟滞平滑 + 碎段吸收
            └───────────┬───────────┘
                        ▼
        speaker_0 / speaker_1 / … + 时间戳
```

入口链：`streaming_asr/claude_a2_v12.py` 的 `diarize()` 是生产入口，
内部调用 `claude_a2_v11.py` 的聚类组件。

---

## 快速开始

### 1. 环境

```bash
pip install -r requirements.txt          # numpy / scipy / soundfile / funasr / torch / modelscope
```

需要系统里有 `ffmpeg`（任意格式转 16 kHz 单声道）。

### 2. 模型（不入库，需自行下载）

四个模型都来自 ModelScope，权重不进仓库：

```python
from modelscope import snapshot_download

snapshot_download("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")              # VAD
snapshot_download("iic/speech_campplus_sv_zh-cn_16k-common")                   # 声纹
snapshot_download("iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online")  # 流式 ASR
snapshot_download("iic/speech_frcrn_ans_cirm_16k")                             # 降噪（可选）
```

把下载后的实际目录填进配置：

```bash
cp models.example.json models.json     # 再把 vad / spk / asr / frcrn 四项改成你的路径
```

### 3. 跑一条音频

```bash
python run.py meeting.wav --models-config models.json -o meeting.json
```

`run.py` 走的是 **c1 基线路径**（单人 / 双人场景，可选 FRCRN 降噪后再分离），
适合先把环境和模型跑通、确认没装错。

**多人会议**走生产入口 `streaming_asr/claude_a2_v12.py` 的 `diarize(wav, models)`——
2.0 秒长窗定身份、0.6 秒短窗找换人边界。目前主要通过 8735 对比台调用，
也可以直接 import（`streaming_asr/` 不是包，用路径注入）：

```python
import sys
sys.path.insert(0, "streaming_asr")
from web_server import get_models
from claude_a2_v12 import diarize

models = get_models(need_stream=False, need_offline=False)   # 只加载 VAD + 声纹
result = diarize("meeting.wav", models)
```

### 4. 起网页诊断台

```bash
# 对比台：多种配置并排跑同一场，看真值对照与句级分解
python streaming_asr/web_server.py          # http://127.0.0.1:8735

# 诊断台：选 K 决策链，拧 λ / min_win / cap / refine 秒级看变化
python web/server.py                        # http://127.0.0.1:8737
```

两个服务都只监听 `127.0.0.1`，不对外。

### 5. 跑测试

```bash
cd streaming_asr && python -B -m pytest -q      # 154 passed（2026-09-29 基准）
```

> 必须在 `streaming_asr/` 里跑：那里有 `pytest.ini` 设了 `testpaths`；
> 在仓库根跑会把 `milestones/` 下的同名测试一起收进去，报 import mismatch。

---

## 三种用法

| 场景 | 入口 | 说明 |
|---|---|---|
| 一条音频出结果 | `run.py` | CLI，输出 JSON |
| 对比多条配置 | `streaming_asr/web_server.py` → 8735 | 网页并排泳道 + 真值对照 |
| 拧选 K 参数 | `web/server.py` → 8737 | 诊断台，秒级反馈 |

---

## 项目结构

| 目录 | 内容 |
|---|---|
| `streaming_asr/` | **算法主线**：v7–v12 泳道、生产入口 `claude_a2_v12.py`、DER 评分器、8735 对比台 |
| `web/` | 选 K 诊断台（8737）：前端 + 服务 + 契约与指标校验 |
| `scripts/evaluation/` | 评测、对照与验收脚本 |
| `docs/` | 设计文档、交接记录、环境清单 |
| `artifacts/` | 实测记录与 JSON 产物 |
| `milestones/`、`archive/` | 历史版本归档（**只读**） |

---

## 已知边界

引用这些数字时必须一并声明：

- **单通道天花板**：说话人重叠时单标签输出必然错一部分。分离式提纯已尝试并否证
  （ClearVoice MossFormer2_SS 在轮替对话上退化成对半拆）。
- **参考里多是小说话人**：53 个短发言者时长占比 <5%、中位 2.0 秒；把"参考全人数"
  当已知 K 会造出假收益。
- **DER 口径**：union 分母（md-eval 口径），并人落在 `confusion` 而非 FA。
- **重叠率是区间级上界**：按标注区间相交算，不能当帧级重叠率引用。
- **数据不入库**：AISHELL-4 需自行从 [OpenSLR SLR111](https://www.openslr.org/111/) 下载。

---

## 文档

| 文件 | 内容 |
|---|---|
| `docs/HANDOFF.md` | 最新交接与主指标收口 |
| `docs/README_交接版_20261016.md` | 本 README 改写前的完整交接版（保留原样） |
| `docs/环境清单.md` | 绝对路径、本机未装的备选模型 |
| `docs/交接文档_*.md` | 各轮次的结论与"被数据推翻的判断" |

> `docs/` 里的交接记录保留了**主动标注的未验证项**（"待验证假设""不得引用"）。
> 那是工程记录，不是对外结论；对外引用请只看本页「效果」一节。

---

## 配套系统

本仓库只负责"谁说的"。完整链路还包括：

- 流式 VAD + Paraformer 增量解码，实时上屏草稿；
- 句尾锁定后异步 LLM 同音字纠错；
- 会后离线二遍重编码出终稿；
- 会议前端与说话人徽章渲染。

这套链路的延迟账本、chunk / 端点阈值的对拍数据、以及前端演示页在配套仓库。

---

## 致谢

- [FunASR](https://github.com/modelscope/FunASR) —— VAD、声纹与 ASR 推理
- [ModelScope](https://modelscope.cn) —— 模型分发
- [AISHELL-4](https://www.openslr.org/111/) —— 会议测试集
- AliMeeting —— 说话人分离评测集

---

## License

**尚未确定。** 在补上 `LICENSE` 之前，默认保留所有权利，他人不可直接复用。
若计划开源，建议 MIT 或 Apache-2.0（后者带专利授权，更适合含模型推理的项目）。
