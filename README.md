# 会议智能转写与纪要系统 · Meeting ASR & Executive Workspace

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](#快速开始)
[![Inference](https://img.shields.io/badge/Inference-CPU_Only_·_Local_First-059669)](#核心能力)
[![AliMeeting DER](https://img.shields.io/badge/AliMeeting_DER-10.25%25_(8%2F8_Exact_K)-2563EB)](#实测性能与基准数据)
[![AISHELL-4 DER](https://img.shields.io/badge/AISHELL--4_DER-4.61%25-2563EB)](#实测性能与基准数据)
[![Offline CER](https://img.shields.io/badge/Far--Field_CER-25.35%25_(RTF_0.047)-7C3AED)](#实测性能与基准数据)
[![Tests](https://img.shields.io/badge/Pytest-223_Passed-10B981)](#4-自动化测试)

把一段 **36–40 分钟的多人远场会议录音**（或实时麦克风发言），转化为**带说话人身份、毫秒级时间轴、同音字纠错与结构化待办清单**的完整会议工作台。

- **双轨引擎设计**：
  - **实时麦克风流式通道**：600 ms 增量解码，首字约 **0.92 秒**上屏（灰色斜体草稿 → 句尾静音定稿 → 异步上下文纠错转正黑体），边录边写盘，断网断电不丢音频。
  - **录音文件高精直通通道**：跳过流式冗余损耗，由 `pyannote/segmentation-3.0`（17 ms 帧分辨率）提取 **99.88%** 全覆盖语音区间，配合 SEACO-Paraformer 连续长段解码与**字级毫秒时间戳对齐**，39.4 分钟远场会议仅需 **111.2 秒**完成重转写（**RTF 0.047**，21 倍实时速度）。
- **声学与声纹全部本地 CPU 运行**：音频采样、VAD、局部幂集分割、CAM++ 声纹聚类与离线 ASR 均在本地完成；可选挂载 DeepSeek API 完成跨句同音字精修与会后一键纪要提炼。

---

## 核心交付效果

```text
[时间轴与说话人精修终稿]
00:01:35  张总 (speaker_2)   这个季度的供应链成本我们整体压降了八个百分点。
00:01:41  李经理 (speaker_0) 主要压在哪些环节？
00:01:43  张总 (speaker_2)   核心是华东仓的干线物流和包材集中采购。

[会后小秘书 · 结构化纪要]
■ 会议主题：Q3 供应链降本复盘与华东仓排期决议
■ 核心结论：确认华东仓干线物流整合方案，维持 Q4 八个点降本目标不变。
■ 待办事项：[李经理] 本周五前提交包材集采合同终稿（关联时间点 00:14:20）
```

---

## 系统总体架构

系统采用**「单一采样点时钟真相 + 实时/离线双轨流水线 + 幂等五层数据落盘」**架构。无论走实时麦克风还是录音直通，所有时间戳均严格由 16 kHz 累计采样点（`samples * 1000 // 16000`）唯一推导，杜绝墙钟漂移。

```mermaid
flowchart TB
    subgraph L1["Layer 1 · 音频接入与采样时钟层 (Audio Ingress & Sample Clock)"]
        direction LR
        MIC["🎙️ 浏览器实时麦克风<br/>WebAudio 16kHz PCM 分块流"]
        WAV["📁 会议录音文件导入<br/>WAV / MP3 / M4A (FFmpeg 取 ch0)"]
        CLK[("⏱️ session_audio.wav<br/>16 kHz 单声道采样点唯一时钟")]
        MIC -->|"POST /api/live/chunk (0.7s)"| CLK
        WAV -->|"归一化 / 写入"| CLK
    end

    subgraph L2["Layer 2 · 双轨语音转写层 (Dual-Track ASR Engine)"]
        direction TB
        subgraph T_LIVE["轨 A：实时低延迟流式轨 (Live Streaming Track)"]
            direction LR
            SVAD["FSMN-VAD 流式切句<br/>600ms 步进 / 1000ms 句尾判定"]
            SASR["Paraformer-Streaming<br/>增量解码 (首字 ≈ 0.92s)"]
            SLLM["StreamPolisher 流式纠错<br/>800字滑窗 + 停顿连续性感知"]
            GAP["漏句补偿扫描<br/>recover_uncovered_spans"]
            SVAD -->|"asr_partial (灰斜体)"| SASR
            SASR -->|"asr_final (锁死边界)"| SLLM
            SLLM -->|"llm_confirmed (正黑体)"| GAP
        end

        subgraph T_DIRECT["轨 B：录音文件高精直通轨 (Direct High-Accuracy Track)"]
            direction LR
            PSPAN["Pyannote 17ms 语音检测<br/>99.88% 远场弱音全覆盖"]
            PEAK["自适应峰值归一化<br/>远场低音量增益补偿"]
            OASR["SEACO-Paraformer 离线引擎<br/>≤25s 连续语境整段解码"]
            PSPAN --> PEAK --> OASR
        end
    end

    subgraph L3["Layer 3 · 说话人分离与字级对齐层 (Diarization & Token Alignment)"]
        direction LR
        PSEG["Pyannote Powerset 分割<br/>10s 局部窗 · 7类幂集 · 17ms 帧"]
        CAMP["CAM++ 192维纯声纹提取<br/>剔除重叠帧 + 会话均值中心化"]
        SPEC["谱聚类 + 双门限合并<br/>KNN(k=12) · Eigengap · Fisher判别"]
        HUNG["匈牙利 1-to-1 全局对齐<br/>全会议 17ms 活跃度矩阵重构"]
        ALIGN["字级毫秒时间戳对齐<br/>按 Token 中心点精确切分 sub_turns"]
        PSEG --> CAMP --> SPEC --> HUNG --> ALIGN
    end

    subgraph L4["Layer 4 · 会后并发精修与AI小秘书 (Post-Meeting Intelligence)"]
        direction LR
        FPOL["4线程并发双向精修<br/>前序+后序语境 · Prefix Cache"]
        SEC["会后小秘书 (Meeting Summary)<br/>主题 / 核心决议 / 责任人待办 / 角色推断"]
        FPOL --> SEC
    end

    subgraph L5["Layer 5 · 原子存储与极简工作台 (Atomic Persistence & Web Workspace)"]
        direction LR
        STORE[("💾 SessionStore 幂等合并<br/>session.json / summary.json<br/>speaker_names.json")]
        WEB["🖥️ 极简高管工作台 (:8736)<br/>HTTP 206 点击即播 · 发言人筛选与改名<br/>全文检索 · 纪要一键导出"]
        STORE --> WEB
    end

    CLK --> SVAD
    CLK --> PSPAN
    CLK --> PSEG
    GAP --> ALIGN
    OASR --> ALIGN
    ALIGN --> FPOL
    SEC --> STORE

    classDef clsInput fill:#F1F5F9,stroke:#475569,stroke-width:1.5px,color:#0F172A
    classDef clsStream fill:#ECFDF5,stroke:#059669,stroke-width:1.5px,color:#064E3B
    classDef clsCore fill:#EFF6FF,stroke:#2563EB,stroke-width:1.5px,color:#1E3A8A
    classDef clsLLM fill:#FFFBEB,stroke:#D97706,stroke-width:1.5px,color:#78350F
    classDef clsStore fill:#F5F3FF,stroke:#7C3AED,stroke-width:1.5px,color:#4C1D95

    class MIC,WAV,CLK clsInput
    class SVAD,SASR,SLLM,GAP clsStream
    class PSPAN,PEAK,OASR,PSEG,CAMP,SPEC,HUNG,ALIGN clsCore
    class FPOL,SEC clsLLM
    class STORE,WEB clsStore
```

---

## 核心业务时序图

下图展示从浏览器发起**实时麦克风录制**或**导入音频文件**，到后台完成流式上屏、声纹聚类、字级对齐与纪要生成的完整生命周期：

```mermaid
sequenceDiagram
    autonumber
    actor User as 👤 用户 / 会议助理
    participant UI as 🖥️ 前端工作台 (diar_web.html)
    participant API as ⚙️ stream_service.py (:8736)
    participant ASR as 🎙️ FunASR + Pyannote 引擎
    participant LLM as 🧠 DeepSeek API (Prefix Cache)
    participant Disk as 💾 SessionStore (原子落盘)

    alt 场景 A：实时麦克风会议录制
        User->>UI: 点击「开始录音」
        UI->>API: POST /api/live/start
        API->>Disk: 创建会话目录 + 初始化 session_audio.wav
        loop 每 0.7 秒实时推流
            UI->>API: POST /api/live/chunk (16kHz PCM)
            API->>Disk: 追加写入采样点
            API->>ASR: 600ms 流式 VAD + Paraformer 增量推理
            ASR-->>UI: asr_partial (灰色斜体实时字幕)
            opt 检测到 1000ms 句尾静音
                ASR->>Disk: 写入 utterances.jsonl (asr_final)
                API->>LLM: 异步提交 800 字滑窗 + 停顿间隔提示
                LLM-->>UI: llm_confirmed (替换为正黑体纠错文本)
            end
        end
        User->>UI: 点击「结束录音」
        UI->>API: POST /api/live/finish (NDJSON 流)
        API->>ASR: 扫描流式 VAD 漏段 (recover_uncovered_spans) 并补转写
    else 场景 B：导入已有会议录音 (默认 direct=1 高精直通)
        User->>UI: 拖拽导入 WAV / MP3 录音
        UI->>API: POST /api/stream_meeting?direct=1
        API->>Disk: FFmpeg 转码 16kHz 单声道 ch0 落盘
        API->>ASR: Pyannote 17ms 全覆盖检测 + SEACO 连续段离线转写
        ASR-->>UI: 实时推送分段转写进度
    end

    rect rgb(239, 246, 255)
        Note over API,Disk: 会后深度处理阶段 (Post-Meeting Finalization)
        API->>ASR: Pyannote 10s 幂集分割 + CAM++ 谱聚类 + 匈牙利全局对齐
        ASR->>Disk: 生成 diarization.json (含句内换人 sub_turns)
        API->>ASR: 字级毫秒时间戳对齐 (_assign_tokens_by_timestamp)
        ASR->>Disk: 生成 final_text.json
        API->>LLM: 4 线程并发双向语境精修 (final_polish)
        LLM->>Disk: 生成 turns_polished.json
        API->>LLM: 提炼会议主题、决议、待办与说话人角色 (meeting_summary)
        LLM->>Disk: 生成 summary.json + 幂等合成 session.json
        API-->>UI: 推送完成事件，渲染发言人徽章与会后小秘书纪要
    end

    User->>UI: 点击任意句子跳转播放 / 点击发言人改名
    UI->>API: GET /api/sessions/<id>/audio (HTTP 206 Range)
    UI->>API: POST /api/sessions/<id>/speakers (持久化真实姓名)
```

---

## 第一性原理：延迟账本与二遍精度突破

### 1. 为什么实时首字上屏稳定在 ≈ 0.92 秒？

实时延迟由四段物理与算法开销叠加而成：

```mermaid
flowchart LR
    S(["🗣️ 开始说话 (0 ms)"]) --> A["📦 采样攒块<br/>0 – 600 ms"]
    A --> B["👀 右上下文前瞻<br/>300 ms (5帧×60ms)"]
    B --> C["⚡ CPU 增量推理<br/>≈ 170 ms"]
    C --> U(["🖥️ 首字上屏<br/>中位 918 ms"])
    U --> D["🤫 句尾静音判定<br/>1000 ms"]
    D --> F(["🔒 句尾锁死定稿<br/>中位 1254 ms"])
    F -.->|"异步 FIFO 队列<br/>不阻塞下一句上屏"| L(["✨ LLM 纠错转正<br/>llm_confirmed"])

    classDef clsFixed fill:#EFF6FF,stroke:#2563EB,stroke-width:1.5px,color:#1E3A8A
    classDef clsWindow fill:#FFFBEB,stroke:#D97706,stroke-width:1.5px,color:#78350F
    classDef clsMilestone fill:#ECFDF5,stroke:#059669,stroke-width:2px,color:#064E3B

    class A,B,D clsWindow
    class C clsFixed
    class S,U,F,L clsMilestone
```

- **解码器内建攒块壁垒**：实测将送流 `chunk` 从 `600 ms` 砍半至 `300 ms`，首字延迟仍为 `926 ms`（`600 ms` 时为 `918 ms`）——Paraformer 在线解码器必须攒够声学上下文才会吐出首个 Token，因此保持 `600 ms` 默认分块兼具高吞吐与低开销。
- **可调静音开关（`--silence-ms`）**：将句尾静音从 `1000 ms` 降至 `800 ms` 可缩短约 `340 ms` 定稿等待，但长会议断句碎片会增加 1.5 倍；系统将其暴露为可调参数，默认坚守 `1000 ms` 保障语义完整性。

### 2. 为什么二遍离线架构能把远场 CER 从 35.87% 压到 25.35%？

传统会议系统常直接复用「流式 VAD 切出的碎片句子」去做离线重转写与声纹切分，存在三个根本缺陷：
1. **流式 VAD 漏检远场弱音**：单声道远场会议中，流式 FSMN-VAD 会漏掉约 25% 的轻声与抢话片段；
2. **句边界双侧 Padding 回声**：对几十个相邻短句分别加 80 ms 前后缓冲再拼回，会在拼接处产生 `"提提了"`、`"我我们"` 等重复伪影；
3. **按时长比例盲切多说话人文本**：句内换人（`sub_turns`）时按时间比例切字符串，极易把一个完整词劈成两半（如 `"小 | 区"`）。

本系统通过**第一性原理重构**彻底根除了上述三类误差：

```mermaid
flowchart LR
    subgraph OLD["❌ 传统二遍流水线 (CER 35.87% · Del 22.54% · 耗时 659s)"]
        direction TB
        O1["流式 FSMN-VAD 碎片<br/>漏检 25.85% 远场弱音"] --> O2["逐句 ±80ms 独立裁剪<br/>边界重叠产生复读伪影"]
        O2 --> O3["按时间长度比例盲切字<br/>跨说话人双音节词被劈裂"]
    end

    subgraph NEW["✅ 第一性原理重构流水线 (CER 25.35% · Del 9.94% · 耗时 111s)"]
        direction TB
        N1["Pyannote 17ms 全覆盖检测<br/>99.88% 语音召回 + 弱音峰值增益"] --> N2["连续语境大段只解码一次<br/>零边界重音 · RTF 0.047 (提速 6 倍)"]
        N2 --> N3["字级毫秒时间戳精确归属<br/>按每个字中心时间点落入 sub_turn"]
    end

    OLD ==>|"CER 净降 10.52 pt<br/>找回 +1,290 个漏字"| NEW

    classDef clsBad fill:#FEF2F2,stroke:#DC2626,stroke-width:1.5px,color:#7F1D1D
    classDef clsGood fill:#ECFDF5,stroke:#059669,stroke-width:1.5px,color:#064E3B
    class O1,O2,O3 clsBad
    class N1,N2,N3 clsGood
```

---

## 实测性能与基准数据

所有指标均在**纯 CPU 本地环境**、真实多说话人远场会议数据集上实测验证，口径严格透明：

### 1. 语音识别（ASR）精度与速度基准
测试集：**AISHELL-4 `L_R003S01C02`**（39.4 分钟真实远场会议 · 6 名说话人 · 区间重叠率 49.3% · 仅取单声道 `ch0`）

| 流水线模式 | CER (字错率) | 替换 (Sub) | 漏字 (Del) | 插入 (Ins) | 语音覆盖率 | 处理耗时 (RTF) | 适用场景 |
|---|---|---|---|---|---|---|---|
| **流式 Raw 层** (`--no-llm --no-final-asr`) | `36.90%` | 16.25% | 16.58% | 4.07% | 74.15% | 572.5 s (`0.242`) | 会议进行中实时上屏字幕 |
| **旧版二遍** (流式 VAD 段 + 离线 SEACO) | `35.87%` | 10.31% | 22.54% | 3.02% | 74.15% | 659.0 s (`0.279`) | 历史对照基线 |
| **新版高精二遍** (**Pyannote 全覆盖 + 字级时间戳对齐**) | **`25.35%`** | **10.12%** | **9.94%** | **5.29%** | **99.88%** | **111.2 s (`0.047`)** | **录音文件直通 / 会后终稿（快 21 倍实时）** |

### 2. 说话人分离（Speaker Diarization）基准
架构：`pyannote/segmentation-3.0`（10s 局部幂集分割）+ `CAM++`（192 维纯声纹）+ 会话均值中心化谱聚类（`KNN_K=12` + 簇平衡质心 + Fisher/Cosine 双门限合并）+ 匈牙利 1-to-1 全局帧对齐。

| 评测基准集 | 样本规模与难度 | 平均 DER (说话人错误率) | 人数估计 (K) 准确率 | 可判句说话人归属率 |
|---|---|---|---|---|
| **AliMeeting Eval** | 8 场远场多人高重叠会议 | **`10.25%`** *(原基线 35.39%)* | **8 / 8 (100% 精确命中)** | — |
| **AISHELL-4 Eval** | 3 场标准远场会议评测集 | **`4.61%`** | **100% 精确命中** | — |
| **AISHELL-4 39.4min 长会议** | 6 人 · 39.4 分钟连续讨论 | — | **K = 6 精确命中** | **`96.2%`** |

---

## 快速开始

### 1. 环境准备

系统需要 Python 3.12 与 `ffmpeg`（用于将任意音频格式或声道转换为 16 kHz 单声道 `ch0`）：

```bash
git clone https://github.com/zhangshuai05/my_study_meeting_asr.git
cd my_study_meeting_asr/Meeting_main

conda create -n funasr python=3.12 -y
conda activate funasr
pip install -r ../diarization_cl-main/requirements.txt
```

### 2. 模型准备

声学与声纹模型托管于 ModelScope 与 HuggingFace（本地缓存加载，权重不入库）：

```python
from modelscope import snapshot_download

# 1. 流式端点检测 (FSMN-VAD)
snapshot_download("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")
# 2. 说话人声纹提取 (CAM++)
snapshot_download("iic/speech_campplus_sv_zh-cn_16k-common")
# 3. 实时流式语音识别 (Paraformer-large Online)
snapshot_download("iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online")
# 4. 离线高精语音识别 (SEACO-Paraformer-large)
snapshot_download("iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
```

- 局部重叠感知分割采用 `pyannote/segmentation-3.0`（支持从本地 HuggingFace 缓存离线加载）。
- 本地模型路径常量集中定义在 [`diarization_cl-main/streaming_asr/web_server.py`](diarization_cl-main/streaming_asr/web_server.py) 顶部（`MODEL_ROOT` / `VAD_LOCAL` / `STREAM_ASR_LOCAL` / `OFFLINE_ASR_LOCAL`），迁移机器时修改对应路径即可。

### 3. 启动服务或命令行运行

```powershell
# Windows PowerShell 建议设置 UTF-8 输出与 DeepSeek 密钥（不配密钥则自动跳过 LLM 润色，声学主链照常运行）
$env:PYTHONIOENCODING="utf-8"
$env:DEEPSEEK_API_KEY="sk-your-key"

# 方式一：启动极简 Web 会议工作台（支持实时麦克风录音、文件导入、点击跳转播放、发言人改名、一键生成纪要）
python stream_service.py
# 浏览器访问 http://127.0.0.1:8736

# 方式二：命令行高精直通模式（处理已有录音文件最快、最准的方式）
python run_offline.py --wav meeting.wav --direct-offline

# 方式三：命令行流式仿真模式（模拟实时推流并评测首字/定稿延迟）
python run_offline.py --wav meeting.wav --fast
```

### 4. 自动化测试

仓库内置 **223 项**单元测试与端到端契约测试（覆盖流式状态机、时间戳对齐、漏段恢复、并发精修、会后纪要、实时麦克风 HTTP 接口及会话原子持久化）：

```bash
cd Meeting_main
python -m pytest -q                              # 223 passed（秒级完成，无需加载重型模型）
python -m pytest -m slow -k equivalence -v       # 真实模型事件级等价性回归测试（约 30s）
```

---

## 仓库目录结构

```text
my_study_meeting_asr/
├── README.md                          # 仓库总览、全链路架构图与核心基准数据（本文件）
├── Meeting_main/                      # 会议智能转写与工作台主工程
│   ├── README.md                      # 工程手册：数据落盘契约、API 路由表、CLI 参数与排障指南
│   ├── stream_service.py              # HTTP 工作台服务 (:8736)：实时麦克风流、文件导入、纪要与音频 Range 服务
│   ├── run_offline.py                 # CLI 编排引擎：支持流式仿真与 --direct-offline 高精直通双模式
│   ├── offline_asr.py                 # 离线高精引擎：Pyannote 17ms 语音检测、字级时间戳对齐、流式漏段补偿
│   ├── stream_llm_polish.py           # 流式纠错器：800 字上文滑窗、停顿间隔感知、同音字纠错
│   ├── final_polish.py                # 会后精修器：4 线程并发、前后双向说话人语境、Prefix Cache
│   ├── meeting_summary.py             # 会后小秘书：自动提炼会议标题、综述、议题、核心决议、待办与角色
│   ├── diarization_bridge.py          # 说话人回填桥接器：将句级标签与句内 sub_turns 注入时间轴
│   ├── session_store.py               # 原子持久化存储：幂等合并 9 类产物文件生成 session.json
│   ├── diar_web.html                  # 极简主义 Web 前端界面（无框架依赖，内联 SVG 与响应式布局）
│   ├── meeting_bridge.js              # 前端交互状态机：WebAudio 实时采集、NDJSON 流解析、音频同步跳转
│   └── tests/                         # 223 项自动化测试套件
└── diarization_cl-main/               # 说话人分离核心算法引擎
    ├── README.md                      # 声纹分离引擎文档：Powerset 分割、谱聚类与匈牙利对齐架构
    ├── streaming_asr/
    │   ├── claude_a2_v12.py           # 生产级分离入口 diarize()：Pyannote 10s 幂集 + 匈牙利全局重构
    │   ├── claude_a2_v11.py           # 谱聚类核心：会话均值中心化、k-NN 稀疏图、Eigengap 与双门限合并
    │   └── web_server.py              # 模型加载中心与 8735 算法对比诊断台
    └── web/                           # 8737 聚类选 K 交互式诊断台
```

---

## 已知工程边界

1. **单通道物理上限**：当前定位为单麦克风（或阵列单声道 `ch0`）零硬件门槛部署。在多人同时激烈抢话的纯重叠区间，Pyannote 幂集能精确剥离重叠帧以保护声纹纯度，但单通道 ASR 解码器在同一时刻只能输出主讲人文本。
2. **模型路径配置化**：本地模型路径常量目前集中在 `diarization_cl-main/streaming_asr/web_server.py` 顶部，后续可抽离为环境变量或独立配置文件。
3. **运行平台验证**：全套流水线在 Windows + CPU（Python 3.12）下完成长期稳定性与 39.4 分钟长会议基准验证；Linux 环境具备代码级兼容，尚未做端到端回归。

---

## 深入阅读文档

| 文档路径 | 核心内容 |
|---|---|
| [`Meeting_main/README.md`](Meeting_main/README.md) | **工程手册**：9 文件原子落盘契约、HTTP 接口协议、CLI 参数详解与排障指南 |
| [`diarization_cl-main/README.md`](diarization_cl-main/README.md) | **说话人分离引擎手册**：Powerset 局部感知 + 谱聚类 + 匈牙利对齐算法详解 |
| [`Meeting_main/HANDOFF.md`](Meeting_main/HANDOFF.md) | 架构演进总纲与历次性能优化实验记录 |
| [`Meeting_main/reports/`](Meeting_main/reports/) | 延迟拆解对拍、静音阈值扫描与 39.4 分钟长会议实测报告 |

---

## 致谢

- [FunASR](https://github.com/modelscope/FunASR) & [ModelScope](https://modelscope.cn) —— 提供 FSMN-VAD、Paraformer-Streaming、SEACO-Paraformer 与 CAM++ 核心声学模型
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) —— 提供 `pyannote/segmentation-3.0` 局部幂集重叠感知分割模型
- [AISHELL-4](https://www.openslr.org/111/) & [AliMeeting](https://www.openslr.org/119/) —— 提供真实远场多人会议评测基准集
