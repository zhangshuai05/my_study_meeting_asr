/* 会议回放与麦克风录音桥接：把静态原型页的 window.MeetingUI 接到后端服务。
 * 2026-10-19 审计整改升级：
 * 1. [A1] 补齐 `window.MeetingUI.onStart` 真实麦克风录音能力 (Web Audio API 16kHz 单声道 WAV 采集)，
 *    点击"开始会议"实时采集麦克风，点击"结束会议"自动停止采集并进入全速转写与说话人分离。
 * 2. [A4] 前端 `cleanCjkSpaces` 同步保护中文报号数字边界，绝不破坏后端保留的数字顿号/空格。
 * 3. [A5/C1] `renderTurns` 完整透传 `text` (最终定稿)、`stream_text` (直播润色稿)、`raw_text` (原始识别稿)
 *    以及 `text_source` / `polish_status`，支持页面顶部一键切换对比三态文本。
 * 4. [C5] 页面加载及会议完成时自动刷新 `/api/sessions` 历史会话列表，支持随时回看任意历史会议。
 */
(function () {
  "use strict";

  if (!window.MeetingUI) {
    return;
  }

  if (typeof window.MeetingUI.onAudioSimulation === "function") {
    return;
  }

  var controller = null;
  var resetRequest = null;
  var finished = false;
  var rendering = false;

  // Web Audio 麦克风采集状态
  var micState = null;

  // 清除汉字间多余空格（排除中文数字报号串之间的单空格，防止 A4 抹掉数字组边界）
  function cleanCjkSpaces(str) {
    if (!str) return "";
    var s = String(str);
    var nonDigitCjk = /([\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])\s+([\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])/g;
    var cjkDigits = /^[零〇一二两三四五六七八九十百千万亿幺洞拐勾]+$/;
    function replacer(match, left, right) {
      if (cjkDigits.test(left) && cjkDigits.test(right)) {
        return left + " " + right;
      }
      return left + right;
    }
    s = s.replace(nonDigitCjk, replacer).replace(nonDigitCjk, replacer);
    return s.trim();
  }

  function emitFatal(msg) {
    if (typeof window.MeetingUI.handleEvent === "function") {
      window.MeetingUI.handleEvent({ t: "error", msg: msg, fatal: true });
    }
  }

  function waitForServerIdle(deadline) {
    return fetch("/api/health", { cache: "no-store" }).then(function (response) {
      if (!response.ok) {
        throw new Error("读取会议服务状态失败 (HTTP " + response.status + ")");
      }
      return response.json();
    }).then(function (state) {
      if (!state.busy && !state.resetting) {
        return;
      }
      if (Date.now() >= deadline) {
        throw new Error("上一场会议仍在停止，请稍后重试");
      }
      var hint = document.getElementById("transcript-hint");
      if (hint) {
        hint.textContent = "正在结束上一次会议任务...";
      }
      return new Promise(function (resolve) {
        window.setTimeout(resolve, 250);
      }).then(function () {
        return waitForServerIdle(deadline);
      });
    });
  }

  function requestBackendReset() {
    if (resetRequest) {
      return resetRequest;
    }
    resetRequest = fetch("/api/stream_meeting/reset", {
      method: "POST",
      cache: "no-store",
      keepalive: true
    }).then(function (response) {
      return response.json().then(function (result) {
        if (!response.ok || !result.ok) {
          throw new Error(result.error || ("重置失败 (HTTP " + response.status + ")"));
        }
        return result;
      });
    });
    resetRequest = resetRequest.finally(function () {
      resetRequest = null;
    });
    return resetRequest;
  }

  function formatSessionDate(startedAt, sessionId) {
    var m = null;
    if (startedAt) {
      m = String(startedAt).match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
    }
    if (!m && sessionId) {
      m = String(sessionId).match(/^s_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/);
    }
    if (!m) return sessionId || "";
    var now = new Date();
    var sameDay = (
      Number(m[1]) === now.getFullYear() &&
      Number(m[2]) === (now.getMonth() + 1) &&
      Number(m[3]) === now.getDate()
    );
    if (sameDay) return "今天 " + m[4] + ":" + m[5];
    return Number(m[2]) + "月" + Number(m[3]) + "日 " + m[4] + ":" + m[5];
  }

  // [C5] 刷新历史会话下拉框
  function refreshHistorySessions(selectedId) {
    var selectEl = document.getElementById("history-session-select");
    if (!selectEl) return;
    fetch("/api/sessions", { cache: "no-store" }).then(function (res) {
      if (!res.ok) return null;
      return res.json();
    }).then(function (data) {
      if (!data || !Array.isArray(data.sessions)) return;
      var currentVal = typeof selectedId === "string" ? selectedId : (selectEl.value || "");
      selectEl.innerHTML = '<option value="">历史会议 (' + data.sessions.length + ')</option>';
      for (var i = 0; i < data.sessions.length; i += 1) {
        var s = data.sessions[i];
        var opt = document.createElement("option");
        opt.value = s.session_id;
        var durSec = Math.round((Number(s.duration_ms) || 0) / 1000);
        var durLabel = durSec >= 60 ? (Math.round(durSec / 60) + "分钟") : (durSec + "秒");
        var dateLabel = formatSessionDate(s.started_at, s.session_id);
        var head = s.title ? (s.title + " · " + dateLabel) : dateLabel;
        opt.textContent = head + " (" + durLabel + " · " + (s.n_speakers || 1) + "人)";
        if (s.session_id === currentVal) {
          opt.selected = true;
        }
        selectEl.appendChild(opt);
      }
      if (typeof window.MeetingUI.syncDeleteButton === "function") {
        window.MeetingUI.syncDeleteButton();
      }
    }).catch(function () {});
  }

  // 用最终 turns 重绘整篇记录；turns 为空时保留流式视图。
  function renderTurns(event) {
    var ui = window.MeetingUI;
    var scrollY = window.scrollY;
    rendering = true;
    try {
      if (typeof ui.reset === "function") {
        ui.reset({ keepFile: true });
      }
      document.body.classList.add("is-diarized");
      var mapping = {};
      for (var i = 0; i < event.turns.length; i += 1) {
        var turn = event.turns[i];
        var displayText = cleanCjkSpaces(turn.text || "");
        var streamText = cleanCjkSpaces(turn.stream_text || turn.text || "");
        var rawText = String(turn.raw_text || turn.text || "");
        ui.handleEvent({
          t: "asr_final",
          utt_id: turn.turn_id,
          start_ms: turn.start_ms,
          end_ms: turn.end_ms,
          raw_text: rawText
        });
        ui.handleEvent({
          t: "llm_confirmed",
          utt_id: turn.turn_id,
          start_ms: turn.start_ms,
          end_ms: turn.end_ms,
          raw_text: rawText,
          stream_text: streamText,
          polished_text: displayText,
          text_source: turn.text_source || (turn.corrected ? "stream_polished" : "final_asr"),
          polish_status: turn.polish_status || null,
          corrected: !!turn.corrected,
          bypassed: !!turn.bypassed,
          error: null
        });
        mapping[turn.turn_id] = {
          spk: turn.spk,
          spk_label: turn.spk_label,
          overlap_spk: null
        };
      }
      ui.handleEvent({
        t: "speaker_diarized",
        session_id: event.session_id || null,
        n_clusters: event.n_clusters,
        mapping: mapping,
        speaker_names: event.speaker_names || {},
        has_audio: event.has_audio !== false
      });
      if (event.summary) {
        ui.handleEvent({
          t: "meeting_summary",
          session_id: event.session_id || null,
          summary: event.summary,
          speaker_names: event.speaker_names || {}
        });
      }
      ui.handleEvent({ t: "finish", session_id: event.session_id || null });
      if (event.session_id) {
        refreshHistorySessions(event.session_id);
      }
    } finally {
      window.scrollTo(0, scrollY);
      rendering = false;
    }
  }

  function adaptEvent(event) {
    if (!event) {
      return event;
    }
    if (event.utt_id) {
      event.source_utt_id = event.utt_id;
      event.utt_id = event.turn_id || event.utt_id;
    }
    if (typeof event.text === "string") {
      event.text = cleanCjkSpaces(event.text);
    }
    // [D1] 保持 raw_text 原样，不在桥接层篡改
    if (typeof event.polished_text === "string") {
      event.polished_text = cleanCjkSpaces(event.polished_text);
    }
    return event;
  }

  function dispatchLine(line) {
    var text = line.replace(/\r$/, "");
    if (!text) {
      return;
    }
    var event;
    try {
      event = JSON.parse(text);
    } catch (err) {
      return;
    }
    if (typeof window.MeetingUI.handleEvent === "function") {
      if (event.t === "finish") {
        finished = true;
        refreshHistorySessions(event.session_id);
      }
      if (event.t === "speaker_diarized") {
        document.body.classList.add("is-diarized");
      }
      if (event.t === "speaker_diarized" &&
          Array.isArray(event.turns) && event.turns.length > 0) {
        renderTurns(event);
        return;
      }
      window.MeetingUI.handleEvent(adaptEvent(event));
    }
  }

  function pump(reader, decoder, buffer) {
    return reader.read().then(function (result) {
      if (result.done) {
        dispatchLine(buffer + decoder.decode());
        return;
      }
      var text = decoder.decode(result.value, { stream: true });
      var parts = (buffer + text).split("\n");
      buffer = parts.pop();
      for (var i = 0; i < parts.length; i += 1) {
        dispatchLine(parts[i]);
      }
      return pump(reader, decoder, buffer);
    });
  }

  // 将 Float32 PCM 片段线性重采样为 16kHz 单声道 int16 PCM 字节（低内存实时流式上传）
  function encodePcm16kInt16(buffers, inputSampleRate) {
    var totalLen = 0;
    for (var i = 0; i < buffers.length; i += 1) {
      totalLen += buffers[i].length;
    }
    if (totalLen === 0) return new Uint8Array(0);
    var merged = new Float32Array(totalLen);
    var offset = 0;
    for (var j = 0; j < buffers.length; j += 1) {
      merged.set(buffers[j], offset);
      offset += buffers[j].length;
    }
    var targetRate = 16000;
    var samples = merged;
    if (inputSampleRate !== targetRate) {
      var ratio = inputSampleRate / targetRate;
      var newLen = Math.max(1, Math.round(totalLen / ratio));
      samples = new Float32Array(newLen);
      for (var k = 0; k < newLen; k += 1) {
        var pos = k * ratio;
        var i0 = Math.min(totalLen - 1, Math.floor(pos));
        var i1 = Math.min(totalLen - 1, i0 + 1);
        var frac = pos - i0;
        samples[k] = merged[i0] * (1 - frac) + merged[i1] * frac;
      }
    }
    var out = new Int16Array(samples.length);
    for (var s = 0; s < samples.length; s += 1) {
      var val = Math.max(-1, Math.min(1, samples[s]));
      out[s] = val < 0 ? Math.round(val * 0x8000) : Math.round(val * 0x7FFF);
    }
    return new Uint8Array(out.buffer);
  }

  function releaseMicHardware(state) {
    if (!state) return;
    if (state.timer) window.clearInterval(state.timer);
    if (state.flushTimer) window.clearInterval(state.flushTimer);
    try {
      if (state.processor) state.processor.disconnect();
      if (state.source) state.source.disconnect();
      if (state.stream) {
        state.stream.getTracks().forEach(function (t) { t.stop(); });
      }
      if (state.audioCtx && state.audioCtx.state !== "closed") {
        state.audioCtx.close();
      }
    } catch (err) {}
  }

  function stopMicCapture() {
    if (!micState) return null;
    var currentMic = micState;
    micState = null;
    currentMic.stopped = true;
    releaseMicHardware(currentMic);
    return currentMic;
  }

  function sendLiveChunk(state, pcmBytes) {
    if (!state || !state.sessionId) return Promise.resolve();
    return fetch("/api/live/chunk?session_id=" + encodeURIComponent(state.sessionId), {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: pcmBytes
    }).then(function (res) {
      if (!res.ok) {
        return res.json().catch(function () { return {}; }).then(function (d) {
          throw new Error((d && d.error) || ("实时音频发送失败 (HTTP " + res.status + ")"));
        });
      }
      return res.json();
    }).then(function (data) {
      if (data && Array.isArray(data.events) && typeof window.MeetingUI.handleEvent === "function") {
        for (var i = 0; i < data.events.length; i += 1) {
          window.MeetingUI.handleEvent(adaptEvent(data.events[i]));
        }
      }
    });
  }

  // 真实麦克风实时流式采集：每 ~700ms 边录边落盘、实时上屏斜体→正黑体字幕，点击“结束并整理”进入会后高精度整理
  window.MeetingUI.onStart = function () {
    if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
      emitFatal("当前浏览器或非安全上下文不支持麦克风采集，请通过 http://127.0.0.1:8736 访问或使用导入录音");
      return;
    }

    rendering = true;
    try {
      document.body.classList.remove("is-diarized");
      if (typeof window.MeetingUI.reset === "function") {
        window.MeetingUI.reset({ keepFile: true });
      }
    } finally {
      rendering = false;
    }

    var liveStartUrl = "/api/live/start";
    try {
      var silence = new URLSearchParams(location.search).get("silence") || "";
      if (/^\d+$/.test(silence)) {
        liveStartUrl += "?silence=" + silence;
      }
    } catch (err) {}

    var capturedStream = null;
    navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
    }).then(function (stream) {
      capturedStream = stream;
      return waitForServerIdle(Date.now() + 60000);
    }).then(function () {
      return fetch(liveStartUrl, { method: "POST" });
    }).then(function (res) {
      return res.json().then(function (data) {
        if (!res.ok || !data.ok || !data.session_id) {
          throw new Error((data && data.error) || ("启动实时会议失败 (HTTP " + res.status + ")"));
        }
        return data.session_id;
      });
    }).then(function (sessionId) {
      var AudioCtx = window.AudioContext || window.webkitAudioContext;
      var audioCtx = new AudioCtx();
      var source = audioCtx.createMediaStreamSource(capturedStream);
      var processor = audioCtx.createScriptProcessor(4096, 1, 1);
      var buffers = [];
      var startedAt = Date.now();
      processor.onaudioprocess = function (ev) {
        var chan = ev.inputBuffer.getChannelData(0);
        buffers.push(new Float32Array(chan));
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);

      var state = {
        sessionId: sessionId,
        stream: capturedStream,
        audioCtx: audioCtx,
        source: source,
        processor: processor,
        buffers: buffers,
        sampleRate: audioCtx.sampleRate || 48000,
        stopped: false,
        sendChain: Promise.resolve(),
        timer: null,
        flushTimer: null
      };

      state.timer = window.setInterval(function () {
        var sec = Math.floor((Date.now() - startedAt) / 1000);
        var mm = String(Math.floor(sec / 60)).padStart(2, "0");
        var ss = String(sec % 60).padStart(2, "0");
        var stateEl = document.getElementById("state-text");
        if (stateEl) stateEl.textContent = "录音中 " + mm + ":" + ss;
      }, 500);

      state.flushTimer = window.setInterval(function () {
        if (state.stopped) return;
        var batch = state.buffers.splice(0, state.buffers.length);
        var pcm = encodePcm16kInt16(batch, state.sampleRate);
        // 即使静音也要定期带走已完成的 llm_confirmed 润色事件
        state.sendChain = state.sendChain.then(function () {
          if (state.stopped) return;
          return sendLiveChunk(state, pcm);
        }).catch(function (err) {
          if (!state.stopped) {
            emitFatal(String(err));
            stopMicCapture();
          }
        });
      }, 700);

      micState = state;
      var endBtn = document.getElementById("end-meeting");
      if (endBtn) endBtn.disabled = false;
      var hint = document.getElementById("transcript-hint");
      if (hint) {
        hint.textContent = "麦克风已实时连通：说话后字幕会自动上屏并逐句润色，音频实时落盘防丢失";
      }
    }).catch(function (err) {
      if (capturedStream) {
        capturedStream.getTracks().forEach(function (t) { t.stop(); });
      }
      var startBtn = document.getElementById("start-meeting");
      if (startBtn) { startBtn.hidden = false; startBtn.disabled = false; }
      emitFatal("麦克风启动失败: " + (err && err.message ? err.message : String(err)));
    });
  };

  // 音频文件模拟：支持第二个参数 options = { fast: true/false }
  window.MeetingUI.onAudioSimulation = function (file, options) {
    var hadActiveController = !!controller;
    if (controller) {
      controller.abort();
      controller = null;
    }

    rendering = true;
    try {
      document.body.classList.remove("is-diarized");
      if (typeof window.MeetingUI.reset === "function") {
        window.MeetingUI.reset({ keepFile: true });
      }
    } finally {
      rendering = false;
    }

    controller = new AbortController();
    var current = controller;
    finished = false;

    var fast = "0";
    if (options && typeof options.fast !== "undefined") {
      fast = options.fast ? "1" : "0";
    } else {
      try {
        var requested = new URLSearchParams(location.search).get("fast");
        if (requested) {
          fast = requested;
        }
      } catch (err) {
        fast = "0";
      }
    }

    var modeLabel = (fast === "1" || fast === "true") ? "正在整理录音" : "实时演示中";
    var stateTextEl = document.getElementById("state-text");
    var stateBadgeEl = document.getElementById("session-state");
    if (stateTextEl) stateTextEl.textContent = modeLabel;
    if (stateBadgeEl) {
      stateBadgeEl.classList.remove("is-done", "is-error", "is-live");
      stateBadgeEl.classList.add("is-busy");
    }
    var startBtn = document.getElementById("start-meeting");
    var endBtn = document.getElementById("end-meeting");
    var simBtn = document.getElementById("simulate-audio");
    var fastBtn = document.getElementById("simulate-audio-fast");
    if (startBtn) startBtn.hidden = true;
    if (endBtn) { endBtn.hidden = false; endBtn.disabled = false; }
    if (simBtn) simBtn.disabled = true;
    if (fastBtn) fastBtn.disabled = true;

    var form = new FormData();
    form.append("file", file, file.name);

    var url = "/api/stream_meeting?name=" + encodeURIComponent(file.name) +
              "&fast=" + encodeURIComponent(fast);
    // 导入音频默认直通高精度离线管线（Pyannote 全覆盖 + SEACO 连续转写），比走流式草稿快 4~6 倍且少漏 10+ 点字错率
    if (fast === "1" || fast === "true") {
      url += "&direct=1";
    }
    // 允许用 ?silence=800 调低句尾静音阈值（定稿更快，句数更多）
    var silence = "";
    try {
      silence = new URLSearchParams(location.search).get("silence") || "";
    } catch (err) {
      silence = "";
    }
    if (/^\d+$/.test(silence)) {
      url += "&silence=" + silence;
    }

    var preReset = hadActiveController ? requestBackendReset().catch(function () {}) : Promise.resolve();

    preReset.then(function () {
      return waitForServerIdle(Date.now() + 120000);
    }).then(function () {
      if (current.signal.aborted) {
        return null;
      }
      return fetch(url, {
        method: "POST",
        body: form,
        signal: current.signal
      });
    }).then(function (response) {
      if (!response) {
        return;
      }
      if (!response.ok) {
        return response.json().catch(function () {
          return {};
        }).then(function (result) {
          throw new Error(result.error || ("HTTP " + response.status));
        });
      }
      if (!response.body || typeof response.body.getReader !== "function") {
        emitFatal("当前浏览器不支持流式响应");
        return;
      }
      return pump(response.body.getReader(), new TextDecoder("utf-8"), "");
    }).catch(function (err) {
      if (err && err.name === "AbortError") {
        return;
      }
      emitFatal(String(err));
    }).finally(function () {
      if (controller === current) {
        controller = null;
      }
      var audioInput = document.getElementById("audio-file");
      var hasFile = !!(audioInput && audioInput.files && audioInput.files.length);
      if (startBtn) { startBtn.hidden = false; startBtn.disabled = false; }
      if (endBtn) { endBtn.hidden = true; endBtn.disabled = true; }
      if (simBtn) simBtn.disabled = !hasFile;
      if (fastBtn) { fastBtn.disabled = !hasFile; fastBtn.hidden = !hasFile; }
    });
  };

  if (typeof window.MeetingUI.onReset !== "function") {
    window.MeetingUI.onReset = function () {
      if (rendering) {
        return;
      }
      if (micState) {
        stopMicCapture();
      }
      document.body.classList.remove("is-diarized");
      var pending = requestBackendReset();
      if (controller) {
        controller.abort();
        controller = null;
      }
      finished = false;
      return pending;
    };
  }

  window.addEventListener("pagehide", function () {
    if (micState) {
      stopMicCapture();
    }
    if (!controller || finished) {
      return;
    }
    controller.abort();
    controller = null;
  });

  if (typeof window.MeetingUI.onEnd !== "function") {
    window.MeetingUI.onEnd = function () {
      if (micState) {
        var finishedMic = stopMicCapture();
        var tailBatch = finishedMic.buffers.splice(0, finishedMic.buffers.length);
        var tailPcm = encodePcm16kInt16(tailBatch, finishedMic.sampleRate);
        controller = new AbortController();
        var current = controller;
        finished = false;

        finishedMic.sendChain
          .catch(function () {})
          .then(function () {
            if (tailPcm.length > 0) {
              return sendLiveChunk(finishedMic, tailPcm).catch(function () {});
            }
          })
          .then(function () {
            return fetch("/api/live/finish?session_id=" + encodeURIComponent(finishedMic.sessionId), {
              method: "POST",
              signal: current.signal
            });
          })
          .then(function (response) {
            if (!response.ok) {
              return response.json().catch(function () { return {}; }).then(function (d) {
                throw new Error((d && d.error) || ("结束录音整理失败 (HTTP " + response.status + ")"));
              });
            }
            if (!response.body || typeof response.body.getReader !== "function") {
              emitFatal("当前浏览器不支持流式响应");
              return;
            }
            return pump(response.body.getReader(), new TextDecoder("utf-8"), "");
          })
          .catch(function (err) {
            if (err && err.name === "AbortError") return;
            emitFatal(String(err));
          })
          .finally(function () {
            if (controller === current) controller = null;
          });
        return;
      }
      if (!controller) {
        var endButton = document.getElementById("end-meeting");
        if (endButton) {
          endButton.disabled = false;
        }
        var hint = document.getElementById("transcript-hint");
        if (hint) {
          hint.textContent = "当前没有进行中的会议";
        }
        if (typeof window.MeetingUI.handleEvent === "function") {
          window.MeetingUI.handleEvent({
            t: "error",
            msg: "当前没有进行中的会议",
            fatal: false
          });
        }
        return;
      }
      if (typeof window.MeetingUI.handleEvent === "function") {
        window.MeetingUI.handleEvent({
          t: "error",
          msg: "已请求结束，正在收尾并回填说话人...",
          fatal: false
        });
      }
      fetch("/api/stream_meeting/cancel", { method: "POST" }).catch(function (err) {
        emitFatal(String(err));
      });
    };
  }

  window.MeetingUI.onRenameSpeaker = function (sessionId, spkLabel, newName) {
    if (!sessionId || !spkLabel) return Promise.resolve();
    var payload = { speaker_names: {} };
    payload.speaker_names[spkLabel] = newName || "";
    return fetch("/api/sessions/" + encodeURIComponent(sessionId) + "/speakers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (res) {
      return res.json().then(function (data) {
        if (!res.ok || !data.ok) {
          throw new Error((data && data.error) || ("保存发言人失败 (HTTP " + res.status + ")"));
        }
        return data;
      });
    }).catch(function (err) {
      if (typeof window.MeetingUI.handleEvent === "function") {
        window.MeetingUI.handleEvent({ t: "error", msg: String(err), fatal: false });
      }
    });
  };

  window.MeetingUI.onGenerateSummary = function (sessionId) {
    if (!sessionId) return Promise.resolve();
    return fetch("/api/sessions/" + encodeURIComponent(sessionId) + "/summary", {
      method: "POST"
    }).then(function (res) {
      return res.json().then(function (data) {
        if (!res.ok || !data.ok) {
          throw new Error((data && data.error) || ("生成会议纪要失败 (HTTP " + res.status + ")"));
        }
        if (typeof window.MeetingUI.handleEvent === "function") {
          window.MeetingUI.handleEvent({
            t: "meeting_summary",
            session_id: sessionId,
            summary: data.summary,
            speaker_names: data.speaker_names || {}
          });
        }
        refreshHistorySessions(sessionId);
        return data;
      });
    }).catch(function (err) {
      if (typeof window.MeetingUI.handleEvent === "function") {
        window.MeetingUI.handleEvent({ t: "error", msg: String(err), fatal: false });
      }
    });
  };

  window.MeetingUI.onDeleteSession = function (sessionId) {
    if (!sessionId) return Promise.resolve();
    return fetch("/api/sessions/" + encodeURIComponent(sessionId), {
      method: "DELETE"
    }).then(function (res) {
      return res.json().then(function (data) {
        if (!res.ok || !data.ok) {
          throw new Error((data && data.error) || ("删除失败 (HTTP " + res.status + ")"));
        }
        if (typeof window.MeetingUI.reset === "function") {
          window.MeetingUI.reset({ keepFile: true });
        }
        document.body.classList.remove("is-diarized");
        refreshHistorySessions("");
        if (typeof window.MeetingUI.handleEvent === "function") {
          window.MeetingUI.handleEvent({ t: "error", msg: "已删除该历史会议记录", fatal: false });
        }
        return data;
      });
    }).catch(function (err) {
      if (typeof window.MeetingUI.handleEvent === "function") {
        window.MeetingUI.handleEvent({ t: "error", msg: String(err), fatal: false });
      }
    });
  };

  // [C5] 绑定历史会话选择事件
  var historySelect = document.getElementById("history-session-select");
  if (historySelect) {
    historySelect.addEventListener("change", function () {
      var sid = historySelect.value;
      if (typeof window.MeetingUI.syncDeleteButton === "function") {
        window.MeetingUI.syncDeleteButton();
      }
      if (!sid) return;
      fetch("/api/sessions/" + encodeURIComponent(sid), { cache: "no-store" })
        .then(function (res) {
          if (!res.ok) throw new Error("加载历史会话失败 (HTTP " + res.status + ")");
          return res.json();
        })
        .then(function (sessionObj) {
          var turns = sessionObj.turns || [];
          var speakers = {};
          for (var i = 0; i < turns.length; i += 1) {
            if (turns[i].spk_label) speakers[turns[i].spk_label] = true;
          }
          renderTurns({
            session_id: sid,
            n_clusters: Object.keys(speakers).length || 1,
            turns: turns,
            speaker_names: sessionObj.speaker_names || {},
            summary: sessionObj.summary || null,
            has_audio: sessionObj.has_audio !== false
          });
        })
        .catch(function (err) {
          emitFatal(String(err));
        });
    });
    refreshHistorySessions();
  }
})();
