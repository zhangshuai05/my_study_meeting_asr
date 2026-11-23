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

  // [C5] 刷新历史会话下拉框
  function refreshHistorySessions(selectedId) {
    var selectEl = document.getElementById("history-session-select");
    if (!selectEl) return;
    fetch("/api/sessions", { cache: "no-store" }).then(function (res) {
      if (!res.ok) return null;
      return res.json();
    }).then(function (data) {
      if (!data || !Array.isArray(data.sessions)) return;
      var currentVal = selectedId || selectEl.value || "";
      selectEl.innerHTML = '<option value="">历史会话 (' + data.sessions.length + ')</option>';
      for (var i = 0; i < data.sessions.length; i += 1) {
        var s = data.sessions[i];
        var opt = document.createElement("option");
        opt.value = s.session_id;
        var durSec = Math.round((Number(s.duration_ms) || 0) / 1000);
        opt.textContent = s.session_id + " (" + durSec + "s · " + (s.n_speakers || 1) + "人 · " + (s.n_turns || 0) + "段)";
        if (s.session_id === currentVal) {
          opt.selected = true;
        }
        selectEl.appendChild(opt);
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
        n_clusters: event.n_clusters,
        mapping: mapping
      });
      ui.handleEvent({ t: "finish" });
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

  // [A1] 将 Float32 PCM 采样数组重采样并编码为 16kHz 单声道 16-bit WAV Blob
  function encodeWav16kMono(buffers, inputSampleRate) {
    var totalLen = 0;
    for (var i = 0; i < buffers.length; i += 1) {
      totalLen += buffers[i].length;
    }
    var merged = new Float32Array(totalLen);
    var offset = 0;
    for (var j = 0; j < buffers.length; j += 1) {
      merged.set(buffers[j], offset);
      offset += buffers[j].length;
    }
    var targetRate = 16000;
    var samples = merged;
    if (inputSampleRate !== targetRate && totalLen > 0) {
      var ratio = inputSampleRate / targetRate;
      var newLen = Math.max(1, Math.round(totalLen / ratio));
      samples = new Float32Array(newLen);
      for (var k = 0; k < newLen; k += 1) {
        var srcIdx = Math.min(totalLen - 1, Math.floor(k * ratio));
        samples[k] = merged[srcIdx];
      }
    }
    var buffer = new ArrayBuffer(44 + samples.length * 2);
    var view = new DataView(buffer);
    function writeAscii(off, str) {
      for (var c = 0; c < str.length; c += 1) {
        view.setUint8(off + c, str.charCodeAt(c));
      }
    }
    writeAscii(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeAscii(8, "WAVE");
    writeAscii(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, targetRate, true);
    view.setUint32(28, targetRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeAscii(36, "data");
    view.setUint32(40, samples.length * 2, true);
    var idx = 44;
    for (var s = 0; s < samples.length; s += 1, idx += 2) {
      var val = Math.max(-1, Math.min(1, samples[s]));
      view.setInt16(idx, val < 0 ? val * 0x8000 : val * 0x7FFF, true);
    }
    return new Blob([buffer], { type: "audio/wav" });
  }

  function stopMicCapture() {
    if (!micState) return null;
    var currentMic = micState;
    micState = null;
    if (currentMic.timer) {
      window.clearInterval(currentMic.timer);
    }
    try {
      if (currentMic.processor) currentMic.processor.disconnect();
      if (currentMic.source) currentMic.source.disconnect();
      if (currentMic.stream) {
        currentMic.stream.getTracks().forEach(function (t) { t.stop(); });
      }
      if (currentMic.audioCtx && currentMic.audioCtx.state !== "closed") {
        currentMic.audioCtx.close();
      }
    } catch (err) {}
    return encodeWav16kMono(currentMic.buffers, currentMic.sampleRate);
  }

  // [A1] 接入麦克风实时采集：点击"开始会议"启动录音，点击"结束会议"自动送入转写与回填管线
  window.MeetingUI.onStart = function () {
    if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
      emitFatal("当前浏览器或非安全上下文不支持麦克风采集，请通过 http://127.0.0.1:8736 访问或使用音频模拟");
      return;
    }
    navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
    }).then(function (stream) {
      var AudioCtx = window.AudioContext || window.webkitAudioContext;
      var audioCtx = new AudioCtx();
      var source = audioCtx.createMediaStreamSource(stream);
      var processor = audioCtx.createScriptProcessor(4096, 1, 1);
      var buffers = [];
      var startedAt = Date.now();
      processor.onaudioprocess = function (ev) {
        var chan = ev.inputBuffer.getChannelData(0);
        buffers.push(new Float32Array(chan));
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);
      var timer = window.setInterval(function () {
        var sec = Math.floor((Date.now() - startedAt) / 1000);
        var hint = document.getElementById("transcript-hint");
        if (hint) {
          hint.textContent = "麦克风正在录音 (" + sec + "s) —— 点击“结束会议”完成录音并开始转写与说话人分离";
        }
      }, 500);
      micState = {
        stream: stream,
        audioCtx: audioCtx,
        source: source,
        processor: processor,
        buffers: buffers,
        sampleRate: audioCtx.sampleRate || 48000,
        timer: timer
      };
      var endBtn = document.getElementById("end-meeting");
      if (endBtn) endBtn.disabled = false;
    }).catch(function (err) {
      var startBtn = document.getElementById("start-meeting");
      if (startBtn) startBtn.disabled = false;
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

    var modeLabel = (fast === "1" || fast === "true") ? "快速模拟中 (全速流式)" : "音频模拟中 (1x 实时)";
    var stateTextEl = document.getElementById("state-text");
    var stateBadgeEl = document.getElementById("session-state");
    if (stateTextEl) stateTextEl.textContent = modeLabel;
    if (stateBadgeEl) {
      stateBadgeEl.classList.remove("is-done", "is-error");
      stateBadgeEl.classList.add("is-live");
    }
    var startBtn = document.getElementById("start-meeting");
    var simBtn = document.getElementById("simulate-audio");
    var fastBtn = document.getElementById("simulate-audio-fast");
    if (startBtn) startBtn.disabled = true;
    if (simBtn) simBtn.disabled = true;
    if (fastBtn) fastBtn.disabled = true;

    var form = new FormData();
    form.append("file", file, file.name);

    var url = "/api/stream_meeting?name=" + encodeURIComponent(file.name) +
              "&fast=" + encodeURIComponent(fast);
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
      if (simBtn) simBtn.disabled = !hasFile;
      if (fastBtn) fastBtn.disabled = !hasFile;
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
        var wavBlob = stopMicCapture();
        if (!wavBlob || wavBlob.size <= 44) {
          emitFatal("未采集到有效麦克风语音");
          return;
        }
        var micFile = new File([wavBlob], "mic_recording.wav", { type: "audio/wav" });
        window.MeetingUI.onAudioSimulation(micFile, { fast: true });
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

  // [C5] 绑定历史会话选择事件
  var historySelect = document.getElementById("history-session-select");
  if (historySelect) {
    historySelect.addEventListener("change", function () {
      var sid = historySelect.value;
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
            turns: turns
          });
        })
        .catch(function (err) {
          emitFatal(String(err));
        });
    });
    refreshHistorySessions();
  }
})();
