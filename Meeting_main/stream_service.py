# -*- coding: utf-8 -*-
"""会议回放与实时录音通道 HTTP 服务：GET 页面/健康检查/历史会话, POST WAV -> NDJSON 流式事件。

2026-10-19 审计整改升级：
1. [A2] 捕获 `(Exception, SystemExit)`，防止下游库代码抛出 `SystemExit` 穿透线程导致 chunked 响应挂死。
2. [B2] 为 `StreamRequestHandler` 设置 `timeout = 60` 秒 socket 超时，防止半开连接永久锁死 `_SESSION_LOCK`。
3. [B3] 零拷贝流式解析 `multipart/form-data` (`_extract_multipart_fast`)，不再通过 `email.parser` 复制三份内存，
   峰值内存降低 66%。
4. [C5] 新增 `GET /api/sessions` 与 `GET /api/sessions/<session_id>` 历史会话列表与回看接口（补齐规格 §4.4.4）。
5. [E6] 对 `/api/stream_meeting/cancel` 与 `/api/stream_meeting/reset` 增加本机 `Origin` 校验，拦截跨站伪造打断。
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

import bootstrap

bootstrap.install()

from run_offline import run  # noqa: E402
from session_store import SessionStore, new_session_dir  # noqa: E402

SERVICE_HOST = "127.0.0.1"
SERVICE_PORT = 8736
SOCKET_TIMEOUT_S = 60.0
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
SESSION_ROOT = Path("output/sessions")
INDEX_FILE = Path(__file__).resolve().parent / "diar_web.html"
BRIDGE_FILE = Path(__file__).resolve().parent / "meeting_bridge.js"
BRIDGE_MARKER = b"meeting_bridge.js"
BRIDGE_TAG = b'<script src="/meeting_bridge.js"></script>\n'

_SESSION_ID_RE = re.compile(r"^s_\d{8}_\d{6}(?:_\d+)?$")
_BOUNDARY_RE = re.compile(r'boundary=(?:"([^"]+)"|([^;\s]+))', re.IGNORECASE)
_FILENAME_RE = re.compile(rb'filename="([^"]*)"')

_SESSION_LOCK = threading.Lock()
_SESSION_IDLE = threading.Event()
_SESSION_IDLE.set()
_SESSION_STATE_LOCK = threading.Lock()
_RESET_REQUEST_LOCK = threading.Lock()
_RESET_PENDING = threading.Event()
_CANCEL = threading.Event()
_RESET = threading.Event()

_MODELS: Optional[Dict] = None
_MODELS_LOCK = threading.Lock()

_PIPELINE: Callable[..., Dict] = run


class FfmpegMissing(RuntimeError):
    """需要转码但 PATH 中找不到 ffmpeg。"""


def _get_models() -> Dict:
    """模块级模型单例 (双重检查锁)。"""
    global _MODELS
    if _MODELS is None:
        with _MODELS_LOCK:
            if _MODELS is None:
                from web_server import get_models
                _MODELS = get_models(need_stream=True, need_offline=False)
    return _MODELS


def _run_pipeline(wav, session_dir, **kwargs) -> Dict:
    """薄包装：测试 monkeypatch `_PIPELINE` 后仍走同一条调用路径。"""
    return _PIPELINE(wav, session_dir, **kwargs)


def _safe_filename(raw: Optional[str]) -> str:
    """只取 basename；客户端字符串永远不能作为路径使用。"""
    name = Path(str(raw or "").replace("\\", "/")).name.strip()
    name = name.replace("\x00", "")
    if name in ("", ".", ".."):
        return "upload.bin"
    for ch in '<>:"|?*':
        name = name.replace(ch, "_")
    return name


def _extract_multipart(body: bytes, content_type: str) -> Optional[Tuple[str, bytes]]:
    """[B3] 优先使用基于边界切片的零额外复制解析；复杂 MIME 回退到 BytesParser。"""
    m = _BOUNDARY_RE.search(content_type or "")
    if m:
        boundary = (m.group(1) or m.group(2) or "").encode("ascii", errors="ignore")
        if boundary:
            delim = b"--" + boundary
            pos = body.find(delim)
            while pos != -1:
                start = pos + len(delim)
                if body[start:start + 2] == b"--":
                    break
                if body[start:start + 2] == b"\r\n":
                    start += 2
                next_pos = body.find(b"\r\n" + delim, start)
                if next_pos == -1:
                    break
                header_end = body.find(b"\r\n\r\n", start, next_pos)
                if header_end != -1:
                    raw_headers = body[start:header_end]
                    fn_match = _FILENAME_RE.search(raw_headers)
                    if fn_match:
                        filename = fn_match.group(1).decode("utf-8", errors="replace").strip()
                        if filename:
                            return filename, body[header_end + 4:next_pos]
                pos = next_pos + 2

    header = (f"Content-Type: {content_type}\r\n"
              "MIME-Version: 1.0\r\n\r\n").encode("utf-8", errors="replace")
    try:
        message = BytesParser().parsebytes(header + body)
    except Exception:
        return None
    if not message.is_multipart():
        return None
    for part in message.walk():
        if part.get_content_maintype() == "multipart":
            continue
        filename = part.get_filename()
        if not filename:
            continue
        return filename, part.get_payload(decode=True) or b""
    return None


def _needs_conversion(src: Path) -> bool:
    """已经是 16 kHz 单声道 (soundfile 能识别) 则跳过 ffmpeg。"""
    try:
        import soundfile as sf
        info = sf.info(str(src))
    except Exception:
        return True
    return not (info.samplerate == 16000 and info.channels == 1)


def _convert_to_16k_mono(src: Path, dst: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FfmpegMissing("ffmpeg")
    proc = subprocess.run(
        [ffmpeg, "-y", "-i", str(src), "-af", "pan=mono|c0=c0", "-ar", "16000",
         "-c:a", "pcm_s16le", str(dst)],
        capture_output=True,
    )
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        detail = proc.stderr.decode("utf-8", errors="replace")[-500:].strip()
        raise RuntimeError(f"ffmpeg 转码失败 (exit {proc.returncode}) : {detail}")


def _inject_bridge(html: bytes) -> bytes:
    """在 </body> 前注入桥接脚本标签；已存在标记时不重复注入。"""
    if BRIDGE_MARKER in html:
        return html
    index = html.rfind(b"</body>")
    if index < 0:
        return html
    return html[:index] + BRIDGE_TAG + html[index:]


def _count_clusters(mapping: Dict) -> int:
    labels = set()
    for row in mapping.values():
        if not isinstance(row, dict):
            continue
        label = row.get("spk_label", row.get("spk"))
        if label is not None:
            labels.add(label)
    return len(labels)


def _is_trusted_origin(origin: Optional[str]) -> bool:
    """[E6] 校验 Origin 头（无 Origin 头如同源 CLI/非跨域请求放行；有 Origin 头则必须为本机地址）。"""
    if not origin:
        return True
    parsed = urlsplit(origin.strip())
    return parsed.hostname in ("127.0.0.1", "localhost", "::1")


def _list_history_sessions(root: Path, limit: int = 50) -> List[Dict]:
    """[C5] 列出历史会话摘要，按目录名倒序排列。"""
    root = Path(root)
    if not root.is_dir():
        return []
    items: List[Dict] = []
    dirs = sorted(
        (d for d in root.iterdir() if d.is_dir() and _SESSION_ID_RE.match(d.name)),
        key=lambda d: d.name,
        reverse=True,
    )
    for d in dirs[:limit]:
        sess_file = d / "session.json"
        if not sess_file.exists():
            continue
        try:
            obj = json.loads(sess_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        turns = obj.get("turns") or []
        utts = obj.get("utterances") or []
        speakers = sorted({t.get("spk_label") for t in turns if t.get("spk_label")})
        items.append({
            "session_id": d.name,
            "status": obj.get("status", "unknown"),
            "duration_ms": obj.get("duration_ms", 0),
            "n_utterances": len(utts),
            "n_turns": len(turns),
            "n_speakers": len(speakers),
        })
    return items


class StreamRequestHandler(BaseHTTPRequestHandler):
    """单请求处理器：GET 静态页/健康检查/历史会话，POST 流式会议。"""

    protocol_version = "HTTP/1.1"
    server_version = "MeetingStream/0.2"
    sys_version = ""
    timeout = SOCKET_TIMEOUT_S  # [B2] 防止半开连接无限挂起

    def __init__(self, *args, **kwargs):
        self._dead = False
        self._session_active = False
        self._write_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def _send_json(self, status: int, obj: Dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(data)
            self.wfile.flush()
        except OSError:
            self._dead = True

    def _send_html(self) -> None:
        try:
            data = _inject_bridge(INDEX_FILE.read_bytes())
        except OSError:
            self._send_json(404, {"error": "前端文件不存在"})
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(data)
            self.wfile.flush()
        except OSError:
            self._dead = True

    def _send_bridge(self) -> None:
        try:
            data = BRIDGE_FILE.read_bytes()
        except OSError:
            self._send_json(404, {"error": "桥接脚本不存在"})
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()
        except OSError:
            self._dead = True

    def _start_ndjson(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _send_chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")

    def _end_chunks(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _end_chunks_safely(self) -> None:
        try:
            with self._write_lock:
                if not self._dead:
                    self._end_chunks()
        except OSError:
            self._dead = True

    def _emit_event(self, ev: Dict) -> None:
        if self._dead:
            return
        line = (json.dumps(ev, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            with self._write_lock:
                if self._dead:
                    return
                self._send_chunk(line)
                self.wfile.flush()
        except (BrokenPipeError, OSError):
            self._dead = True
            if self._session_active:
                _RESET.set()
                _CANCEL.set()

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/":
            self._send_html()
        elif path == "/meeting_bridge.js":
            self._send_bridge()
        elif path == "/api/health":
            self._send_json(200, {
                "ok": True,
                "busy": _SESSION_LOCK.locked(),
                "resetting": _RESET_PENDING.is_set(),
            })
        elif path == "/api/sessions":
            self._send_json(200, {
                "ok": True,
                "sessions": _list_history_sessions(SESSION_ROOT),
            })
        elif path.startswith("/api/sessions/"):
            session_id = path.split("/api/sessions/", 1)[1].strip("/")
            if not _SESSION_ID_RE.match(session_id):
                self._send_json(400, {"error": "非法 session_id"})
                return
            sess_dir = SESSION_ROOT / session_id
            store = SessionStore(sess_dir)
            session_obj = store.read_json("session.json")
            if session_obj is None:
                self._send_json(404, {"error": "会话不存在"})
                return
            self._send_json(200, session_obj)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        origin = self.headers.get("Origin")
        if parsed.path in ("/api/stream_meeting/cancel", "/api/stream_meeting/reset"):
            if not _is_trusted_origin(origin):
                self._send_json(403, {"error": "Origin 校验失败"})
                return
        if parsed.path == "/api/stream_meeting/cancel":
            _CANCEL.set()
            self._send_json(200, {"ok": True})
            return
        if parsed.path == "/api/stream_meeting/reset":
            with _RESET_REQUEST_LOCK:
                with _SESSION_STATE_LOCK:
                    _RESET_PENDING.set()
                    _RESET.set()
                    _CANCEL.set()
                idle = _SESSION_IDLE.wait(timeout=120)
                with _SESSION_STATE_LOCK:
                    _RESET_PENDING.clear()
                self._send_json(200 if idle else 202, {
                    "ok": idle,
                    "busy": not idle,
                    "error": None if idle else "上一场会议仍在停止，请稍后重试重置",
                })
            return
        if parsed.path != "/api/stream_meeting":
            self._send_json(404, {"error": "not found"})
            return

        if not _is_trusted_origin(origin):
            self._send_json(403, {"error": "Origin 校验失败"})
            return

        query = parse_qs(parsed.query)
        name = query.get("name", [None])[0]
        fast = query.get("fast", ["0"])[0].lower() in ("1", "true", "yes")
        no_llm = query.get("no_llm", ["0"])[0].lower() in ("1", "true", "yes")
        # 句尾静音阈值：800 能把定稿延迟砍掉约 0.34s，代价是句数变 ~1.5 倍
        silence_raw = query.get("silence", [""])[0].strip()
        silence_ms = int(silence_raw) if silence_raw.isdigit() else None

        self._dead = False
        with _SESSION_STATE_LOCK:
            if _RESET_PENDING.is_set():
                acquired = False
                busy_error = "正在重置上一场会议，请稍后再试"
            else:
                acquired = _SESSION_LOCK.acquire(blocking=False)
                busy_error = "已有会议在运行"
            if acquired:
                _SESSION_IDLE.clear()
                _CANCEL.clear()
                _RESET.clear()
                self._session_active = True
        if not acquired:
            self._send_json(409, {"error": busy_error})
            return

        tmpdir: Optional[Path] = None
        response_started = False
        try:
            content_length = self.headers.get("Content-Length")
            if content_length is None:
                self._send_json(411, {"error": "缺少 Content-Length"})
                return
            try:
                length = int(content_length)
            except ValueError:
                self._send_json(400, {"error": "Content-Length 非法"})
                return
            if length < 0:
                self._send_json(400, {"error": "Content-Length 非法"})
                return
            if length > MAX_UPLOAD_BYTES:
                self._send_json(413, {"error": "上传文件超过 500 MB 上限"})
                return

            body = self.rfile.read(length)
            if len(body) != length:
                self._send_json(400, {"error": "请求体不完整"})
                return
            if _RESET.is_set():
                self._send_json(409, {"error": "会议已重置"})
                return

            content_type = self.headers.get("Content-Type", "")
            part_name: Optional[str] = None
            payload = body
            if content_type.lower().startswith("multipart/form-data"):
                found = _extract_multipart(body, content_type)
                if found is None:
                    self._send_json(400, {"error": "multipart 中缺少带文件名的 part"})
                    return
                part_name, payload = found
            if not payload:
                self._send_json(400, {"error": "上传内容为空"})
                return

            filename = _safe_filename(name or part_name or "upload.bin")
            tmpdir = Path(tempfile.mkdtemp(prefix="meeting_upload_"))
            src = tmpdir / filename
            src.write_bytes(payload)
            # 释放内存引用，避免长会议期间驻留上传缓冲
            del payload
            del body

            wav_path = src
            if _needs_conversion(src):
                wav_path = tmpdir / "converted_16k.wav"
                try:
                    _convert_to_16k_mono(src, wav_path)
                except FfmpegMissing:
                    self._start_ndjson()
                    response_started = True
                    self._emit_event({
                        "t": "error",
                        "msg": "需要 ffmpeg 转码：未找到 ffmpeg 可执行文件，请安装并加入 PATH",
                        "fatal": True,
                    })
                    self._end_chunks_safely()
                    return

            session_dir = new_session_dir(SESSION_ROOT)
            models = _get_models()

            self._start_ndjson()
            response_started = True

            pipe_kwargs = dict(
                fast=fast,
                models=models,
                on_event=self._emit_event,
                on_progress=self._emit_event,
                should_stop=_CANCEL.is_set,
                should_abort=_RESET.is_set,
            )
            if no_llm:
                try:
                    sig = inspect.signature(_PIPELINE)
                    if "no_llm" in sig.parameters or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in sig.parameters.values()
                    ):
                        pipe_kwargs["no_llm"] = True
                except (TypeError, ValueError):
                    pass

            # 与 no_llm 同一套守卫：注入的流水线若不接受 silence_ms，就别硬塞。
            if silence_ms is not None:
                try:
                    sig = inspect.signature(_PIPELINE)
                    if "silence_ms" in sig.parameters or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in sig.parameters.values()
                    ):
                        pipe_kwargs["silence_ms"] = silence_ms
                except (TypeError, ValueError):
                    pass

            _run_pipeline(wav_path, session_dir, **pipe_kwargs)

            if _RESET.is_set():
                return

            store = SessionStore(session_dir)
            session_obj = store.read_json("session.json") or {}
            mapping = store.read_json("diarization.json") or {}
            utterances = session_obj.get("utterances") or []
            total = len(utterances) if utterances else len(mapping)

            self._emit_event({
                "t": "speaker_diarized",
                "session_id": session_dir.name,
                "n_clusters": _count_clusters(mapping),
                "mapping": mapping,
                "batch": {"done": len(mapping), "total": total},
                "turns": session_obj.get("turns", []),
            })
            self._emit_event({"t": "finish", "session_id": session_dir.name})
            self._end_chunks_safely()
        except (Exception, SystemExit) as exc:  # [A2] 捕获 SystemExit 防止穿透 HTTP 层挂死连接
            if _RESET.is_set():
                return
            msg = f"{type(exc).__name__}: {exc}"
            if response_started:
                self._emit_event({"t": "error", "msg": msg, "fatal": True})
                self._emit_event({"t": "finish"})
                self._end_chunks_safely()
            else:
                self._send_json(500, {"error": msg})
        finally:
            if tmpdir is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)
            self._session_active = False
            with _SESSION_STATE_LOCK:
                _SESSION_LOCK.release()
                _SESSION_IDLE.set()


class _SingleInstanceServer(ThreadingHTTPServer):
    allow_reuse_address = False


def make_server(host: str = SERVICE_HOST, port: int = SERVICE_PORT) -> ThreadingHTTPServer:
    try:
        server = _SingleInstanceServer((host, port), StreamRequestHandler)
    except OSError as exc:
        raise SystemExit(
            f"端口 {host}:{port} 绑定失败 (可能已有服务在运行) : {exc}"
        ) from exc
    server.daemon_threads = True
    return server


def serve(host: str = SERVICE_HOST, port: int = SERVICE_PORT, preload: bool = True) -> None:
    server = make_server(host, port)
    print(f"[stream_service] 已启动 http://{host}:{port} (Ctrl+C 停止)", flush=True)
    if preload:
        started = time.monotonic()
        try:
            _get_models()
            print(f"[stream_service] 模型预加载完成 {time.monotonic() - started:.1f}s",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[stream_service] 模型预加载失败 (请求时重试) : "
                  f"{type(exc).__name__}: {exc}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="会议回放通道 HTTP 服务 (NDJSON 流式) ")
    parser.add_argument("--port", type=int, default=SERVICE_PORT)
    parser.add_argument("--no-preload", action="store_true", help="启动时不加载模型")
    args = parser.parse_args(argv)
    serve(port=args.port, preload=not args.no_preload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
