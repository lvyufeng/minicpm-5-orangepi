#!/usr/bin/env python3
"""Local MiniCPM5-1B demo wrapper for the Orange Pi Ascend runtime.

The Python side loads only the Hugging Face tokenizer, formats chat prompts,
and decodes streamed token IDs from a persistent local C++ backend.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal

os.environ.setdefault("USE_TORCH", "0")
import subprocess
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

DEMO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DEMO_ROOT.parent
BUILD_DIR = Path(os.environ.get("BUILD_DIR", REPO_ROOT / "build"))
BACKEND_BIN = Path(os.environ.get("MINICPM5_SERVER_BIN", BUILD_DIR / "minicpm5_server"))
WEIGHTS = os.environ.get("WEIGHTS", os.environ.get("MINICPM5_MODEL_PATH", str(REPO_ROOT / "models" / "MiniCPM5-1B")))
TOKENIZER_PATH = os.environ.get("TOKENIZER", os.environ.get("MINICPM5_TOKENIZER", WEIGHTS if Path(WEIGHTS).exists() else "openbmb/MiniCPM5-1B"))
MAX_SEQ = int(os.environ.get("MAX_SEQ", "4096"))
MAX_NEW = int(os.environ.get("MAX_NEW", "256"))
DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))

app = FastAPI(title="MiniCPM5-1B OrangePi Demo")

if (DEMO_ROOT / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DEMO_ROOT / "assets"), name="assets")

_tokenizer = None
_current_process: subprocess.Popen[str] | None = None
_current_stderr: list[str] = []
_backend_ready = False
_process_lock = threading.Lock()
_generation_lock = asyncio.Lock()


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[list[Optional[str]]] = Field(default_factory=list)
    thinking_mode: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: Optional[int] = None


def tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    return _tokenizer


def organize_messages(message: str, history: Optional[list[list[Optional[str]]]] = None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for turn in history or []:
        if not turn:
            continue
        user = turn[0] if len(turn) > 0 else None
        assistant = turn[1] if len(turn) > 1 else None
        if user:
            messages.append({"role": "user", "content": str(user)})
        if assistant:
            messages.append({"role": "assistant", "content": str(assistant)})
    messages.append({"role": "user", "content": message})
    return messages


def encode_prompt(req: PredictRequest) -> list[int]:
    tok = tokenizer()
    messages = organize_messages(req.message, req.history)
    try:
        ids = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=req.thinking_mode,
            tokenize=True,
        )
    except TypeError:
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = tok.encode(text, add_special_tokens=False)
    return [int(x) for x in ids]


def decode_ids(ids: list[int]) -> str:
    if not ids:
        return ""
    text = tokenizer().decode(ids, skip_special_tokens=False)
    return text.replace("<|im_end|>", "")


def ensure_backend_ready() -> None:
    if not BACKEND_BIN.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Backend binary not found: {BACKEND_BIN}. Run scripts/run/run_demo.sh first.",
        )


def _clear_backend_process(proc: subprocess.Popen[str] | None = None) -> None:
    global _current_process, _backend_ready
    with _process_lock:
        if proc is None or _current_process is proc:
            _current_process = None
            _backend_ready = False


def _is_backend_alive(proc: subprocess.Popen[str] | None) -> bool:
    return proc is not None and proc.poll() is None


def _start_backend_process() -> subprocess.Popen[str]:
    global _current_process, _current_stderr, _backend_ready
    cmd = [
        str(BACKEND_BIN),
        "--weights",
        WEIGHTS,
        "--max-seq",
        str(MAX_SEQ),
        "--device-id",
        str(DEVICE_ID),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    _current_process = proc
    _current_stderr = []
    _backend_ready = False
    stderr_thread = threading.Thread(target=drain_stderr, args=(proc, _current_stderr), daemon=True)
    stderr_thread.start()
    return proc


def ensure_backend_process() -> subprocess.Popen[str]:
    global _current_process
    with _process_lock:
        if _is_backend_alive(_current_process):
            return _current_process  # type: ignore[return-value]
        _current_process = None
        return _start_backend_process()


def _mark_backend_ready_from_stderr(line: str) -> None:
    global _backend_ready
    if "minicpm5_server ready" in line:
        _backend_ready = True


def drain_stderr(proc: subprocess.Popen[str], sink: list[str]) -> None:
    if proc.stderr is None:
        return
    for line in proc.stderr:
        sink.append(line)
        _mark_backend_ready_from_stderr(line)


def terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=2)
    except Exception:
        proc.kill()
    finally:
        _clear_backend_process(proc)


async def run_backend_stream(input_ids: list[int], max_new: int, request: Request) -> AsyncIterator[str]:
    global _current_stderr
    ensure_backend_ready()
    if len(input_ids) + max_new > MAX_SEQ:
        raise HTTPException(status_code=400, detail="Prompt plus max_new_tokens exceeds MAX_SEQ")

    proc = ensure_backend_process()
    generated: list[int] = []
    request_line = f"REQUEST {max_new} {','.join(str(x) for x in input_ids)}\n"

    try:
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("backend process is missing pipes")
        proc.stdin.write(request_line)
        proc.stdin.flush()

        while True:
            if await request.is_disconnected():
                terminate_process(proc)
                break

            line = await asyncio.to_thread(proc.stdout.readline)
            if line == "":
                if await request.is_disconnected():
                    break
                stderr = "".join(_current_stderr)
                rc = proc.poll()
                _clear_backend_process(proc)
                raise RuntimeError(stderr or f"backend exited with {rc if rc is not None else 'EOF'}")

            line = line.strip()
            if not line:
                continue
            if line == "DONE":
                yield json.dumps({"done": True, "text": decode_ids(generated)}, ensure_ascii=False) + "\n"
                break
            if line.startswith("ERROR "):
                raise RuntimeError(line[6:])

            try:
                token_id = int(line)
            except ValueError:
                continue

            generated.append(token_id)
            yield json.dumps({"text": decode_ids(generated), "token_id": token_id}, ensure_ascii=False) + "\n"
    finally:
        if proc.poll() is not None:
            _clear_backend_process(proc)


@app.on_event("startup")
async def startup_backend() -> None:
    if not BACKEND_BIN.exists():
        return
    def start() -> None:
        try:
            ensure_backend_process()
        except Exception as exc:
            _current_stderr.append(f"backend startup failed: {exc}\n")
    threading.Thread(target=start, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    index_path = DEMO_ROOT / "index.html"
    if not index_path.exists():
        return "<h1>MiniCPM5 Demo</h1><p>index.html missing.</p>"
    return index_path.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, Any]:
    proc = _current_process
    return {
        "ok": True,
        "backend": str(BACKEND_BIN),
        "backend_exists": BACKEND_BIN.exists(),
        "backend_alive": _is_backend_alive(proc),
        "backend_ready": _backend_ready,
        "backend_pid": proc.pid if _is_backend_alive(proc) else None,
        "weights": WEIGHTS,
        "tokenizer": TOKENIZER_PATH,
        "max_seq": MAX_SEQ,
        "max_new_default": MAX_NEW,
        "greedy_only": True,
    }


@app.post("/predict")
async def predict(req: PredictRequest, request: Request) -> StreamingResponse:
    # The local C++ backend is greedy-only today. Keep temperature/top_p in the
    # API for Hugging Face Space compatibility, but they are intentionally not
    # forwarded until sampling exists in the C++ runtime.
    async def stream() -> AsyncIterator[str]:
        async with _generation_lock:
            input_ids = encode_prompt(req)
            max_new = req.max_new_tokens or MAX_NEW
            yield json.dumps({"meta": {"prompt_tokens": len(input_ids), "greedy_only": True}}, ensure_ascii=False) + "\n"
        try:
            async for item in run_backend_stream(input_ids, max_new, request):
                yield item
        except Exception as exc:
            yield json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/cancel")
def cancel() -> dict[str, bool]:
    proc = _current_process
    if proc is not None:
        terminate_process(proc)
        return {"cancelled": True}
    return {"cancelled": False}


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)
