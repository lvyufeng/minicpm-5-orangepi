# MiniCPM5-1B on Orange Pi Ascend

This repository contains a local inference runtime for running **MiniCPM5-1B** on Orange Pi boards with Ascend NPU support. It is focused on edge/on-device deployment: a lightweight C++ Ascend backend executes the model, and an optional FastAPI web demo provides a browser chat UI.

## What is included

- **Ascend C++ inference runtime** for MiniCPM5-1B text generation.
- **Custom Ascend kernels / ops** for performance-critical decode paths, including fused RoPE/cache update, attention, RMSNorm, W8A8 decode helpers, SiLU, and logits top-1.
- **Persistent C++ backend** (`minicpm5_server`) that loads weights once and serves multiple generation requests over a simple stdin/stdout streaming protocol.
- **One-shot CLI** (`minicpm5_decode`) for debugging and scripted generation.
- **Web demo** (`app.py` + `index.html`) that uses the Hugging Face tokenizer in Python and streams generated text from the persistent C++ backend.

The current local backend is **greedy-only**. The web UI keeps temperature/top-p controls for compatibility with the upstream MiniCPM5 demo, but those parameters are not applied by the C++ runtime yet.

## Repository layout

```text
src/csrc/include/minicpmv/   C++ runtime headers
src/csrc/lib/                C++ runtime implementation
src/csrc/custom_ops/         Ascend custom op host/kernel code
src/csrc/tools/              CLI tools: minicpm5_decode, minicpm5_server
scripts/                     environment, build, and demo scripts
app.py                       FastAPI wrapper for the web demo
index.html                   browser chat UI
requirements.txt             Python dependencies for the web demo
```

The `models/` directory is intentionally ignored by git. Put local MiniCPM5 weights there, but do not commit model files.

## Requirements

- Orange Pi / Linux environment with Ascend CANN installed.
- Ascend toolkit available at the default path, or set `MINICPM5_ASCEND_TOOLKIT_ROOT` / `MINICPMV_ASCEND_TOOLKIT_ROOT`.
- MiniCPM5-1B weights available locally, typically at:

```text
models/MiniCPM5-1B
```

- Python 3.9+ for the web demo.
- CMake 3.20+ and a C++17 compiler.

## Prepare the environment

The runtime scripts source `scripts/set_env.sh`, which configures the Ascend toolkit libraries and custom-op path:

```bash
source scripts/set_env.sh
```

Custom ops are installed by `scripts/install_custom_ops.sh`. The web demo launcher runs this automatically if the expected custom-op install directory is missing.

## Run one-shot text decode

```bash
WEIGHTS=models/MiniCPM5-1B \
INPUT_IDS=0 \
MAX_NEW=16 \
MAX_SEQ=4096 \
scripts/run/run_minicpm5_decode.sh
```

This builds and runs `minicpm5_decode`, which accepts token IDs and prints generated token IDs.

## Run the web demo

Install Python dependencies once:

```bash
python3 -m pip install -r requirements.txt
```

Start the demo from the repository root:

```bash
scripts/run/run_demo.sh
```

Then open:

```text
http://127.0.0.1:7860
```

or, from another machine on the same LAN:

```text
http://<orange-pi-ip>:7860
```

### Useful demo environment variables

```bash
WEIGHTS=models/MiniCPM5-1B        # model directory or safetensors path
TOKENIZER=models/MiniCPM5-1B      # tokenizer path; defaults to WEIGHTS
HOST=0.0.0.0
PORT=7860
MAX_SEQ=4096
MAX_NEW=256
DEVICE_ID=0
scripts/run/run_demo.sh
```

### Demo startup behavior

`run_demo.sh`:

1. checks/installs custom Ascend ops if needed;
2. sources `scripts/set_env.sh`;
3. builds `minicpm5_server`;
4. starts `uvicorn app:app`.

The FastAPI app prewarms the persistent C++ backend after startup. Wait until the UI status badge shows:

```text
ready · greedy backend
```

before measuring first-token latency. The first backend load can take much longer than later requests because weights and runtime state are initialized once.

## Persistent backend protocol

`minicpm5_server` is a long-running C++ process. It loads weights once, then accepts newline-delimited requests on stdin:

```text
REQUEST <max_new> <comma_separated_input_ids>
```

It streams generated token IDs on stdout, one per line, ending with:

```text
DONE
```

On request errors it prints:

```text
ERROR <message>
```

Example:

```bash
source scripts/set_env.sh
printf 'REQUEST 8 0\n' | build/minicpm5_server --weights models/MiniCPM5-1B --max-seq 4096 --device-id 0
```

## Build targets

```bash
cmake -S . -B build -DMINICPM5_ENABLE_ENGINE=ON
cmake --build build --target minicpm5_decode -j$(nproc)
cmake --build build --target minicpm5_server -j$(nproc)
```

## Notes and limitations

- Current generation is greedy-only.
- The local C++ backend consumes token IDs; Python handles tokenizer/chat-template formatting in the web demo.
- The current default max sequence length for the Orange Pi runtime is `4096`.
- `models/`, build outputs, and local dependency directories should remain untracked.

## Related links

- MiniCPM5-1B model: https://huggingface.co/openbmb/MiniCPM5-1B
- Online MiniCPM5-1B demo: https://huggingface.co/spaces/openbmb/MiniCPM5-1B-Demo
