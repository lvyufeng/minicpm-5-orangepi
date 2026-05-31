# MiniCPM5-1B on Orange Pi Ascend 310B

This repository contains a local inference runtime for running **MiniCPM5-1B** on Orange Pi boards with Ascend 310B NPU support. It is focused on edge/on-device deployment: a lightweight C++ Ascend backend executes the model, and an optional FastAPI web demo provides a browser chat UI.

The current local backend is **greedy-only**. The web UI keeps temperature/top-p controls for compatibility with the upstream MiniCPM5 demo, but those parameters are not applied by the C++ runtime yet.

## What is included

- **Ascend C++ inference runtime** for MiniCPM5-1B text generation.
- **Custom Ascend kernels / ops** for performance-critical decode paths, including fused RoPE/cache update, attention, RMSNorm, W8A8 decode helpers, SiLU, and logits top-1.
- **Persistent C++ backend** (`minicpm5_server`) that loads weights once and serves multiple generation requests over a simple stdin/stdout streaming protocol.
- **One-shot CLI** (`minicpm5_decode`) for debugging and scripted generation.
- **Web demo** (`demo/app.py` + `demo/index.html`) that uses the Hugging Face tokenizer in Python and streams generated text from the persistent C++ backend.

The `models/` directory is intentionally ignored by git. Put local MiniCPM5 weights there, but do not commit model files.

## How to start

### 1. Prepare the environment

The runtime scripts source `scripts/set_env.sh`, which configures the Ascend toolkit libraries and custom-op path:

```bash
source scripts/set_env.sh
```

Custom ops are installed by `scripts/install_custom_ops.sh`. The web demo launcher runs this automatically if the expected custom-op install directory is missing.

Install the Python dependencies once:

```bash
python3 -m pip install -r requirements.txt
```

### 2. Run the web demo

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

Useful demo environment variables:

```bash
WEIGHTS=models/MiniCPM5-1B        # model directory or safetensors path
TOKENIZER=models/MiniCPM5-1B      # tokenizer path; defaults to WEIGHTS
HOST=0.0.0.0
PORT=7860
MAX_SEQ=4096
MAX_NEW=256
DEVICE_ID=0
```

`run_demo.sh`:

1. checks/installs custom Ascend ops if needed;
2. sources `scripts/set_env.sh`;
3. builds `minicpm5_server`;
4. starts `uvicorn demo.app:app`.

The FastAPI app prewarms the persistent C++ backend after startup. Wait until the UI status badge shows:

```text
ready · greedy backend
```

before measuring first-token latency. The first backend load can take much longer than later requests because weights and runtime state are initialized once.

### 3. Run one-shot text decode

```bash
WEIGHTS=models/MiniCPM5-1B \
INPUT_IDS=0 \
MAX_NEW=16 \
MAX_SEQ=4096 \
scripts/run/run_minicpm5_decode.sh
```

This builds and runs `minicpm5_decode`, which accepts token IDs and prints generated token IDs.

### 4. Build targets

```bash
cmake -S . -B build -DMINICPM5_ENABLE_ENGINE=ON
cmake --build build --target minicpm5_decode -j$(nproc)
cmake --build build --target minicpm5_server -j$(nproc)
```

## Performance

Measured on the local Orange Pi Ascend 310B runtime with MiniCPM5-1B weights, `MAX_SEQ=4096`, `INPUT_IDS=0`, `MAX_NEW=16`, greedy decoding, and `MINICPM_PROFILE=1`.

| Item | Measured value |
| --- | ---: |
| One-time weight load | 77.1 s |
| Prefill + first lm_head, 1-token prompt | 801 ms |
| Steady decode step, average over 15 generated tokens | 139 ms/token |
| Steady decode throughput | 7.19 tokens/s |
| One-shot CLI total time, including weight load | 80.0 s |

Notes:

- The local C++ backend consumes token IDs; Python handles tokenizer/chat-template formatting in the web demo.
- The current default max sequence length for the Orange Pi runtime is `4096`.
- The persistent backend avoids reloading weights on every request; the weight-load cost is paid when `minicpm5_server` starts or restarts, not for every prompt.

Persistent backend protocol example:

```bash
source scripts/set_env.sh
printf 'REQUEST 8 0\n' | build/minicpm5_server --weights models/MiniCPM5-1B --max-seq 4096 --device-id 0
```

## Acknowledgement and links

Thanks to the upstream **MiniCPM** / **MiniCPM5** project and the Hugging Face demo for the reference model card, chat template behavior, and online demo UX.

- MiniCPM5-1B model: https://huggingface.co/openbmb/MiniCPM5-1B
- Online MiniCPM5-1B demo: https://huggingface.co/spaces/openbmb/MiniCPM5-1B-Demo
