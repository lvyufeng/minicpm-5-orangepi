# MiniCPM5-1B 香橙派 Ascend 310B 本地运行时

本仓库用于在带 Ascend 310B NPU 支持的香橙派设备上本地运行 **MiniCPM5-1B**。目标是端侧 / 本地部署：底层使用轻量 C++ Ascend 推理后端执行模型，可选的 FastAPI Web Demo 提供浏览器聊天界面。

当前本地后端是 **greedy-only**。Web UI 保留 temperature/top-p 控件是为了兼容上游 MiniCPM5 Demo，但这些参数目前还不会作用到 C++ runtime。

## 已支持内容

- 面向 MiniCPM5-1B 文本生成的 **Ascend C++ 推理运行时**。
- 多个性能关键路径的 **Ascend 自定义算子 / kernel**，包括融合 RoPE/cache 更新、attention、RMSNorm、W8A8 decode 辅助算子、SiLU、logits top-1 等。
- **常驻 C++ 后端**（`minicpm5_server`）：权重只加载一次，通过简单的 stdin/stdout 流式协议服务多次生成请求。
- **一次性 CLI**（`minicpm5_decode`）：用于调试和脚本化生成。
- **Web Demo**（`demo/app.py` + `demo/index.html`）：Python 侧使用 Hugging Face tokenizer 组织 chat template 和解码文本，底层从常驻 C++ 后端流式获取 token。

`models/` 目录被 git 忽略，只用于放本地 MiniCPM5 权重，不应该提交模型文件。

## How to start

### 1. 准备环境

运行时脚本会 source `scripts/set_env.sh`，用于配置 Ascend toolkit 库路径和自定义算子路径：

```bash
source scripts/set_env.sh
```

自定义算子通过 `scripts/install_custom_ops.sh` 安装。Web Demo 启动脚本会在检测不到自定义算子安装目录时自动安装。

首次运行前安装 Python 依赖：

```bash
python3 -m pip install -r requirements.txt
```

### 2. 启动 Web Demo

在仓库根目录启动 Demo：

```bash
scripts/run/run_demo.sh
```

然后打开：

```text
http://127.0.0.1:7860
```

如果从同一局域网的其他机器访问：

```text
http://<orange-pi-ip>:7860
```

常用 Demo 环境变量：

```bash
WEIGHTS=models/MiniCPM5-1B        # 模型目录或 safetensors 路径
TOKENIZER=models/MiniCPM5-1B      # tokenizer 路径，默认跟随 WEIGHTS
HOST=0.0.0.0
PORT=7860
MAX_SEQ=4096
MAX_NEW=256
DEVICE_ID=0
```

`run_demo.sh` 会：

1. 检查 / 安装 Ascend 自定义算子；
2. source `scripts/set_env.sh`；
3. 构建 `minicpm5_server`；
4. 启动 `uvicorn demo.app:app`。

FastAPI App 启动后会后台预热常驻 C++ 后端。建议等页面右上角状态显示：

```text
ready · greedy backend
```

之后再测试首 token 延迟。第一次后端加载会明显慢于后续请求，因为权重和 runtime 状态只在初始化时加载一次。

### 3. 运行一次性文本 decode

```bash
WEIGHTS=models/MiniCPM5-1B \
INPUT_IDS=0 \
MAX_NEW=16 \
MAX_SEQ=4096 \
scripts/run/run_minicpm5_decode.sh
```

该脚本会构建并运行 `minicpm5_decode`。这个 CLI 接收 token ID，并打印生成的 token ID。

### 4. 构建目标

```bash
cmake -S . -B build -DMINICPM5_ENABLE_ENGINE=ON
cmake --build build --target minicpm5_decode -j$(nproc)
cmake --build build --target minicpm5_server -j$(nproc)
```

## Performance

以下数据在本地香橙派 Ascend 310B runtime 上实测，使用 MiniCPM5-1B 权重、`MAX_SEQ=4096`、`INPUT_IDS=0`、`MAX_NEW=16`、greedy decode，并开启 `MINICPM_PROFILE=1`。

| 项目 | 实测数据 |
| --- | ---: |
| 一次性权重加载 | 77.1 s |
| 1-token prompt 的 prefill + first lm_head | 801 ms |
| 稳态 decode step，15 个生成 token 平均 | 139 ms/token |
| 稳态 decode 吞吐 | 7.19 tokens/s |
| 一次性 CLI 总耗时，包含权重加载 | 80.0 s |

说明：

- 本地 C++ 后端消费 token ID；Web Demo 中由 Python 负责 tokenizer / chat-template 格式化。
- 当前香橙派 runtime 默认最大序列长度为 `4096`。
- 常驻后端不会在每次请求时重新加载权重；权重加载开销发生在 `minicpm5_server` 启动或重启时，而不是每个 prompt 都重新加载。

常驻后端协议示例：

```bash
source scripts/set_env.sh
printf 'REQUEST 8 0\n' | build/minicpm5_server --weights models/MiniCPM5-1B --max-seq 4096 --device-id 0
```

## Acknowledgement and links

感谢上游 **MiniCPM** / **MiniCPM5** 项目和 Hugging Face demo 提供的参考模型卡、chat template 行为以及在线 Demo 交互方式。

- MiniCPM5-1B 模型：https://huggingface.co/openbmb/MiniCPM5-1B
- MiniCPM5-1B 在线 Demo：https://huggingface.co/spaces/openbmb/MiniCPM5-1B-Demo
