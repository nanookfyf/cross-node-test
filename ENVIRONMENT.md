# NIXL EP 测试环境构建

本文说明如何在两台 GPU 机器上构建 `test/nixl_ep_cross_machine_bench.py` 所需环境。目标是让两端能够导入 `nixl_ep_cu12`，加载 UCX/NIXL 动态库，并通过 RDMA 完成 GPU memory registration。

## 1. 版本与组件

测试脚本直接依赖以下组件：

| 组件 | 要求 |
| --- | --- |
| Linux | 两台机器内核、NVIDIA 驱动和 RDMA 栈应尽量一致 |
| Python | 3.12 |
| PyTorch | CUDA wheel；两端 Torch 主版本、CUDA 架构和 ABI 保持一致 |
| NIXL | `nixl-cu12`；当前已验证环境为 `1.4.1` |
| Python 模块 | `nixl_ep_cu12` |
| 网络 | RDMA 网卡、GPUDirect RDMA、UCX IB transport |
| 控制面 | rank 1 能访问 rank 0 的 TCP Store 端口，默认 `29591` |

本测试导入的是 `nixl_ep_cu12`，不要仅安装 `nixl-cu13` 后直接运行。若已有 vLLM 工作环境，请确认其中实际安装的是本测试所需的 `nixl-cu12`。

## 2. 系统级检查

两台机器分别执行：

```bash
nvidia-smi
python3 --version
ibdev2netdev || true
rdma link show || true
which ibv_devinfo || true
which ucx_info || true
```

确认：

- `nvidia-smi` 能看到目标 GPU，驱动版本满足所安装 PyTorch CUDA wheel 的要求；
- RDMA 设备处于可用状态，且 `ibdev2netdev` 能看到实际端口；
- 两台机器可以互相访问 RDMA 网络地址；
- rank 1 能访问 rank 0 的 TCP Store 地址和端口：

```bash
# 在 rank 1 机器执行；任选一种工具即可
nc -vz NODE_A_IP 29591
```

如果 TCP Store 端口尚未监听，`nc` 失败是正常的；真正运行 benchmark 时需要保证防火墙允许该端口。

## 3. 创建 Python 虚拟环境

推荐使用 `uv`，两台机器使用相同 Python 版本：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
cd /path/to/cross-node
uv venv --python 3.12 test/.venv --seed
source test/.venv/bin/activate
python --version
```

如果机器不能联网，应提前把 `uv`、Python 3.12 wheel、PyTorch wheel 和 NIXL wheel 放入内部镜像或离线目录；不要在两端混用系统 Python 与虚拟环境 Python。

## 4. 安装 PyTorch 与 NIXL

先根据 NVIDIA 驱动和目标 CUDA 版本选择 PyTorch CUDA wheel。下面命令仅展示安装形式，CUDA 版本必须按机器实际情况替换：

```bash
# 示例：使用内部镜像或 PyTorch 对应 CUDA wheel 源
uv pip install torch torchvision torchaudio
uv pip install "nixl-cu12==1.4.1"
uv pip install numpy
```

若项目已有可用 vLLM 虚拟环境，可以不新建 `test/.venv`，直接使用该环境，但两端必须分别确认：

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
python -c 'import nixl_ep_cu12 as n; print(n.__file__)'
```

当前仓库已有两个可参考环境：

```text
`/path/to/vllm/.venv`     Torch 与 CUDA 版本应与目标机器匹配，包含 nixl-cu12 1.4.1
```

实际跨机运行时不要把这两个环境混搭；建议两台机器选择同一个环境构建方案。

## 5. 配置动态库与 UCX

使用仓库环境脚本：

```bash
cd /path/to/cross-node
export NIXL_VENV=/absolute/path/to/vllm/.venv
export OPENSSL_LIB=/absolute/path/to/openssl/lib
export UCX_NET_DEVICES=RDMA_DEVICE:PORT
export UCX_TLS=rc_x,sm,cuda_copy
export UCX_IB_GID_INDEX=3
source /path/to/nixl_ep_env.sh
```

`NIXL_VENV` 必须指向包含以下目录的 Python 环境：

```text
$NIXL_VENV/lib/python3.12/site-packages/nixl_ep_cu12
$NIXL_VENV/lib/python3.12/site-packages/nixl_cu12.libs
$NIXL_VENV/lib/python3.12/site-packages/.nixl_cu12.mesonpy.libs
```

环境脚本会做三件事：

1. 加入 OpenSSL、NIXL wheel 和 UCX bundled libraries 的 `LD_LIBRARY_PATH`；
2. 为 NIXL wheel 中带版本号的 UCX `.so` 创建运行时无版本软链接；
3. 设置 `UCX_MODULE_DIR`、RDMA 网卡和 GID/rail 参数。

检查最终环境：

```bash
echo "$NIXL_VENV"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | sed -n '1,20p'
env | grep -E '^(UCX_|NIXL_|OPENSSL_)' | sort
```

如果实际网卡不是示例中的 `RDMA_DEVICE:PORT`，先执行 `ibdev2netdev`，再设置真实值。例如：

```bash
export UCX_NET_DEVICES=RDMA_DEVICE:PORT
```

如果使用 RoCE，`UCX_IB_GID_INDEX` 必须和集群网络配置一致；不要盲目沿用 `3`。

## 6. 两端一致性检查

在两台机器分别执行以下命令，并比较输出：

```bash
source /path/to/nixl_ep_env.sh
python - <<'PY'
import socket
import torch
import nixl_ep_cu12 as nixl_ep

print("host:", socket.gethostname())
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
print("nixl:", nixl_ep.__file__)
print("topk_dtype:", nixl_ep.topk_idx_t)
PY
```

两端至少应满足：

- `cuda_available: True`；
- Torch 主版本、`torch.version.cuda` 和 NIXL wheel 版本匹配；
- `nixl_ep_cu12` 路径位于预期虚拟环境；
- 选择的 GPU 数量和 `--device` 一致。

## 7. 分层验证顺序

不要第一次就使用大规模 benchmark。建议按以下顺序验证：

### 7.1 Python/CUDA 导入

```bash
source /path/to/nixl_ep_env.sh
python -c 'import torch, nixl_ep_cu12; print(torch.cuda.is_available(), nixl_ep_cu12.__file__)'
```

### 7.2 TCP Store 连通性

在 rank 0 和 rank 1 使用 benchmark 的相同 `--master-addr`、`--store-port`，先确认不会出现 TCPStore timeout。

### 7.3 NIXL 小规模 correctness

```bash
bash test/run_nixl_ep_2node.sh \
  --remote-host NODE_B_HOST \
  --master-addr NODE_A_IP \
  --remote-root /path/to/cross-node \
  --env-file nixl_ep_env.sh \
  -- --hidden 2048 --tokens 16 --warmup 0 --iters 1 --check-every
```

### 7.4 性能测试

小规模 correctness 通过后，再增大 `--tokens`、`--iters`，并记录两端的 UCX 配置、GPU、驱动、网卡和结果 JSON。

## 8. 常见环境错误

### `libssl.so.3` 找不到

设置正确的 `OPENSSL_LIB`，并确认该目录含有 `libssl.so.3` 和 `libcrypto.so.3`：

```bash
find "$OPENSSL_LIB" -maxdepth 1 -name 'libssl.so.3' -o -name 'libcrypto.so.3'
```

### `No module named nixl_ep_cu12`

确认当前 `python` 来自目标虚拟环境，并安装的是 `nixl-cu12`，不是只安装 `nixl-cu13`：

```bash
which python
python -m pip show nixl-cu12
python -c 'import nixl_ep_cu12; print(nixl_ep_cu12.__file__)'
```

### `libucp.so` 或 UCX plugin 找不到

重新 source 环境脚本，检查 `$NIXL_VENV/lib/python3.12/site-packages/nixl_cu12.libs` 是否存在，并确认 `UCX_MODULE_DIR` 指向其 `ucx` 子目录。

### `failed to create handle` / `prepMemView` 失败

这通常发生在 UCX 无法注册 GPU 内存时，优先检查 NVIDIA GPUDirect RDMA、驱动/NIC 兼容性、`UCX_NET_DEVICES` 和 `UCX_TLS`。本机双 GPU 的 `cuda_ipc` 路径成功不代表跨机 RDMA 路径已经可用。

### 两端 Torch/NIXL 版本不同

不要仅比较 Python 包名；同时比较 `torch.__version__`、`torch.version.cuda`、NIXL package version、GPU driver 和 `nixl_ep_cu12.__file__`。版本不一致时重新创建虚拟环境通常比逐个替换 `.so` 更可靠。

## 9. 复现记录模板

每次评测建议保存以下信息：

```text
date:
host/rank:
gpu:
nvidia-smi driver:
python:
torch:
torch.version.cuda:
nixl-cu12:
UCX_NET_DEVICES:
UCX_TLS:
UCX_IB_GID_INDEX:
master_addr/store_port:
benchmark command:
RESULT_JSON:
```
