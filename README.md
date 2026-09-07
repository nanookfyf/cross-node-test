# NIXL EP 跨机器 Dispatch/Combine 评测

本目录提供一套可直接用于两台 GPU 机器的 NIXL Expert Parallel（EP）低延迟路径测试：

- `nixl_ep_cross_machine_bench.py`：每个 rank 启动一个进程，执行 dispatch、专家计算模拟和 combine。
- `run_nixl_ep_2node.sh`：本机启动 rank 0，通过 SSH 在另一台机器启动 rank 1。

脚本同时验证**可用性、数据正确性和性能**。测试数据会按确定规则发往不同 rank 的 expert，不依赖真实 MoE 模型，因此可以单独排查 NIXL/UCX/RDMA。

## 1. 环境构建

完整的 Python、PyTorch、NIXL、动态库、UCX/RDMA 和双机一致性配置请先阅读 [`test/ENVIRONMENT.md`](ENVIRONMENT.md)。建议先完成其中的导入检查、TCP Store 连通性检查和小规模 correctness，再运行下面的性能命令。

## 2. 前置条件

两台机器需要满足：

1. 能够通过 SSH 互相访问；运行 rank 0 的机器可以访问 `--master-addr:--store-port`。
2. 两台机器安装兼容的 CUDA、PyTorch 和 `nixl_ep_cu12`，且 Python/Torch 版本一致。
3. GPU 可用，NIXL 的 RDMA/UCX 网卡配置正确。两台机器应使用同一 RDMA 网络、相同的 UCX 传输配置。
4. 测试脚本在两台机器上的路径一致，或者使用启动器的 `--remote-root`、`--env-file` 等参数调整路径。
5. 防火墙放通 TCP Store 端口（默认 `29591`），并放通 NIXL/UCX 实际使用的 RDMA 流量。

请准备一个本机和远端都可使用的 NIXL 环境脚本（例如 `nixl_ep_env.sh`），它需要配置 NIXL 动态库和 UCX 环境。脚本中的节点地址、Python 虚拟环境路径和 `UCX_NET_DEVICES` 需要按实际机器修改。

> `--master-addr` 必须填写 rank 1 能访问的 rank 0 网卡地址，不能填写 rank 0 的回环地址。

## 3. 推荐启动方式：SSH 两节点启动

在 rank 0 机器执行：

```bash
cd /path/to/cross-node
bash test/run_nixl_ep_2node.sh \
  --remote-host NODE_B_HOST \
  --master-addr NODE_A_IP \
  --remote-root /path/to/cross-node \
  --env-file nixl_ep_env.sh \
  --store-port 29591 \
  --device0 0 \
  --device1 0 \
  -- \
  --hidden 2048 \
  --tokens 256 \
  --experts-per-rank 1 \
  --warmup 10 \
  --iters 50 \
  --check-every
```

默认约定：

- 本机运行 rank 0，远端运行 rank 1。
- 远端仓库根目录通过 `--remote-root` 指定，示例中的 `/path/to/cross-node` 需要替换为实际路径。
- 默认使用 `--env-file` 指定的环境脚本和该环境脚本导出的 Python 环境。
- 默认每台机器使用 GPU 0。

如果远端路径不同：

```bash
bash test/run_nixl_ep_2node.sh \
  --remote-host NODE_B_HOST \
  --master-addr NODE_A_IP \
  --remote-root /path/to/cross-node \
  --env-file nixl_ep_env.sh \
  --local-python /path/to/local/python \
  --remote-python /path/to/remote/python \
  -- --tokens 512 --iters 100
```

启动器会等待远端 rank 1 退出；任一 rank 失败时应检查两端日志。若 SSH 默认用户不一致，请使用 SSH config 或 `--remote-host user@host`。

## 4. 手动启动方式

SSH 到两台机器，各执行一次。两端参数必须一致，只有 `--rank` 和 `--device` 可以不同。

rank 0：

```bash
source /path/to/nixl_ep_env.sh
/path/to/nixl-venv/bin/python \
  /path/to/cross-node/nixl_ep_cross_machine_bench.py \
  --rank 0 --world-size 2 \
  --master-addr NODE_A_IP --store-port 29591 --device 0 \
  --hidden 2048 --tokens 256 --experts-per-rank 1 \
  --warmup 10 --iters 50 --check-every
```

rank 1：

```bash
source /path/to/nixl_ep_env.sh
/path/to/nixl-venv/bin/python \
  /path/to/cross-node/nixl_ep_cross_machine_bench.py \
  --rank 1 --world-size 2 \
  --master-addr NODE_A_IP --store-port 29591 --device 0 \
  --hidden 2048 --tokens 256 --experts-per-rank 1 \
  --warmup 10 --iters 50 --check-every
```

如需保存每个 rank 的 JSON 结果，可追加：

```bash
--output-dir /tmp/nixl-ep-results
```

两台机器使用共享目录时可指定同一个结果目录；非共享目录时分别保存，再手动收集 `rank_0.json` 和 `rank_1.json`。

## 5. 测试内容和正确性判定

### Dispatch

每个 rank 的 token `t` 发往：

```text
target_rank = (source_rank + t % world_size) % world_size
local_expert = t % experts_per_rank
expert_id = target_rank * experts_per_rank + local_expert
```

因此每个目标 expert 都能收到来自所有 source rank 的数据。输入 tensor 的前两列编码 `(source_rank + 1, token_id + 1)`，脚本会：

- 检查 `recv_x` 的形状；
- 检查每个 local expert 的 `recv_count`；
- 对接收 payload 按编码排序后与期望 token 集合比较，不依赖 NIXL 的接收排列顺序。

### Combine

脚本把接收到的 expert 输出模拟为 `recv_x * 2`，再使用 dispatch 返回的 `handle` 调用 combine，并验证最终结果等于原始输入 `x * 2`。这同时覆盖：

- dispatch 返回的 source/layout handle 是否可复用；
- combine 的 top-k index/weight 是否匹配；
- 跨机回传和 reduce 是否完整。

首次 warmup 前会强制执行一次完整 correctness check。增加 `--check-every` 后，每个 benchmark iteration 都检查 dispatch 和 combine，适合稳定性/回归测试，但会增加 CPU 校验开销。

## 6. 输出解释

rank 0 最终会打印类似：

```text
correctness=PASS (dispatch payload and combine round-trip)
RESULT rank=0 host=node-a dispatch_avg_ms=... dispatch_p99_ms=... combine_avg_ms=... combine_p99_ms=...
RESULT rank=1 host=node-b dispatch_avg_ms=... dispatch_p99_ms=... combine_avg_ms=... combine_p99_ms=...
RESULT_JSON=[...]
```

延迟是单 rank 的 Python wall-clock 时间，调用后执行 `torch.cuda.synchronize()`，因此包含该操作直到 GPU 完成的时间。`dispatch` 和 `combine` 分开统计；跨 rank 比较时应重点关注较慢 rank 的 `p99_ms`。脚本还直接输出按单向 BF16 payload 计算的 `dispatch_gbps` 和 `combine_gbps`。payload 大小为 `tokens * hidden * 2` 字节（BF16），计算公式为：

```text
GB/s = payload_bytes / (latency_ms / 1000) / 1e9
```

## 7. 参数约束

- `--world-size >= 2`；当前 SSH 启动器固定为两节点、两 rank。
- `--tokens` 必须能被 `world-size` 和 `experts-per-rank` 整除。
- `--hidden >= 2`；NIXL 常用 BF16 hidden size 建议使用实际模型尺寸，例如 `2048`。
- 两端的 `--hidden`、`--tokens`、`--experts-per-rank`、`--warmup`、`--iters` 必须一致。
- `--timeout-ms` 同时影响 TCP Store 等待和 NIXL GPU 通信超时，慢网络可适当增大。

## 8. 常见故障定位

- `libssl.so.3`、`libucp.so` 找不到：先 source `--env-file` 指定的环境脚本，确认 `LD_LIBRARY_PATH` 和 `UCX_MODULE_DIR`。
- `TCPStore` timeout：检查 `--master-addr`、端口、防火墙和 rank 1 到 rank 0 的连通性。
- `connect_ranks` 或 UCX 初始化失败：确认 `UCX_NET_DEVICES`、`UCX_TLS`、`UCX_IB_GID_INDEX` 与 `ibdev2netdev`/网卡实际配置一致。
- 本机双 GPU 试跑在 `ucp_device.c` 报 `failed to create handle`：这通常表示当前 UCX/驱动组合不支持该本机 GPU 内存路径；本测试目标是跨机 RDMA，应优先在目标两台机器上验证，并检查 UCX GPU memory registration、GPUDirect RDMA 和网卡插件。
- correctness 失败：先使用 `--tokens 16 --warmup 0 --iters 1 --check-every` 缩小问题，再查看两端 rank 日志；不要只看最终延迟。
- 运行中途退出：NIXL 低延迟 buffer 只有有限复用槽位，脚本保持每轮 dispatch/combine 串行完成，不要在此基础上并发持有多个结果 tensor。

## 9. 扩展到更多 rank

Python benchmark 本身支持 `--world-size N`，只需在 N 台机器或多 GPU 进程上分别启动 rank `0..N-1`，并让所有进程连接同一个 TCP Store。现有 `run_nixl_ep_2node.sh` 为避免误用只封装了两 rank；多 rank 建议使用作业调度器或按“手动启动方式”批量启动。
