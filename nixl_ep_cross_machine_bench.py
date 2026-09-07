#!/usr/bin/env python3
"""Cross-machine NIXL EP dispatch/combine correctness and performance benchmark."""

from __future__ import annotations

import argparse
import atexit
import datetime
import json
import math
import socket
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a cross-machine NIXL low-latency EP dispatch/combine benchmark. "
            "Start one process per rank; all ranks must use identical arguments."
        )
    )
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--store-port", type=int, default=29591)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--experts-per-rank", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--check-every",
        action="store_true",
        help="Run the CPU correctness checks on every measured iteration.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.rank < args.world_size:
        fail(f"--rank must be in [0, {args.world_size}), got {args.rank}")
    if args.world_size < 2:
        fail("--world-size must be at least 2 for a cross-machine test")
    if args.hidden < 2:
        fail("--hidden must be at least 2; columns 0 and 1 carry correctness IDs")
    if args.tokens <= 0 or args.tokens % args.world_size:
        fail("--tokens must be positive and divisible by --world-size")
    if args.tokens % args.experts_per_rank:
        fail("--tokens must be divisible by --experts-per-rank")
    if args.experts_per_rank <= 0:
        fail("--experts-per-rank must be positive")
    if args.warmup < 0 or args.iters <= 0:
        fail("--warmup must be >= 0 and --iters must be > 0")
    if args.timeout_ms <= 0:
        fail("--timeout-ms must be positive")


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentage * len(ordered)) - 1))
    return ordered[index]


def stats(values: list[float]) -> dict[str, float]:
    return {
        "min_ms": min(values),
        "avg_ms": sum(values) / len(values),
        "p50_ms": percentile(values, 0.50),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values),
    }


def bandwidth_gbps(payload_bytes: int, latency_ms: float) -> float:
    return payload_bytes / (latency_ms / 1000.0) / 1e9


def tensor_key(tensor: Any, token_count: int) -> Any:
    # The first two columns are rank_id + 1 and token_id + 1. Sorting by a
    # scalar key makes validation independent of NIXL's receive ordering.
    return tensor[:, 0].to(dtype=tensor.dtype) * (token_count + 1) + tensor[:, 1]


def assert_dispatch_correct(
    recv_x: Any,
    recv_count: Any,
    rank: int,
    world_size: int,
    tokens: int,
    experts_per_rank: int,
    hidden: int,
) -> None:
    expected_per_expert = tokens // experts_per_rank
    if tuple(recv_x.shape) != (
        experts_per_rank,
        tokens * world_size,
        hidden,
    ):
        fail(f"unexpected recv_x shape: {tuple(recv_x.shape)}")

    counts = recv_count.detach().cpu().tolist()
    expected_counts = [expected_per_expert] * experts_per_rank
    if counts != expected_counts:
        fail(f"dispatch recv_count mismatch: got {counts}, expected {expected_counts}")

    received = recv_x.detach().float().cpu()
    for expert in range(experts_per_rank):
        count = counts[expert]
        actual = received[expert, :count, :2]
        expected_pairs = [
            (source_rank + 1, token + 1)
            for source_rank in range(world_size)
            for token in range(tokens)
            if (source_rank + token % world_size) % world_size == rank
            and token % experts_per_rank == expert
        ]
        expected = recv_x.new_tensor(expected_pairs, dtype=recv_x.dtype).float().cpu()
        actual_key = tensor_key(actual, tokens).sort().values
        expected_key = tensor_key(expected, tokens).sort().values
        if not torch_allclose(actual_key, expected_key):
            fail(
                f"dispatch payload mismatch for rank={rank}, expert={expert}: "
                f"received keys={actual_key.tolist()}, expected keys={expected_key.tolist()}"
            )


def torch_allclose(actual: Any, expected: Any) -> bool:
    # BF16 payloads are compared after conversion to FP32. The IDs are small,
    # so this tolerance catches corruption while allowing BF16 quantization.
    return bool(actual.shape == expected.shape and actual.numel() == expected.numel() and
                (actual - expected).abs().max().item() <= 0.02)


def assert_combine_correct(combined_x: Any, x: Any) -> None:
    if tuple(combined_x.shape) != tuple(x.shape):
        fail(
            f"unexpected combined_x shape: {tuple(combined_x.shape)}, "
            f"expected {tuple(x.shape)}"
        )
    actual = combined_x.detach().float()
    expected = (x * 2).detach().float()
    max_error = (actual - expected).abs().max().item()
    if max_error > 0.05:
        fail(f"combine payload mismatch: max_abs_error={max_error}")


def make_input(torch: Any, rank: int, tokens: int, hidden: int, device: str) -> Any:
    x = torch.zeros((tokens, hidden), dtype=torch.bfloat16, device=device)
    x[:, 0] = rank + 1
    x[:, 1] = torch.arange(1, tokens + 1, dtype=torch.bfloat16, device=device)
    if hidden > 2:
        x[:, 2:] = (rank + 1) * 0.25
    return x


def make_route(torch: Any, nixl_ep: Any, rank: int, world_size: int, tokens: int,
               experts_per_rank: int, device: str) -> tuple[Any, Any]:
    token_ids = torch.arange(tokens, dtype=torch.int64, device=device)
    target_rank = (rank + token_ids.remainder(world_size)).remainder(world_size)
    local_expert = token_ids.remainder(experts_per_rank)
    topk_idx = (target_rank * experts_per_rank + local_expert).view(tokens, 1)
    topk_weights = torch.ones((tokens, 1), dtype=torch.float32, device=device)
    return topk_idx.to(dtype=nixl_ep.topk_idx_t), topk_weights


def print_line(rank: int, message: str) -> None:
    print(f"[{socket.gethostname()} rank={rank}] {message}", flush=True)


def main() -> int:
    args = parse_args()
    validate_args(args)

    import torch
    import torch.distributed as dist
    import nixl_ep_cu12 as nixl_ep

    if not torch.cuda.is_available():
        fail("CUDA is not available")
    torch.cuda.set_device(args.device)
    device = f"cuda:{args.device}"
    timeout = datetime.timedelta(milliseconds=args.timeout_ms)
    store = dist.TCPStore(
        args.master_addr,
        args.store_port,
        world_size=args.world_size,
        is_master=args.rank == 0,
        timeout=timeout,
    )

    num_experts = args.world_size * args.experts_per_rank
    max_tokens_per_rank = args.tokens
    buffer_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        max_tokens_per_rank, args.hidden, args.world_size, num_experts
    )
    buffer = nixl_ep.Buffer(
        rank=args.rank,
        low_latency_mode=True,
        explicitly_destroy=True,
        tcp_store_group=store,
        timeout_ms=args.timeout_ms,
    )

    def destroy_buffer() -> None:
        if getattr(buffer, "runtime", None) is not None:
            try:
                buffer.destroy()
            except Exception as exc:
                print_line(args.rank, f"cleanup warning: {exc}")

    atexit.register(destroy_buffer)

    print_line(
        args.rank,
        f"device={device} world_size={args.world_size} hidden={args.hidden} "
        f"tokens={args.tokens} experts={num_experts} rdma_buffer_bytes={buffer_bytes}",
    )

    x = make_input(torch, args.rank, args.tokens, args.hidden, device)
    topk_idx, topk_weights = make_route(
        torch, nixl_ep, args.rank, args.world_size, args.tokens,
        args.experts_per_rank, device,
    )
    buffer.update_memory_buffers(
        num_ranks=args.world_size,
        num_experts_per_rank=args.experts_per_rank,
        num_rdma_bytes=buffer_bytes,
        num_nvl_bytes=0,
    )
    buffer.connect_ranks([rank for rank in range(args.world_size) if rank != args.rank])
    buffer.barrier()
    print_line(args.rank, "all ranks connected")

    recv_x, recv_count, handle, _, _ = buffer.dispatch(
        x,
        topk_idx,
        max_tokens_per_rank,
        num_experts=num_experts,
        use_fp8=False,
        async_finish=False,
    )
    torch.cuda.synchronize()
    assert_dispatch_correct(
        recv_x, recv_count, args.rank, args.world_size, args.tokens,
        args.experts_per_rank, args.hidden,
    )

    expert_output = recv_x * 2
    combined_x, _, _ = buffer.combine(
        expert_output,
        topk_idx,
        topk_weights,
        handle,
        async_finish=False,
    )
    torch.cuda.synchronize()
    assert_combine_correct(combined_x, x)
    buffer.barrier()
    print_line(args.rank, "correctness=PASS (dispatch payload and combine round-trip)")

    for _ in range(args.warmup):
        recv_x, _, handle, _, _ = buffer.dispatch(
            x, topk_idx, max_tokens_per_rank, num_experts=num_experts,
            use_fp8=False, async_finish=False,
        )
        combined_x, _, _ = buffer.combine(
            recv_x * 2, topk_idx, topk_weights, handle, async_finish=False,
        )
        torch.cuda.synchronize()
    buffer.barrier()

    dispatch_ms: list[float] = []
    combine_ms: list[float] = []
    for iteration in range(args.iters):
        buffer.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        recv_x, recv_count, handle, _, _ = buffer.dispatch(
            x, topk_idx, max_tokens_per_rank, num_experts=num_experts,
            use_fp8=False, async_finish=False,
        )
        torch.cuda.synchronize()
        dispatch_ms.append((time.perf_counter() - start) * 1000.0)

        if args.check_every:
            assert_dispatch_correct(
                recv_x, recv_count, args.rank, args.world_size, args.tokens,
                args.experts_per_rank, args.hidden,
            )

        buffer.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        combined_x, _, _ = buffer.combine(
            recv_x * 2, topk_idx, topk_weights, handle, async_finish=False,
        )
        torch.cuda.synchronize()
        combine_ms.append((time.perf_counter() - start) * 1000.0)

        if args.check_every:
            assert_combine_correct(combined_x, x)
        if iteration == 0 or iteration + 1 == args.iters:
            print_line(
                args.rank,
                f"iteration={iteration + 1}/{args.iters} "
                f"dispatch_ms={dispatch_ms[-1]:.3f} combine_ms={combine_ms[-1]:.3f}",
            )

    result = {
        "rank": args.rank,
        "hostname": socket.gethostname(),
        "device": args.device,
        "world_size": args.world_size,
        "tokens": args.tokens,
        "hidden": args.hidden,
        "experts_per_rank": args.experts_per_rank,
        "dispatch": stats(dispatch_ms),
        "combine": stats(combine_ms),
        "payload_bytes": args.tokens * args.hidden * 2,
        "correctness": "PASS",
    }
    result["dispatch_gbps"] = bandwidth_gbps(
        result["payload_bytes"], result["dispatch"]["avg_ms"]
    )
    result["combine_gbps"] = bandwidth_gbps(
        result["payload_bytes"], result["combine"]["avg_ms"]
    )
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"rank_{args.rank}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )

    store.set(f"NIXL_EP_BENCH_RESULT/{args.rank}", json.dumps(result))
    buffer.barrier()
    if args.rank == 0:
        results = [
            json.loads(store.get(f"NIXL_EP_BENCH_RESULT/{rank}"))
            for rank in range(args.world_size)
        ]
        print("RESULT_JSON=" + json.dumps(results, sort_keys=True), flush=True)
        for item in results:
            print(
                f"RESULT rank={item['rank']} host={item['hostname']} "
                f"dispatch_avg_ms={item['dispatch']['avg_ms']:.3f} "
                f"dispatch_p99_ms={item['dispatch']['p99_ms']:.3f} "
                f"dispatch_gbps={item['dispatch_gbps']:.3f} "
                f"combine_avg_ms={item['combine']['avg_ms']:.3f} "
                f"combine_p99_ms={item['combine']['p99_ms']:.3f} "
                f"combine_gbps={item['combine_gbps']:.3f}",
                flush=True,
            )

    buffer.barrier()
    destroy_buffer()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
