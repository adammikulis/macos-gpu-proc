"""darwin-perf: Live per-process GPU utilization monitor for macOS.

Like `top` or `htop`, but for GPU. Auto-discovers all processes using
the GPU via IORegistry — no sudo needed.

Usage:
    darwin-perf              # monitor all GPU-active processes
    darwin-perf --pid 1234   # monitor specific PID
    darwin-perf --top 10     # show top 10 GPU consumers
    darwin-perf -i 1         # update every 1 second
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time



def _format_table(
    rows: list[tuple[int, str, float, float, float, float]],
) -> str:
    """Format rows as a table string.

    Each row: (pid, name, gpu_pct, cpu_pct, mem_mb, real_mem_mb)
    """
    lines = []
    lines.append(
        f"{'PID':>8}  {'GPU %':>7}  {'CPU %':>7}  {'Memory':>9}  {'Real Mem':>9}  {'Process'}"
    )
    lines.append(
        f"{'─' * 8}  {'─' * 7}  {'─' * 7}  {'─' * 9}  {'─' * 9}  {'─' * 25}"
    )
    for pid, name, gpu_pct, cpu_pct, mem_mb, real_mem_mb in rows:
        bar_len = int(min(gpu_pct, 100) / 5)  # 20 chars = 100%
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(
            f"{pid:>8}  {gpu_pct:>6.1f}%  {cpu_pct:>6.1f}%  "
            f"{mem_mb:>7.1f}MB  {real_mem_mb:>7.1f}MB  {name:<20}  {bar}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="darwin-perf",
        description="Live per-process GPU utilization monitor for macOS.",
    )
    parser.add_argument(
        "--pid", type=int, nargs="+", default=None, help="Monitor specific PIDs"
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Show top N GPU consumers (default: 20)"
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=2.0,
        help="Update interval in seconds (default: 2)",
    )
    parser.add_argument(
        "-n", "--count", type=int, default=0, help="Number of iterations (0 = unlimited)"
    )
    parser.add_argument("-1", "--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch rich terminal UI with sparkline graphs",
    )
    parser.add_argument(
        "--gui", action="store_true", help="Launch native floating window monitor"
    )
    args = parser.parse_args()

    # GUI mode
    if args.gui:
        from darwin_perf.gui import run_gui

        run_gui(interval=args.interval)
        return

    # TUI mode
    if args.tui:
        from darwin_perf.tui import run_tui

        run_tui(pids=args.pid, interval=args.interval, top_n=args.top)
        return

    from darwin_perf import _snapshot
    from darwin_perf._native import cpu_time_ns, proc_info

    if args.once:
        args.count = 1

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    # Initial snapshot
    prev = _snapshot()
    prev_cpu: dict[int, int] = {}
    for pid in prev:
        ns = cpu_time_ns(pid)
        prev_cpu[pid] = ns if ns >= 0 else 0
    prev_time = time.monotonic()
    time.sleep(args.interval)

    iteration = 0
    while True:
        now = time.monotonic()
        elapsed_s = now - prev_time
        elapsed_ns = elapsed_s * 1_000_000_000

        curr = _snapshot()
        curr_cpu: dict[int, int] = {}
        for pid in curr:
            ns = cpu_time_ns(pid)
            curr_cpu[pid] = ns if ns >= 0 else 0

        # Filter to specific PIDs if requested
        pids = set(curr.keys())
        if args.pid:
            pids = pids & set(args.pid)

        rows: list[tuple[int, str, float, float, float, float]] = []
        for pid in pids:
            c_gpu = curr.get(pid, {}).get("gpu_ns", 0)
            p_gpu = prev.get(pid, {}).get("gpu_ns", 0)
            gpu_delta = c_gpu - p_gpu

            c_cpu = curr_cpu.get(pid, 0)
            p_cpu = prev_cpu.get(pid, 0)
            cpu_delta = c_cpu - p_cpu

            gpu_pct = min(gpu_delta / elapsed_ns * 100, 100) if elapsed_ns > 0 else 0
            cpu_pct = cpu_delta / elapsed_ns * 100 if elapsed_ns > 0 else 0

            info = proc_info(pid)
            mem_mb = info["memory"] / (1024 * 1024) if info else 0
            real_mb = info["real_memory"] / (1024 * 1024) if info else 0

            name = curr[pid]["name"]
            if gpu_pct >= 0.1 or gpu_delta > 0:
                rows.append((pid, name, gpu_pct, cpu_pct, mem_mb, real_mb))

        rows.sort(key=lambda r: r[2], reverse=True)
        rows = rows[: args.top]

        if not args.once:
            print("\033[2J\033[H", end="")

        timestamp = time.strftime("%H:%M:%S")
        print(f"darwin-perf  {timestamp}  (every {args.interval}s)\n")

        if rows:
            print(_format_table(rows))
        else:
            print("  No GPU activity detected.")

        prev = curr
        prev_cpu = curr_cpu
        prev_time = now

        iteration += 1
        if 0 < args.count <= iteration:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
