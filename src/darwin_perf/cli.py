"""darwin-perf: Live per-process GPU utilization monitor for macOS.

Like `top` or `htop`, but for GPU. Auto-discovers all processes using
the GPU via IORegistry — no sudo needed.

Usage:
    darwin-perf              # monitor all GPU-active processes
    darwin-perf --pid 1234   # monitor specific PID
    darwin-perf --top 10     # show top 10 GPU consumers
    darwin-perf -i 1         # update every 1 second
    darwin-perf --json       # JSON line per update
    darwin-perf --csv        # CSV output
    darwin-perf --record f   # record snapshots to JSONL file
    darwin-perf --replay f   # replay a recorded JSONL file
"""

from __future__ import annotations

import argparse
import json
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


def _collect_snapshot(args, prev, prev_cpu, prev_time):
    """Collect one snapshot and return (rows, curr, curr_cpu, now)."""
    from darwin_perf import _snapshot
    from darwin_perf._native import cpu_time_ns, proc_info

    now = time.monotonic()
    elapsed_s = now - prev_time
    elapsed_ns = elapsed_s * 1_000_000_000

    curr = _snapshot()
    curr_cpu: dict[int, int] = {}
    for pid in curr:
        ns = cpu_time_ns(pid)
        curr_cpu[pid] = ns if ns >= 0 else 0

    pids = set(curr.keys())
    if args.pid:
        pids = pids & set(args.pid)

    rows: list[dict] = []
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
        api = curr[pid].get("api", "unknown")
        if gpu_pct >= 0.1 or gpu_delta > 0:
            rows.append({
                "pid": pid,
                "name": name,
                "api": api,
                "gpu_pct": round(gpu_pct, 1),
                "cpu_pct": round(cpu_pct, 1),
                "mem_mb": round(mem_mb, 1),
                "real_mem_mb": round(real_mb, 1),
            })

    rows.sort(key=lambda r: r["gpu_pct"], reverse=True)
    rows = rows[: args.top]
    return rows, curr, curr_cpu, now


def _run_json(args):
    """JSON streaming mode: one JSON line per update."""
    from darwin_perf import _snapshot
    from darwin_perf._native import cpu_time_ns

    prev = _snapshot()
    prev_cpu: dict[int, int] = {}
    for pid in prev:
        ns = cpu_time_ns(pid)
        prev_cpu[pid] = ns if ns >= 0 else 0
    prev_time = time.monotonic()
    time.sleep(args.interval)

    iteration = 0
    while True:
        rows, prev, prev_cpu, prev_time = _collect_snapshot(args, prev, prev_cpu, prev_time)
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "epoch": time.time(),
            "processes": rows,
        }
        print(json.dumps(record), flush=True)

        iteration += 1
        if 0 < args.count <= iteration:
            break
        time.sleep(args.interval)


def _run_csv(args):
    """CSV streaming mode: header + one row per process per update."""
    from darwin_perf import _snapshot
    from darwin_perf._native import cpu_time_ns

    prev = _snapshot()
    prev_cpu: dict[int, int] = {}
    for pid in prev:
        ns = cpu_time_ns(pid)
        prev_cpu[pid] = ns if ns >= 0 else 0
    prev_time = time.monotonic()
    time.sleep(args.interval)

    header = "timestamp,pid,name,api,gpu_pct,cpu_pct,mem_mb,real_mem_mb"
    print(header, flush=True)

    iteration = 0
    while True:
        rows, prev, prev_cpu, prev_time = _collect_snapshot(args, prev, prev_cpu, prev_time)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        for r in rows:
            # Escape commas in name
            name = r["name"].replace(",", ";")
            print(f"{ts},{r['pid']},{name},{r['api']},{r['gpu_pct']},{r['cpu_pct']},{r['mem_mb']},{r['real_mem_mb']}", flush=True)

        iteration += 1
        if 0 < args.count <= iteration:
            break
        time.sleep(args.interval)


def _run_record(args):
    """Record snapshots to a JSONL file."""
    from darwin_perf import snapshot

    with open(args.record, "w") as f:
        iteration = 0
        while True:
            data = snapshot(interval=args.interval, detailed=True)
            record = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "epoch": time.time(),
                "interval": args.interval,
                "processes": data,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

            iteration += 1
            if 0 < args.count <= iteration:
                break
            # snapshot() already sleeps for interval, no additional sleep needed


def _run_replay(args):
    """Replay a recorded JSONL file with original timing."""
    with open(args.replay) as f:
        lines = f.readlines()

    if not lines:
        print("Empty recording file.", file=sys.stderr)
        return

    prev_epoch = None
    for line_str in lines:
        record = json.loads(line_str.strip())
        epoch = record.get("epoch", 0)

        if prev_epoch is not None and not args.once:
            delay = epoch - prev_epoch
            if delay > 0:
                time.sleep(delay)
        prev_epoch = epoch

        ts = record.get("timestamp", "")
        procs = record.get("processes", [])

        if not args.once:
            print("\033[2J\033[H", end="")
        print(f"darwin-perf replay  {ts}\n")

        if procs:
            rows = []
            for p in procs:
                rows.append((
                    p.get("pid", 0),
                    p.get("name", "?"),
                    p.get("gpu_percent", 0),
                    p.get("cpu_percent", 0),
                    p.get("memory_mb", 0),
                    p.get("memory_mb", 0),  # real_mem not in snapshot output
                ))
            print(_format_table(rows))
        else:
            print("  No GPU activity.")

        if args.once:
            break


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
    parser.add_argument(
        "--json", action="store_true", help="Output one JSON line per update"
    )
    parser.add_argument(
        "--csv", action="store_true", help="Output CSV (header + one row per process per update)"
    )
    parser.add_argument(
        "--record", type=str, metavar="FILE",
        help="Record detailed snapshots to a JSONL file"
    )
    parser.add_argument(
        "--replay", type=str, metavar="FILE",
        help="Replay a recorded JSONL file"
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

    if args.once:
        args.count = 1

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    # Replay mode
    if args.replay:
        _run_replay(args)
        return

    # Record mode
    if args.record:
        _run_record(args)
        return

    # JSON streaming mode
    if args.json:
        _run_json(args)
        return

    # CSV streaming mode
    if args.csv:
        _run_csv(args)
        return

    # Default table mode
    from darwin_perf import _snapshot
    from darwin_perf._native import cpu_time_ns, proc_info

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
