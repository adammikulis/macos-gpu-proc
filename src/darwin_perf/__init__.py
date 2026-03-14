"""darwin-perf: System performance monitoring for macOS Apple Silicon.

GPU, CPU, memory, energy, temperature, and disk I/O metrics via Mach
kernel APIs, IORegistry, and AppleSMC. No sudo needed.

Quick start::

    from darwin_perf import snapshot

    # Per-process GPU/CPU utilization with system context
    for proc in snapshot():
        print(f"{proc['name']:20s}  GPU {proc['gpu_percent']:5.1f}%")

Temperatures (instant, no sudo)::

    from darwin_perf import temperatures
    t = temperatures()
    print(f"CPU: {t['cpu_avg']:.1f}°C  GPU: {t['gpu_avg']:.1f}°C")

CPU cluster frequency and power::

    from darwin_perf import cpu_power
    c = cpu_power(0.5)
    for name, cluster in c['clusters'].items():
        print(f"{name}: {cluster['freq_mhz']} MHz")

GPU power, frequency, and thermal::

    from darwin_perf import gpu_power
    p = gpu_power(0.5)
    print(f"{p['gpu_power_w']:.1f}W  {p['gpu_freq_mhz']}MHz  throttled={p['throttled']}")
"""

from __future__ import annotations

import os
import threading
import time as _time
from typing import Any

from ._native import (
    cpu_power,
    cpu_time_ns,
    gpu_clients,
    gpu_freq_table,
    gpu_power,
    gpu_time_ns,
    gpu_time_ns_multi,
    ppid,
    proc_info,
    system_gpu_stats,
    system_stats,
    temperatures,
)

__all__ = [
    "GpuMonitor",
    "cpu_power",
    "cpu_time_ns",
    "gpu_clients",
    "gpu_freq_table",
    "gpu_power",
    "gpu_time_ns",
    "gpu_time_ns_multi",
    "gpu_percent",
    "ppid",
    "proc_info",
    "sample_gpu",
    "snapshot",
    "system_gpu_stats",
    "system_stats",
    "temperatures",
]

__version__ = "0.4.0"


def _snapshot() -> dict[int, dict]:
    """Take a snapshot of all GPU clients aggregated by PID.

    Returns dict: pid -> {'name': str, 'gpu_ns': int, 'api': str}
    """
    by_pid: dict[int, dict] = {}
    for c in gpu_clients():
        pid = c["pid"]
        if pid not in by_pid:
            by_pid[pid] = {"name": c["name"], "gpu_ns": 0, "api": c.get("api", "unknown")}
        by_pid[pid]["gpu_ns"] += c["gpu_ns"]
    return by_pid


def gpu_percent(pid: int = 0, interval: float = 0.5) -> float:
    """One-shot GPU utilization percentage for a process.

    Samples GPU time twice over ``interval`` seconds and returns the
    percentage of that interval spent on GPU work.

    Args:
        pid: Process ID. 0 means the calling process.
        interval: Sampling interval in seconds (default 0.5).

    Returns:
        GPU utilization as a percentage (0.0 - 100.0).
    """
    t1 = gpu_time_ns(pid)
    _time.sleep(interval)
    t2 = gpu_time_ns(pid)
    delta_ns = t2 - t1
    interval_ns = interval * 1_000_000_000
    return min((delta_ns / interval_ns) * 100.0, 100.0) if interval_ns > 0 else 0.0


def sample_gpu(pids: list[int] | None = None, interval: float = 0.5) -> dict[int, float]:
    """One-shot GPU utilization for multiple processes.

    Args:
        pids: List of PIDs. None or empty means [0] (current process).
        interval: Sampling interval in seconds.

    Returns:
        Dict mapping PID to GPU utilization percentage.
    """
    if not pids:
        pids = [0]
    t1 = gpu_time_ns_multi(pids)
    _time.sleep(interval)
    t2 = gpu_time_ns_multi(pids)
    interval_ns = interval * 1_000_000_000
    result: dict[int, float] = {}
    for pid in pids:
        ns1 = t1.get(pid, 0)
        ns2 = t2.get(pid, 0)
        result[pid] = min(((ns2 - ns1) / interval_ns) * 100.0, 100.0)
    return result


def snapshot(
    interval: float = 1.0,
    active_only: bool = True,
    detailed: bool = False,
    system: bool = False,
) -> list[dict] | dict:
    """One-call system and process performance snapshot.

    Auto-discovers every process using the GPU, measures utilization
    over ``interval`` seconds, and returns ready-to-use results sorted
    by GPU % descending. No PID lookup needed.

    Args:
        interval: Measurement window in seconds (default 1.0).
        active_only: If True (default), only return processes with GPU
            activity during the interval. Set to False to include all
            processes that have a GPU client (even if idle).
        detailed: If True, include extended process fields (IPC, wakeups,
            disk I/O, peak memory, wired memory, neural engine).
        system: If True, return a dict with full system context instead
            of just the process list. Includes CPU/GPU power, frequencies,
            temperatures, memory, and per-process data.

    Returns:
        Without ``system``: list of dicts sorted by gpu_percent descending::

            pid, name, gpu_percent, gpu_ns, cpu_percent, memory_mb,
            energy_w, threads

        With ``detailed=True``, adds::

            peak_memory_mb, wired_mb, neural_mb, disk_read_mb, disk_write_mb,
            instructions, cycles, ipc, idle_wakeups, pageins

        With ``system=True``: dict with keys::

            processes: list — same as above
            cpu: dict — cpu_power_w, cpu_energy_nj, clusters (ECPU/PCPU)
            gpu: dict — gpu_power_w, gpu_freq_mhz, throttled, frequency_states
            temperatures: dict — cpu_avg, gpu_avg, system_avg, per-sensor
            memory: dict — total, used, available, compressed, etc.
            gpu_stats: dict — device_utilization, model, gpu_core_count, etc.

    Example::

        from darwin_perf import snapshot

        for proc in snapshot():
            print(f"{proc['name']:20s}  GPU {proc['gpu_percent']:5.1f}%  "
                  f"CPU {proc['cpu_percent']:5.1f}%  {proc['memory_mb']:.0f}MB")

        # Full system recording:
        s = snapshot(system=True)
        print(f"CPU: {s['cpu']['cpu_power_w']:.1f}W  GPU: {s['gpu']['gpu_power_w']:.1f}W")
        print(f"Temps: CPU {s['temperatures']['cpu_avg']:.0f}°C")
    """
    if system:
        # Parallel sampling: gpu_power and cpu_power both sleep for interval,
        # so we run them concurrently with process snapshot collection.
        import concurrent.futures

        snap1 = _snapshot()
        info1: dict[int, dict] = {}
        for pid in snap1:
            i = proc_info(pid)
            if i:
                info1[pid] = i

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            gpu_fut = pool.submit(gpu_power, interval)
            cpu_fut = pool.submit(cpu_power, interval)
            # Temperature is instant, read after interval
            _time.sleep(interval)
            temps = temperatures()

        gpu_data = gpu_fut.result()
        cpu_data = cpu_fut.result()

        snap2 = _snapshot()
        process_list = _build_process_list(
            snap1, snap2, info1, interval, active_only, detailed,
        )

        return {
            "processes": process_list,
            "cpu": cpu_data,
            "gpu": gpu_data,
            "temperatures": temps,
            "memory": system_stats(),
            "gpu_stats": system_gpu_stats(),
        }

    # Non-system mode: original behavior
    snap1 = _snapshot()
    info1 = {}
    for pid in snap1:
        i = proc_info(pid)
        if i:
            info1[pid] = i

    _time.sleep(interval)

    snap2 = _snapshot()
    return _build_process_list(snap1, snap2, info1, interval, active_only, detailed)


def _build_process_list(
    snap1: dict, snap2: dict, info1: dict,
    interval: float, active_only: bool, detailed: bool,
) -> list[dict]:
    """Build per-process stats from two GPU snapshots."""
    interval_ns = interval * 1_000_000_000
    results = []
    for pid, c2 in snap2.items():
        c1 = snap1.get(pid)
        gpu_delta = c2["gpu_ns"] - (c1["gpu_ns"] if c1 else c2["gpu_ns"])
        if gpu_delta <= 0 and c1 is None:
            continue

        info = proc_info(pid)
        i1 = info1.get(pid)

        cpu2 = info["cpu_ns"] if info else 0
        cpu1_val = i1["cpu_ns"] if i1 else cpu2
        cpu_delta = cpu2 - cpu1_val

        energy2 = info["energy_nj"] if info else 0
        energy1_val = i1["energy_nj"] if i1 else energy2
        energy_delta = energy2 - energy1_val

        gpu_pct = min(gpu_delta / interval_ns * 100, 100) if interval_ns > 0 else 0
        cpu_pct = cpu_delta / interval_ns * 100 if interval_ns > 0 else 0
        power_w = energy_delta / (interval * 1e9) if interval > 0 else 0

        if active_only and gpu_pct < 0.05 and gpu_delta <= 0:
            continue

        MB = 1024 * 1024
        entry: dict = {
            "pid": pid,
            "name": c2["name"],
            "gpu_percent": round(gpu_pct, 1),
            "gpu_ns": gpu_delta,
            "cpu_percent": round(cpu_pct, 1),
            "memory_mb": round(info["memory"] / MB, 1) if info else 0,
            "energy_w": round(power_w, 2),
            "threads": info["threads"] if info else 0,
        }

        if detailed and info:
            entry.update({
                "peak_memory_mb": round(info["peak_memory"] / MB, 1),
                "wired_mb": round(info["wired_size"] / MB, 1),
                "neural_mb": round(info["neural_footprint"] / MB, 1),
                "disk_read_mb": round(info["disk_read_bytes"] / MB, 1),
                "disk_write_mb": round(info["disk_write_bytes"] / MB, 1),
                "instructions": info["instructions"],
                "cycles": info["cycles"],
                "ipc": round(info["instructions"] / info["cycles"], 2) if info["cycles"] > 0 else 0,
                "idle_wakeups": info["idle_wakeups"],
                "pageins": info["pageins"],
            })

        results.append(entry)

    results.sort(key=lambda r: r["gpu_percent"], reverse=True)
    return results


class GpuMonitor:
    """Continuous per-process GPU utilization monitor.

    Computes GPU % from the delta of cumulative GPU nanosecond counters
    between calls to :meth:`sample`. Can also run a background thread for
    automatic periodic sampling.

    Args:
        pid: Process ID to monitor. 0 (default) = current process.
        children: If True, also monitor child processes by scanning
            GPU clients for matching parent PIDs.

    Example::

        mon = GpuMonitor()
        for batch in dataloader:
            train(batch)
            print(f"GPU: {mon.sample():.1f}%")
    """

    def __init__(self, pid: int = 0, children: bool = False) -> None:
        self.pid = pid
        self.children = children
        self._last_ns: int | None = None
        self._last_time: float | None = None
        self._samples: list[float] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def __enter__(self) -> GpuMonitor:
        self.reset()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def _collect_pids(self) -> list[int]:
        """Return list of PIDs to sample (self + children if enabled)."""
        pid = self.pid if self.pid != 0 else os.getpid()
        pids = [pid]
        if self.children:
            # Only include GPU clients whose parent PID matches target
            for c in gpu_clients():
                cpid = c["pid"]
                if cpid != pid and cpid not in pids and ppid(cpid) == pid:
                    pids.append(cpid)
        return pids

    def _read_total_ns(self) -> int:
        """Read total GPU ns across all tracked PIDs."""
        pids = self._collect_pids()
        if len(pids) == 1:
            return gpu_time_ns(pids[0])
        results = gpu_time_ns_multi(pids)
        return sum(results.values())

    def sample(self) -> float:
        """Compute GPU utilization since the last call to sample().

        Returns:
            GPU utilization percentage (0.0 - 100.0).
            First call returns 0.0 (no prior sample to diff against).
        """
        now = _time.monotonic()
        ns = self._read_total_ns()

        if self._last_ns is None or self._last_time is None:
            self._last_ns = ns
            self._last_time = now
            return 0.0

        delta_ns = ns - self._last_ns
        elapsed_s = now - self._last_time
        self._last_ns = ns
        self._last_time = now

        if elapsed_s <= 0:
            return 0.0

        pct = min((delta_ns / (elapsed_s * 1_000_000_000)) * 100.0, 100.0)
        with self._lock:
            self._samples.append(pct)
        return pct

    def reset(self) -> None:
        """Reset the monitor state."""
        self._last_ns = None
        self._last_time = None
        with self._lock:
            self._samples.clear()

    def start(self, interval: float = 2.0) -> None:
        """Start background sampling thread.

        Args:
            interval: Seconds between samples.
        """
        if self._thread is not None:
            return
        self.reset()
        # Take initial sample so first background sample has a baseline
        self.sample()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._bg_loop, args=(interval,), daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop background sampling thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _bg_loop(self, interval: float) -> None:
        while not self._stop.wait(interval):
            self.sample()

    @property
    def last(self) -> float:
        """Most recent GPU utilization sample, or 0.0 if none."""
        with self._lock:
            return self._samples[-1] if self._samples else 0.0

    def summary(self) -> dict[str, float]:
        """Aggregate statistics from all samples.

        Returns:
            Dict with keys: gpu_pct_avg, gpu_pct_min, gpu_pct_max,
            num_samples.
        """
        with self._lock:
            if not self._samples:
                return {"gpu_pct_avg": 0, "gpu_pct_min": 0, "gpu_pct_max": 0, "num_samples": 0}
            samples = list(self._samples)
        return {
            "gpu_pct_avg": sum(samples) / len(samples),
            "gpu_pct_min": min(samples),
            "gpu_pct_max": max(samples),
            "num_samples": len(samples),
        }
