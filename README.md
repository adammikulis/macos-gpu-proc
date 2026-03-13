# macos-gpu-proc

Per-process GPU utilization, CPU, memory, and energy monitoring for macOS Apple Silicon. **No sudo needed.**

Reads GPU client data directly from the IORegistry — the same data source Activity Monitor uses. Auto-discovers every process using the GPU.

## Install

```bash
pip install macos-gpu-proc
```

## Quick Start

```python
from macos_gpu_proc import snapshot

# One call — auto-discovers all GPU processes, returns utilization %
for proc in snapshot():
    print(f"{proc['name']:20s}  GPU {proc['gpu_percent']:5.1f}%  "
          f"CPU {proc['cpu_percent']:5.1f}%  {proc['memory_mb']:.0f}MB  "
          f"{proc['energy_w']:.1f}W")

# Example output:
#   python3.12            GPU  85.7%  CPU 102.3%  2048MB  12.3W
#   WindowServer          GPU   3.2%  CPU   0.1%    45MB   0.4W
#   Code Helper (GPU      GPU   1.9%  CPU   0.1%   670MB   0.1W
```

Each dict in the list contains: `pid`, `name`, `gpu_percent`, `cpu_percent`, `memory_mb`, `energy_w`, `threads`, and `gpu_ns`.

### System-Wide GPU

```python
from macos_gpu_proc import system_gpu_stats

stats = system_gpu_stats()
print(f"{stats['model']} ({stats['gpu_core_count']} cores)")
print(f"Device utilization: {stats['device_utilization']}%")
print(f"GPU VRAM in use: {stats['in_use_system_memory']/1e9:.1f}GB")
```

### GpuMonitor (continuous monitoring)

Monitor your own training process — no PID lookup needed:

```python
from macos_gpu_proc import GpuMonitor

mon = GpuMonitor()  # monitors the current process
for batch in dataloader:
    train(batch)
    print(f"GPU: {mon.sample():.1f}%")

# Or as a context manager:
with GpuMonitor() as mon:
    mon.start(interval=2.0)  # background sampling
    train()
print(mon.summary())  # {'gpu_pct_avg': 42.1, 'gpu_pct_max': 87.3, ...}
```

### Low-Level Access

```python
from macos_gpu_proc import gpu_clients, gpu_time_ns, proc_info

# All GPU clients (raw cumulative data)
for c in gpu_clients():
    print(f"PID {c['pid']} ({c['name']}): {c['gpu_ns']/1e9:.1f}s GPU time")

# Per-process stats (CPU, memory, energy, disk I/O, threads)
info = proc_info(1234)
print(f"Memory: {info['memory']/1e6:.0f}MB, Energy: {info['energy_nj']/1e9:.1f}J")
```

## CLI

```bash
gpu-proc              # live per-process GPU monitor — auto-discovers all GPU processes
gpu-proc --once       # single snapshot
gpu-proc --tui        # rich terminal UI with sparkline graphs (pip install macos-gpu-proc[tui])
gpu-proc --gui        # native floating window monitor (pip install macos-gpu-proc[gui])
gpu-proc -i 1         # 1-second update interval
gpu-proc --pid 1234   # monitor specific PID
python -m macos_gpu_proc  # alternative entry point (same as gpu-proc)
```

## API Reference

### Python API

| Function | Description |
|----------|-------------|
| `snapshot(interval=1.0)` | **One call does it all** — returns `[{'pid', 'name', 'gpu_percent', 'cpu_percent', 'memory_mb', 'energy_w', ...}]` |

### C Extension Functions

| Function | Description |
|----------|-------------|
| `gpu_clients()` | Auto-discover all GPU-active processes: `[{'pid', 'name', 'gpu_ns'}, ...]` |
| `gpu_time_ns(pid)` | Cumulative GPU nanoseconds for a PID |
| `gpu_time_ns_multi(pids)` | Batch GPU ns for multiple PIDs (single IORegistry scan) |
| `cpu_time_ns(pid)` | Cumulative CPU nanoseconds (user + system) |
| `proc_info(pid)` | Full process stats (CPU, memory, energy, disk, threads) |
| `system_gpu_stats()` | System GPU: utilization %, VRAM, model, core count |
| `ppid(pid)` | Parent process ID for a PID (-1 on error) |

### proc_info fields

| Field | Description |
|-------|-------------|
| `cpu_ns` | Cumulative CPU time (user + system) in nanoseconds |
| `cpu_user_ns` | User CPU time |
| `cpu_system_ns` | System/kernel CPU time |
| `memory` | Physical memory footprint (bytes) |
| `real_memory` | Resident memory (bytes) |
| `neural_footprint` | Neural Engine memory (bytes) |
| `disk_read_bytes` | Cumulative disk reads |
| `disk_write_bytes` | Cumulative disk writes |
| `energy_nj` | Cumulative energy (nanojoules) — delta over time = watts |
| `threads` | Current thread count |

### system_gpu_stats fields

| Field | Description |
|-------|-------------|
| `model` | GPU model name (e.g., "Apple M4 Max") |
| `gpu_core_count` | Number of GPU cores |
| `device_utilization` | Device utilization % (0-100) |
| `tiler_utilization` | Tiler utilization % |
| `renderer_utilization` | Renderer utilization % |
| `alloc_system_memory` | Total GPU-allocated system memory |
| `in_use_system_memory` | Currently used GPU memory |

## How It Works

Apple doesn't provide a public API for per-process GPU metrics on Apple Silicon. The commonly referenced `task_info(TASK_POWER_INFO_V2)` has a `task_gpu_utilisation` field, but Apple never populates it — it always returns 0.

The data *does* exist, in the IORegistry. Every Metal command queue creates an `AGXDeviceUserClient` object as a child of the GPU accelerator. You can see them with:

```bash
ioreg -c AGXDeviceUserClient -r -d 0
```

Each entry carries:

```
"IOUserClientCreator" = "pid 4245, python3.12"
"AppUsage" = ({"API"="Metal", "accumulatedGPUTime"=123632000000})
```

`accumulatedGPUTime` is cumulative GPU nanoseconds — sample twice, divide by elapsed time, and you have utilization %. This is world-readable, no sudo or SIP changes needed.

**The catch:** `IOServiceGetMatchingServices("AGXDeviceUserClient")` returns 0 results because user client objects are `!registered` in the IOKit matching system. You have to find the parent accelerator first and walk its children:

```c
// Find the AGX accelerator
IOServiceGetMatchingServices(kIOMainPortDefault,
    IOServiceMatching("AGXAccelerator"), &iter);

// Iterate its children in the IOService plane
io_service_t accel = IOIteratorNext(iter);
IORegistryEntryGetChildIterator(accel, kIOServicePlane, &child_iter);

// Each child is an AGXDeviceUserClient with AppUsage data
while ((child = IOIteratorNext(child_iter))) {
    CFStringRef creator = IORegistryEntryCreateCFProperty(child,
        CFSTR("IOUserClientCreator"), ...);  // "pid 4245, python3.12"
    CFArrayRef usage = IORegistryEntryCreateCFProperty(child,
        CFSTR("AppUsage"), ...);             // [{accumulatedGPUTime: ns}]
}
```

System-wide GPU utilization comes from the accelerator's `PerformanceStatistics` property (`Device Utilization %`, `Tiler Utilization %`, `Renderer Utilization %`).

CPU, memory, energy, disk I/O, and thread stats come from `proc_pid_rusage(RUSAGE_INFO_V6)` and `proc_pidinfo(PROC_PIDTASKINFO)` — both unprivileged for same-user processes.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.9+
- Zero dependencies

## License

MIT
