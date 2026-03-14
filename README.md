# darwin-perf

System performance monitoring for macOS Apple Silicon — GPU, CPU, memory, energy, and disk I/O via Mach kernel APIs. **No sudo needed.**

Reads GPU client data directly from the IORegistry — the same data source Activity Monitor uses. Auto-discovers every process using the GPU.

## Install

```bash
pip install darwin-perf
```

## Quick Start

```python
from darwin_perf import snapshot

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

### GPU Power & Frequency

```python
from darwin_perf import gpu_power

power = gpu_power(interval=1.0)  # samples over 1 second
print(f"GPU Power: {power['gpu_power_w']:.2f}W")
print(f"GPU Freq:  {power['gpu_freq_mhz']} MHz (weighted avg)")
print(f"Throttled: {power['throttled']}")
for state in power['frequency_states']:
    print(f"  {state['state']}: {state['freq_mhz']}MHz ({state['residency_pct']:.1f}%)")
```

Uses `libIOReport.dylib` (the same data source as `powermetrics`). No sudo needed.

### GPU DVFS Frequency Table

```python
from darwin_perf import gpu_freq_table

for i, freq in enumerate(gpu_freq_table()):
    print(f"P{i+1}: {freq} MHz")
# P1: 338 MHz, P2: 618 MHz, ..., P15: 1578 MHz
```

### System-Wide Stats

```python
from darwin_perf import system_stats, system_gpu_stats

# System memory + CPU (instant, ~3µs)
sys = system_stats()
print(f"RAM: {sys['memory_used']/1e9:.1f} / {sys['memory_total']/1e9:.1f} GB")
print(f"CPU idle: {sys['cpu_idle_pct']:.1f}%")

# GPU utilization + model info
gpu = system_gpu_stats()
print(f"{gpu['model']} ({gpu['gpu_core_count']} cores)")
print(f"Device utilization: {gpu['device_utilization']}%")
print(f"GPU VRAM in use: {gpu['in_use_system_memory']/1e9:.1f}GB")
```

### GpuMonitor (continuous monitoring)

Monitor your own training process — no PID lookup needed:

```python
from darwin_perf import GpuMonitor

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
from darwin_perf import gpu_clients, gpu_time_ns, proc_info

# All GPU clients (raw cumulative data)
for c in gpu_clients():
    print(f"PID {c['pid']} ({c['name']}): {c['gpu_ns']/1e9:.1f}s GPU time")

# Per-process stats (CPU, memory, energy, disk I/O, threads)
info = proc_info(1234)
print(f"Memory: {info['memory']/1e6:.0f}MB, Energy: {info['energy_nj']/1e9:.1f}J")
```

## CLI

```bash
darwin-perf              # live per-process GPU monitor — auto-discovers all GPU processes
darwin-perf --once       # single snapshot
darwin-perf --tui        # rich terminal UI with sparkline graphs (pip install darwin-perf[tui])
darwin-perf --gui        # native floating window monitor (pip install darwin-perf[gui])
darwin-perf -i 1         # 1-second update interval
darwin-perf --pid 1234   # monitor specific PID
python -m darwin_perf    # alternative entry point (same as darwin-perf)
```

## API Reference

### Python API

| Function | Description |
|----------|-------------|
| `snapshot(interval=1.0)` | **One call does it all** — returns `[{'pid', 'name', 'gpu_percent', 'cpu_percent', 'memory_mb', 'energy_w', ...}]` |
| `snapshot(detailed=True)` | Extended fields: IPC, wakeups, peak memory, neural engine, disk I/O |
| `system_stats()` | System-wide memory + CPU ticks (instant, ~3µs) |

### C Extension Functions

| Function | Description |
|----------|-------------|
| `gpu_clients()` | Auto-discover all GPU-active processes: `[{'pid', 'name', 'gpu_ns'}, ...]` |
| `gpu_time_ns(pid)` | Cumulative GPU nanoseconds for a PID |
| `gpu_time_ns_multi(pids)` | Batch GPU ns for multiple PIDs (single IORegistry scan) |
| `cpu_time_ns(pid)` | Cumulative CPU nanoseconds (user + system) |
| `proc_info(pid)` | Full process stats (CPU, memory, energy, disk, threads) |
| `system_stats()` | System-wide memory + CPU ticks via Mach APIs |
| `system_gpu_stats()` | System GPU: utilization %, VRAM, model, core count |
| `gpu_power(interval)` | GPU power (watts), frequency (MHz), P-state residency, thermal throttling |
| `gpu_freq_table()` | GPU DVFS frequency table (MHz per P-state) from pmgr |
| `ppid(pid)` | Parent process ID for a PID (-1 on error) |

### proc_info fields

| Field | Description |
|-------|-------------|
| **CPU** | |
| `cpu_ns` | Cumulative CPU time (user + system) in nanoseconds |
| `cpu_user_ns` | User CPU time |
| `cpu_system_ns` | System/kernel CPU time |
| `instructions` | Retired instructions (for IPC calculation) |
| `cycles` | CPU cycles (for IPC calculation) |
| `runnable_time` | Time process was runnable but not running (ns) |
| `billed_system_time` | Billed CPU time (ns) |
| `serviced_system_time` | Serviced CPU time (ns) |
| **Memory** | |
| `memory` | Physical memory footprint (bytes) |
| `real_memory` | Resident memory (bytes) |
| `wired_size` | Wired (non-pageable) memory (bytes) |
| `peak_memory` | Lifetime peak physical footprint (bytes) |
| `neural_footprint` | Neural Engine memory (bytes) |
| `pageins` | Page-in count (memory pressure indicator) |
| **Disk** | |
| `disk_read_bytes` | Cumulative disk reads |
| `disk_write_bytes` | Cumulative disk writes |
| `logical_writes` | Logical writes including CoW (bytes) |
| **Energy** | |
| `energy_nj` | Cumulative energy (nanojoules) — delta over time = watts |
| `idle_wakeups` | Package idle wakeups (energy efficiency metric) |
| `interrupt_wakeups` | Interrupt wakeups |
| **Other** | |
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
| `in_use_system_memory_driver` | Driver-side in-use memory |
| `allocated_pb_size` | Parameter buffer allocation (bytes) |
| `recovery_count` | GPU recovery (crash) count |
| `last_recovery_time` | Timestamp of last GPU recovery |
| `split_scene_count` | Tiler split scene events |
| `tiled_scene_bytes` | Current tiled scene buffer size |

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
