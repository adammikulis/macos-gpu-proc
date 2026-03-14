# darwin-perf

System performance monitoring for macOS Apple Silicon — GPU, CPU, memory, energy, temperature, and disk I/O via Mach APIs, IORegistry, and AppleSMC. **No sudo needed.**

## Install

```bash
pip install darwin-perf
```

## Quick Start

### Temperatures (instant, no sudo)

```python
from darwin_perf import temperatures

t = temperatures()
print(f"CPU: {t['cpu_avg']:.1f}°C  GPU: {t['gpu_avg']:.1f}°C  System: {t['system_avg']:.1f}°C")
# CPU: 40.1°C  GPU: 36.5°C  System: 31.2°C

# Individual sensors
for name, temp in t['cpu_sensors'].items():
    print(f"  {name}: {temp:.1f}°C")
```

### CPU Power & Frequency

```python
from darwin_perf import cpu_power

c = cpu_power(interval=0.5)
print(f"CPU Power: {c['cpu_power_w']:.2f}W")
for name, cluster in c['clusters'].items():
    print(f"  {name}: {cluster['freq_mhz']} MHz, {cluster['active_pct']:.0f}% active")
# ECPU: 663 MHz, 100% active
# PCPU: 1272 MHz, 100% active
```

### GPU Power & Frequency

```python
from darwin_perf import gpu_power

g = gpu_power(interval=0.5)
print(f"GPU: {g['gpu_power_w']:.2f}W  {g['gpu_freq_mhz']}MHz  throttled={g['throttled']}")
for state in g['frequency_states']:
    print(f"  {state['state']}: {state.get('freq_mhz', '?')}MHz ({state['residency_pct']:.1f}%)")
```

### Per-Process GPU/CPU Utilization

```python
from darwin_perf import snapshot

for proc in snapshot():
    print(f"{proc['name']:20s}  GPU {proc['gpu_percent']:5.1f}%  "
          f"CPU {proc['cpu_percent']:5.1f}%  {proc['memory_mb']:.0f}MB  "
          f"{proc['energy_w']:.1f}W")
```

### Full System Snapshot (everything in one call)

```python
from darwin_perf import snapshot

s = snapshot(interval=1.0, system=True, detailed=True)
# s['cpu']          — cpu_power_w, clusters (ECPU/PCPU freq + residency)
# s['gpu']          — gpu_power_w, gpu_freq_mhz, throttled, frequency_states
# s['temperatures'] — cpu_avg, gpu_avg, system_avg, per-sensor dicts
# s['memory']       — memory_total, memory_used, memory_available, ...
# s['gpu_stats']    — device_utilization, model, gpu_core_count
# s['processes']    — per-process GPU%, CPU%, memory, energy, IPC, disk I/O
```

### System-Wide Stats

```python
from darwin_perf import system_stats, system_gpu_stats

sys = system_stats()     # instant, ~3µs
print(f"RAM: {sys['memory_used']/1e9:.1f} / {sys['memory_total']/1e9:.1f} GB")
print(f"CPU idle: {sys['cpu_idle_pct']:.1f}%")

gpu = system_gpu_stats()
print(f"{gpu['model']} ({gpu['gpu_core_count']} cores)")
print(f"Device utilization: {gpu['device_utilization']}%")
```

### GpuMonitor (continuous monitoring)

```python
from darwin_perf import GpuMonitor

mon = GpuMonitor()  # monitors the current process
for batch in dataloader:
    train(batch)
    print(f"GPU: {mon.sample():.1f}%")

# Or as a context manager with background sampling:
with GpuMonitor() as mon:
    mon.start(interval=2.0)
    train()
print(mon.summary())  # {'gpu_pct_avg': 42.1, 'gpu_pct_max': 87.3, ...}
```

### Low-Level Access

```python
from darwin_perf import gpu_clients, proc_info

# All GPU clients (raw cumulative data) — includes Metal/GL/CL API type
for c in gpu_clients():
    print(f"PID {c['pid']} ({c['name']}): {c['gpu_ns']/1e9:.1f}s GPU time [{c['api']}]")

# Per-process stats (CPU, memory, energy, disk I/O, threads)
info = proc_info(1234)
print(f"Memory: {info['memory']/1e6:.0f}MB, Energy: {info['energy_nj']/1e9:.1f}J")
```

## CLI

```bash
# Live monitoring
darwin-perf              # per-process GPU/CPU monitor
darwin-perf --tui        # rich terminal UI (pip install darwin-perf[tui])
darwin-perf --gui        # floating window (pip install darwin-perf[gui])
darwin-perf -i 1         # 1-second updates
darwin-perf --pid 1234   # specific PID

# Streaming output
darwin-perf --json       # one JSON line per update (pipe to jq, etc.)
darwin-perf --csv        # CSV with header (pipe to file for spreadsheets)

# Recording & export
darwin-perf --record session.jsonl         # capture full system state
darwin-perf --record session.jsonl -n 60   # record 60 samples
darwin-perf --export session.jsonl         # → session_system.csv + session_processes.csv
darwin-perf --replay session.jsonl         # replay with original timing
```

### Record → Export workflow

Record a training run, then analyze in a spreadsheet or pandas:

```bash
# 1. Record during workload
darwin-perf --record training.jsonl -i 2

# 2. Export to CSV
darwin-perf --export training.jsonl

# 3. Produces:
#    training_system.csv    — CPU/GPU power, temps, memory per sample
#    training_processes.csv — per-process GPU%, CPU%, memory, IPC per sample
```

```python
import pandas as pd
df = pd.read_csv("training_system.csv")
df.plot(x="epoch", y=["cpu_power_w", "gpu_power_w", "temp_cpu_avg", "temp_gpu_avg"])
```

## API Reference

### Python API

| Function | Description |
|----------|-------------|
| `snapshot()` | Per-process GPU/CPU utilization, memory, energy |
| `snapshot(system=True)` | Full system: CPU/GPU power, temps, memory + processes |
| `snapshot(detailed=True)` | Extended fields: IPC, wakeups, peak memory, disk I/O |
| `temperatures()` | Instant thermal sensors via AppleSMC (CPU, GPU, system) |
| `cpu_power(interval)` | CPU power (W), ECPU/PCPU frequency + P-state residency |
| `gpu_power(interval)` | GPU power (W), frequency (MHz), throttle, P-state residency |
| `system_stats()` | System memory + CPU ticks (instant, ~3µs) |
| `system_gpu_stats()` | GPU utilization %, model, core count, VRAM |
| `GpuMonitor` | Continuous per-process GPU % with background thread |

### C Extension Functions

| Function | Description |
|----------|-------------|
| `gpu_clients()` | All GPU-active processes: `[{pid, name, gpu_ns, api}, ...]` |
| `gpu_time_ns(pid)` | Cumulative GPU nanoseconds for a PID |
| `gpu_time_ns_multi(pids)` | Batch GPU ns for multiple PIDs (single IORegistry scan) |
| `cpu_time_ns(pid)` | Cumulative CPU nanoseconds (user + system) |
| `proc_info(pid)` | Full process stats (CPU, memory, energy, disk, threads) |
| `temperatures()` | Thermal sensors via AppleSMC |
| `cpu_power(interval)` | CPU package power + per-cluster frequency states |
| `gpu_power(interval)` | GPU power, frequency, throttle, temperature |
| `gpu_freq_table()` | GPU DVFS frequency table (MHz per P-state) |
| `system_stats()` | System memory + CPU via Mach APIs |
| `system_gpu_stats()` | GPU performance statistics from IORegistry |
| `ppid(pid)` | Parent process ID (-1 on error) |

## How It Works

**GPU per-process metrics**: Apple doesn't provide a public API for this. The data exists in the IORegistry — every Metal command queue creates an `AGXDeviceUserClient` with `accumulatedGPUTime` in nanoseconds. Sample twice, divide by elapsed time = utilization %. World-readable, no sudo.

**CPU/GPU power & frequency**: Uses `libIOReport.dylib` (same data source as `powermetrics`). Subscribes to "Energy Model" and "CPU Stats"/"GPU Stats" groups. No sudo.

**Temperatures**: Reads SMC keys (`Tp*`=CPU, `Tg*`=GPU, `Ts*`=system) via `IOServiceOpen("AppleSMC")` + `IOConnectCallStructMethod`. 48+ sensors on M4 Max. No sudo.

**System stats**: `host_statistics64()` for memory, `host_statistics(HOST_CPU_LOAD_INFO)` for CPU ticks, `proc_pid_rusage(RUSAGE_INFO_V6)` for per-process metrics. All unprivileged.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.9+
- Zero dependencies

## License

MIT
