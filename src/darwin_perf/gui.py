"""darwin-perf GUI: Native floating window GPU monitor.

A compact, resizable native window showing live per-process GPU utilization,
CPU, memory, and history charts. Designed to tuck in a corner while training.

Uses pywebview for a native macOS window (no browser chrome).

Usage:
    darwin-perf --gui
    darwin-perf --gui -i 1
"""

from __future__ import annotations

import json
import os
import threading
import time

_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #0f172a; color: #e2e8f0; font-family: -apple-system, system-ui, sans-serif;
    font-size: 12px; padding: 8px; overflow: hidden; height: 100vh;
    display: flex; flex-direction: column;
}
.header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 4px 0 8px; border-bottom: 1px solid #1e293b; margin-bottom: 8px;
    flex-shrink: 0;
}
.title { font-size: 13px; font-weight: 600; color: #94a3b8; }
.stats { font-size: 11px; color: #64748b; }
.stats b { color: #10b981; }

.gpu-section { flex-shrink: 0; margin-bottom: 8px; }
.section-label { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.bar-row { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }
.bar-label { font-size: 11px; color: #94a3b8; width: 50px; flex-shrink: 0; }
.bar-track { flex: 1; height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }
.bar-value { font-size: 11px; font-family: 'SF Mono', 'Menlo', monospace; color: #94a3b8; width: 55px; text-align: right; flex-shrink: 0; }

.fill-gpu { background: #8b5cf6; }
.fill-cpu { background: #64748b; }
.fill-mem { background: #10b981; }

canvas {
    flex: 1; min-height: 60px; width: 100%; border-radius: 4px;
    background: #1e293b; margin-top: 4px;
}
.proc-list { flex: 1; overflow-y: auto; min-height: 0; margin-top: 6px; }
.proc-row {
    display: flex; align-items: center; gap: 6px; padding: 2px 0;
    border-bottom: 1px solid #1e293b20;
}
.proc-name { font-size: 11px; color: #cbd5e1; width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.proc-pid { font-size: 10px; color: #475569; width: 50px; text-align: right; font-family: monospace; }
.proc-bar { flex: 1; height: 4px; background: #1e293b; border-radius: 2px; overflow: hidden; }
.proc-fill-gpu { height: 100%; background: #8b5cf6; border-radius: 2px; transition: width 0.5s; }
.proc-val { font-size: 10px; color: #94a3b8; width: 50px; text-align: right; font-family: monospace; }
</style>
</head>
<body>
<div class="header">
    <span class="title">darwin-perf</span>
    <span class="stats" id="stats">--</span>
</div>

<div class="gpu-section">
    <div class="bar-row">
        <span class="bar-label">GPU</span>
        <div class="bar-track"><div class="bar-fill fill-gpu" id="gpu-bar" style="width:0%"></div></div>
        <span class="bar-value" id="gpu-val">--%</span>
    </div>
    <div class="bar-row">
        <span class="bar-label">CPU</span>
        <div class="bar-track"><div class="bar-fill fill-cpu" id="cpu-bar" style="width:0%"></div></div>
        <span class="bar-value" id="cpu-val">--%</span>
    </div>
    <div class="bar-row">
        <span class="bar-label">RAM</span>
        <div class="bar-track"><div class="bar-fill fill-mem" id="ram-bar" style="width:0%"></div></div>
        <span class="bar-value" id="ram-val">--</span>
    </div>
</div>

<div class="section-label">GPU History</div>
<canvas id="chart"></canvas>

<div class="section-label" style="margin-top:6px">Top Processes (GPU)</div>
<div class="proc-list" id="procs"></div>

<script>
const gpuHistory = [];
const MAX_HIST = 120;
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');

function drawChart() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    if (gpuHistory.length < 2) return;

    const maxV = Math.max(100, ...gpuHistory);
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let i = 0; i < gpuHistory.length; i++) {
        const x = (i / (MAX_HIST - 1)) * w;
        const y = h - (gpuHistory[i] / maxV) * h;
        ctx.lineTo(x, y);
    }
    ctx.lineTo((gpuHistory.length - 1) / (MAX_HIST - 1) * w, h);
    ctx.closePath();
    ctx.fillStyle = 'rgba(139, 92, 246, 0.15)';
    ctx.fill();

    ctx.beginPath();
    for (let i = 0; i < gpuHistory.length; i++) {
        const x = (i / (MAX_HIST - 1)) * w;
        const y = h - (gpuHistory[i] / maxV) * h;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#8b5cf6';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Grid lines
    ctx.strokeStyle = '#1e293b80';
    ctx.lineWidth = 0.5;
    for (let pct of [25, 50, 75]) {
        const y = h - (pct / maxV) * h;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
}

function update(data) {
    const gpu = data.total_gpu_pct ?? 0;
    const cpu = data.total_cpu_pct ?? 0;
    const ramPct = data.memory_total_gb > 0 ? (data.memory_used_gb / data.memory_total_gb) * 100 : 0;

    document.getElementById('gpu-bar').style.width = Math.min(gpu, 100) + '%';
    document.getElementById('gpu-val').textContent = gpu.toFixed(1) + '%';
    document.getElementById('cpu-bar').style.width = Math.min(cpu, 100) + '%';
    document.getElementById('cpu-val').textContent = cpu.toFixed(0) + '%';
    document.getElementById('ram-bar').style.width = ramPct + '%';
    document.getElementById('ram-val').textContent = data.memory_used_gb.toFixed(0) + '/' + data.memory_total_gb.toFixed(0) + 'GB';

    // Stats
    document.getElementById('stats').innerHTML = '<b>' + data.gpu_clients + ' GPU clients</b>';

    // History
    gpuHistory.push(gpu);
    if (gpuHistory.length > MAX_HIST) gpuHistory.shift();
    drawChart();

    // Process list (sorted by GPU%)
    const procs = data.processes || [];
    const el = document.getElementById('procs');
    el.innerHTML = procs.slice(0, 15).map(p =>
        '<div class="proc-row">' +
        '<span class="proc-pid">' + p.pid + '</span>' +
        '<span class="proc-name">' + p.name + '</span>' +
        '<div class="proc-bar"><div class="proc-fill-gpu" style="width:' + Math.min(p.gpu, 100) + '%"></div></div>' +
        '<span class="proc-val">' + p.gpu.toFixed(1) + '% gpu</span>' +
        '<span class="proc-val">' + p.cpu.toFixed(0) + '% cpu</span>' +
        '<span class="proc-val">' + p.mem.toFixed(0) + 'MB</span>' +
        '</div>'
    ).join('');
}

window.addEventListener('resize', drawChart);
</script>
</body>
</html>"""




class _GpuGuiApi:
    """Python API exposed to the webview JS context."""

    def __init__(self, interval: float = 2.0) -> None:
        self.interval = interval
        self._window = None
        self._prev_snap: dict[int, dict] | None = None
        self._prev_cpu: dict[int, int] = {}
        self._prev_time: float = 0

    def set_window(self, window: object) -> None:
        self._window = window

    def start_polling(self) -> None:
        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()

    def _collect(self) -> dict:
        from darwin_perf import _snapshot
        from darwin_perf._native import cpu_time_ns, proc_info

        now = time.monotonic()
        snap = _snapshot()

        curr_cpu: dict[int, int] = {}
        for pid in snap:
            ns = cpu_time_ns(pid)
            curr_cpu[pid] = ns if ns >= 0 else 0

        data: dict = {"processes": [], "gpu_clients": len(snap)}

        if self._prev_snap is not None:
            elapsed_s = now - self._prev_time
            elapsed_ns = elapsed_s * 1_000_000_000 if elapsed_s > 0 else 1

            procs = []
            total_gpu = 0.0
            total_cpu = 0.0
            for pid, info in snap.items():
                gpu_delta = info["gpu_ns"] - self._prev_snap.get(pid, {}).get("gpu_ns", info["gpu_ns"])
                cpu_delta = curr_cpu.get(pid, 0) - self._prev_cpu.get(pid, curr_cpu.get(pid, 0))

                gpu_pct = min(gpu_delta / elapsed_ns * 100, 100) if elapsed_ns > 0 else 0
                cpu_pct = cpu_delta / elapsed_ns * 100 if elapsed_ns > 0 else 0

                pinfo = proc_info(pid)
                mem_mb = pinfo["real_memory"] / (1024 * 1024) if pinfo else 0

                total_gpu += gpu_pct
                total_cpu += cpu_pct

                if gpu_pct >= 0.1 or gpu_delta > 0:
                    procs.append({
                        "pid": pid,
                        "name": info["name"],
                        "gpu": round(gpu_pct, 1),
                        "cpu": round(cpu_pct, 1),
                        "mem": round(mem_mb, 0),
                    })

            procs.sort(key=lambda p: p["gpu"], reverse=True)
            data["processes"] = procs[:15]
            data["total_gpu_pct"] = round(total_gpu, 1)
            data["total_cpu_pct"] = round(total_cpu, 1)
        else:
            data["total_gpu_pct"] = 0
            data["total_cpu_pct"] = 0

        # System memory via Mach APIs (no subprocess)
        from darwin_perf._native import system_stats
        sys = system_stats()
        GB = 1024**3
        data["memory_total_gb"] = round(sys.get("memory_total", 0) / GB, 1)
        data["memory_used_gb"] = round(sys.get("memory_used", 0) / GB, 1)

        self._prev_snap = snap
        self._prev_cpu = curr_cpu
        self._prev_time = now
        return data

    def _poll_loop(self) -> None:
        # Take initial baseline
        from darwin_perf import _snapshot
        from darwin_perf._native import cpu_time_ns
        self._prev_snap = _snapshot()
        for pid in self._prev_snap:
            ns = cpu_time_ns(pid)
            self._prev_cpu[pid] = ns if ns >= 0 else 0
        self._prev_time = time.monotonic()
        time.sleep(self.interval)

        while True:
            try:
                data = self._collect()
                if self._window:
                    js = f"update({json.dumps(data)})"
                    self._window.evaluate_js(js)
            except Exception as e:
                print(f"Poll error: {e}", file=__import__('sys').stderr)
            time.sleep(self.interval)


def run_gui(interval: float = 2.0, width: int = 380, height: int = 520) -> None:
    """Launch native floating GPU monitor window."""
    import webview  # type: ignore

    api = _GpuGuiApi(interval=interval)
    window = webview.create_window(
        "darwin-perf",
        html=_HTML,
        width=width,
        height=height,
        resizable=True,
        on_top=True,
        frameless=False,
    )
    api.set_window(window)
    window.events.loaded += api.start_polling
    webview.start()
