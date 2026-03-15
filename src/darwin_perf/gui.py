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
import threading
import time

from ._sampler import PowerSampler

_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #0f172a; color: #e2e8f0; font-family: -apple-system, system-ui, sans-serif;
    font-size: 12px; padding: 8px; overflow-y: auto; overflow-x: hidden; height: 100vh;
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
.section-label {
    font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 4px; cursor: pointer; user-select: none;
}
.section-label:hover { color: #94a3b8; }
.bar-row { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }
.bar-label { font-size: 11px; color: #94a3b8; width: 50px; flex-shrink: 0; }
.bar-track { flex: 1; height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }
.bar-value { font-size: 11px; font-family: 'SF Mono', 'Menlo', monospace; color: #94a3b8; width: 55px; text-align: right; flex-shrink: 0; }

.fill-gpu { background: #8b5cf6; }
.fill-cpu { background: #64748b; }
.fill-mem { background: #10b981; }
.fill-tiler { background: #06b6d4; }
.fill-renderer { background: #ec4899; }
.fill-active { background: #22c55e; }
.fill-inactive { background: #eab308; }
.fill-wired { background: #ef4444; }
.fill-compressed { background: #06b6d4; }

canvas {
    flex-shrink: 0; min-height: 60px; max-height: 80px; width: 100%; border-radius: 4px;
    background: #1e293b; margin-top: 4px;
}
.proc-list { flex: 1; overflow-y: auto; min-height: 0; margin-top: 6px; }
.proc-row {
    display: flex; align-items: center; gap: 6px; padding: 2px 0;
    border-bottom: 1px solid #1e293b20; cursor: pointer;
}
.proc-row:hover { background: #1e293b40; }
.proc-name { font-size: 11px; color: #cbd5e1; width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.proc-pid { font-size: 10px; color: #475569; width: 50px; text-align: right; font-family: monospace; }
.proc-bar { flex: 1; height: 4px; background: #1e293b; border-radius: 2px; overflow: hidden; }
.proc-fill-gpu { height: 100%; background: #8b5cf6; border-radius: 2px; transition: width 0.5s; }
.proc-val { font-size: 10px; color: #94a3b8; width: 50px; text-align: right; font-family: monospace; }
.proc-detail {
    font-size: 10px; color: #64748b; padding: 2px 0 4px 56px; line-height: 1.5;
    display: none; border-bottom: 1px solid #1e293b40;
}

.collapsible { display: none; margin-bottom: 8px; }
.collapsible.open { display: block; }

.detail-row { font-size: 11px; color: #94a3b8; padding: 1px 0; }
.detail-label { color: #64748b; display: inline-block; width: 80px; }
.detail-val { color: #e2e8f0; font-family: 'SF Mono', 'Menlo', monospace; }
.pstate-bar { display: inline-block; height: 8px; border-radius: 2px; margin-right: 1px; }
.throttle-badge { color: #ef4444; font-weight: 600; font-size: 10px; }

/* Responsive two-column layout */
@media (min-width: 600px) {
    body { flex-direction: row; flex-wrap: wrap; gap: 12px; }
    .header { width: 100%; flex-shrink: 0; }
    .col-left { flex: 1; min-width: 280px; display: flex; flex-direction: column; }
    .col-right { flex: 1; min-width: 280px; display: flex; flex-direction: column; overflow-y: auto; }
    .col-right .collapsible { display: block !important; }
}
</style>
</head>
<body>
<div class="header">
    <span class="title">darwin-perf</span>
    <span class="stats" id="stats">--</span>
</div>

<div class="col-left">
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
    <div class="bar-row" style="cursor:pointer" onclick="toggleSensors()">
        <span class="bar-label">Temps</span>
        <span class="bar-value" id="temp-val" style="width:auto;flex:1;text-align:left;color:#f59e0b">--</span>
        <span id="temp-toggle" style="color:#475569;font-size:10px">&#x25B6; sensors</span>
    </div>
    <div id="sensor-detail" style="display:none;font-size:10px;color:#94a3b8;padding:2px 0 4px 56px;line-height:1.6"></div>
</div>

<div class="section-label">GPU History</div>
<canvas id="chart"></canvas>

<div class="section-label" style="margin-top:6px">Top Processes (GPU)</div>
<div class="proc-list" id="procs"></div>
</div>

<div class="col-right">
<!-- Power & Frequency section -->
<div class="section-label" onclick="toggleSection('power-section')">&#x26A1; Power &amp; Frequency <span id="power-toggle" style="font-size:10px;color:#475569">&#x25B6;</span></div>
<div class="collapsible" id="power-section"></div>

<!-- GPU Detail section -->
<div class="section-label" onclick="toggleSection('gpu-detail-section')">&#x1F3AE; GPU Detail <span id="gpu-detail-toggle" style="font-size:10px;color:#475569">&#x25B6;</span></div>
<div class="collapsible" id="gpu-detail-section"></div>

<!-- Memory Breakdown section -->
<div class="section-label" onclick="toggleSection('memory-section')">&#x1F4BE; Memory Breakdown <span id="memory-toggle" style="font-size:10px;color:#475569">&#x25B6;</span></div>
<div class="collapsible" id="memory-section"></div>
</div>

<script>
const gpuHistory = [];
const MAX_HIST = 120;
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
let expandedPid = null;

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

    ctx.strokeStyle = '#1e293b80';
    ctx.lineWidth = 0.5;
    for (let pct of [25, 50, 75]) {
        const y = h - (pct / maxV) * h;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
}

function fmtBytes(b) {
    if (b < 1024) return b + 'B';
    if (b < 1024*1024) return (b/1024).toFixed(0) + 'K';
    if (b < 1024*1024*1024) return (b/(1024*1024)).toFixed(0) + 'M';
    return (b/(1024*1024*1024)).toFixed(1) + 'G';
}

function barHtml(pct, cls, w) {
    w = w || 80;
    return '<div style="display:inline-block;width:'+w+'px;height:6px;background:#1e293b;border-radius:3px;overflow:hidden;vertical-align:middle">' +
        '<div style="width:'+Math.min(pct,100)+'%;height:100%;border-radius:3px" class="'+cls+'"></div></div>';
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

    // Stats + temps
    const ct = (data.cpu_temp ?? 0).toFixed(0);
    const gt = (data.gpu_temp ?? 0).toFixed(0);
    document.getElementById('stats').innerHTML =
        '<b>' + data.gpu_clients + ' clients</b>';
    document.getElementById('temp-val').textContent =
        'CPU ' + ct + '\\u00B0C  GPU ' + gt + '\\u00B0C';
    if (data.sensors) updateSensors(data.sensors);

    // History
    gpuHistory.push(gpu);
    if (gpuHistory.length > MAX_HIST) gpuHistory.shift();
    drawChart();

    // Process list
    const procs = data.processes || [];
    const el = document.getElementById('procs');
    el.innerHTML = procs.slice(0, 15).map(function(p) {
        const isExpanded = expandedPid === p.pid;
        let html = '<div class="proc-row" onclick="toggleProcDetail('+p.pid+')">' +
            '<span class="proc-pid">' + p.pid + '</span>' +
            '<span class="proc-name">' + p.name + '</span>' +
            '<div class="proc-bar"><div class="proc-fill-gpu" style="width:' + Math.min(p.gpu, 100) + '%"></div></div>' +
            '<span class="proc-val">' + p.gpu.toFixed(1) + '% gpu</span>' +
            '<span class="proc-val">' + p.cpu.toFixed(0) + '% cpu</span>' +
            '<span class="proc-val">' + p.mem.toFixed(0) + 'MB</span>' +
            '</div>';
        if (isExpanded && p.detail) {
            const d = p.detail;
            const ipc = d.cycles > 0 ? (d.instructions / d.cycles).toFixed(2) : '0';
            const MB = 1024*1024;
            html += '<div class="proc-detail" style="display:block">' +
                'threads=' + d.threads +
                '  disk: R=' + (d.disk_read_bytes/MB).toFixed(1) + 'M W=' + (d.disk_write_bytes/MB).toFixed(1) + 'M' +
                '  IPC=' + ipc +
                '  peak=' + (d.peak_memory/MB).toFixed(0) + 'M' +
                '  wired=' + (d.wired_size/MB).toFixed(0) + 'M' +
                '  neural=' + (d.neural_footprint/MB).toFixed(0) + 'M' +
                '<br>wakeups: idle=' + d.idle_wakeups + ' int=' + d.interrupt_wakeups +
                '  pageins=' + d.pageins +
                '  CPU: usr=' + ((d.cpu_user_ns/(d.cpu_user_ns+d.cpu_system_ns+1))*100).toFixed(0) + '%' +
                ' sys=' + ((d.cpu_system_ns/(d.cpu_user_ns+d.cpu_system_ns+1))*100).toFixed(0) + '%' +
                '</div>';
        }
        return html;
    }).join('');

    // Power section
    if (data.cpu_power || data.gpu_power) {
        updatePowerSection(data.cpu_power, data.gpu_power);
    }

    // GPU detail section
    if (data.gpu_detail) {
        updateGpuDetailSection(data.gpu_detail);
    }

    // Memory breakdown section
    if (data.memory_breakdown) {
        updateMemorySection(data.memory_breakdown);
    }
}

function toggleProcDetail(pid) {
    expandedPid = expandedPid === pid ? null : pid;
}

function toggleSection(id) {
    const el = document.getElementById(id);
    const isWide = window.innerWidth >= 600;
    if (isWide) return; // auto-expanded in wide mode
    el.classList.toggle('open');
    const toggleEl = document.getElementById(id + '-toggle');
    if (toggleEl) {
        toggleEl.innerHTML = el.classList.contains('open') ? '&#x25BC;' : '&#x25B6;';
    }
}

let sensorsVisible = false;
function toggleSensors() {
    sensorsVisible = !sensorsVisible;
    const el = document.getElementById('sensor-detail');
    const tog = document.getElementById('temp-toggle');
    el.style.display = sensorsVisible ? 'block' : 'none';
    tog.innerHTML = sensorsVisible ? '&#x25BC; sensors' : '&#x25B6; sensors';
}

function updateSensors(sensors) {
    if (!sensorsVisible) return;
    const el = document.getElementById('sensor-detail');
    let html = '';
    for (const [cat, label, color] of [
        ['cpu_sensors', 'CPU', '#ef4444'],
        ['gpu_sensors', 'GPU', '#a855f7'],
        ['system_sensors', 'SYS', '#3b82f6']
    ]) {
        const s = sensors[cat] || {};
        const keys = Object.keys(s).sort();
        if (keys.length === 0) continue;
        const vals = keys.map(function(k) {
            return k + ':' + s[k].toFixed(0);
        }).join('  ');
        html += '<span style="color:' + color + '">' +
            label + '</span> ' + vals + '<br>';
    }
    el.innerHTML = html || 'No sensors';
}

function updatePowerSection(cpuPwr, gpuPwr) {
    const el = document.getElementById('power-section');
    if (!cpuPwr && !gpuPwr) { el.innerHTML = '<div class="detail-row" style="color:#475569">Waiting for power data...</div>'; return; }

    let html = '';
    // GPU power
    if (gpuPwr) {
        const throttle = gpuPwr.throttled ? ' <span class="throttle-badge">THROTTLED</span>' : '';
        html += '<div class="detail-row">' +
            '<span class="detail-label">GPU</span>' +
            '<span class="detail-val">' + (gpuPwr.gpu_power_w||0).toFixed(1) + 'W  ' +
            (gpuPwr.gpu_freq_mhz||0) + 'MHz</span>' +
            '  state=' + (gpuPwr.active_state||'-') +
            '  pwr_limit=' + (gpuPwr.power_limit_pct||0) + '%' +
            throttle + '</div>';

        // GPU P-states
        const states = gpuPwr.frequency_states || [];
        if (states.length > 0) {
            html += '<div class="detail-row" style="font-size:10px">';
            states.forEach(function(s) {
                if (s.residency_pct > 0.5) {
                    const w = Math.max(2, s.residency_pct * 0.8);
                    html += '<span class="pstate-bar" style="width:'+w+'px;background:#8b5cf6"></span>';
                }
            });
            html += ' ';
            states.filter(function(s){return s.residency_pct > 0.5}).forEach(function(s) {
                html += (s.freq_mhz||0) + ':' + s.residency_pct.toFixed(0) + '%  ';
            });
            html += '</div>';
        }
    }

    // CPU power
    if (cpuPwr) {
        html += '<div class="detail-row">' +
            '<span class="detail-label">CPU</span>' +
            '<span class="detail-val">' + (cpuPwr.cpu_power_w||0).toFixed(1) + 'W</span></div>';

        const clusters = cpuPwr.clusters || {};
        Object.keys(clusters).sort().forEach(function(name) {
            const c = clusters[name];
            html += '<div class="detail-row" style="padding-left:16px">' +
                name + ': ' + (c.freq_mhz||0) + 'MHz  active=' + (c.active_pct||0).toFixed(0) + '%';
            const cs = c.frequency_states || [];
            if (cs.length > 0) {
                html += '  <span style="font-size:10px;color:#475569">';
                cs.filter(function(s){return s.residency_pct > 0.5}).forEach(function(s) {
                    html += (s.freq_mhz||0) + ':' + s.residency_pct.toFixed(0) + '%  ';
                });
                html += '</span>';
            }
            html += '</div>';
        });
    }

    el.innerHTML = html;
}

function updateGpuDetailSection(g) {
    const el = document.getElementById('gpu-detail-section');
    const dev = g.device_utilization || 0;
    const tiler = g.tiler_utilization || 0;
    const rend = g.renderer_utilization || 0;

    let html = '<div class="detail-row">' +
        'Device ' + barHtml(dev, 'fill-gpu', 80) + ' ' + dev + '%  ' +
        'Tiler ' + barHtml(tiler, 'fill-tiler', 80) + ' ' + tiler + '%  ' +
        'Renderer ' + barHtml(rend, 'fill-renderer', 80) + ' ' + rend + '%</div>';

    html += '<div class="detail-row">' +
        '<span class="detail-label">Memory</span>' +
        'in_use=' + fmtBytes(g.in_use_system_memory||0) +
        '  alloc=' + fmtBytes(g.alloc_system_memory||0) +
        '  driver=' + fmtBytes(g.in_use_system_memory_driver||0) +
        '  PB=' + fmtBytes(g.allocated_pb_size||0) + '</div>';

    html += '<div class="detail-row">' +
        '<span class="detail-label">Recovery</span>' +
        'count=' + (g.recovery_count||0) +
        '  last=' + (g.last_recovery_time||0) +
        '  |  split_scenes=' + (g.split_scene_count||0) +
        '  tiled=' + fmtBytes(g.tiled_scene_bytes||0) + '</div>';

    el.innerHTML = html;
}

function updateMemorySection(m) {
    const el = document.getElementById('memory-section');
    const total = m.memory_total || 1;
    const GB = 1024*1024*1024;

    // Stacked bar
    const pcts = {
        active: (m.memory_active||0) / total * 100,
        inactive: (m.memory_inactive||0) / total * 100,
        wired: (m.memory_wired||0) / total * 100,
        compressed: (m.memory_compressed||0) / total * 100,
    };
    const freePct = Math.max(0, 100 - pcts.active - pcts.inactive - pcts.wired - pcts.compressed);

    let html = '<div class="detail-row" style="margin-bottom:4px">' +
        '<div style="display:flex;height:10px;border-radius:4px;overflow:hidden;background:#1e293b">' +
        '<div class="fill-active" style="width:'+pcts.active+'%"></div>' +
        '<div class="fill-inactive" style="width:'+pcts.inactive+'%"></div>' +
        '<div class="fill-wired" style="width:'+pcts.wired+'%"></div>' +
        '<div class="fill-compressed" style="width:'+pcts.compressed+'%"></div>' +
        '</div>' +
        fmtBytes(m.memory_used||0) + ' / ' + fmtBytes(total) + '</div>';

    html += '<div class="detail-row">' +
        '<span style="color:#22c55e">active</span>=' + fmtBytes(m.memory_active||0) + '  ' +
        '<span style="color:#eab308">inactive</span>=' + fmtBytes(m.memory_inactive||0) + '  ' +
        '<span style="color:#ef4444">wired</span>=' + fmtBytes(m.memory_wired||0) + '  ' +
        '<span style="color:#06b6d4">compressed</span>=' + fmtBytes(m.memory_compressed||0) + '  ' +
        'free=' + fmtBytes(m.memory_free||0) + '  ' +
        'avail=' + fmtBytes(m.memory_available||0) + '</div>';

    html += '<div class="detail-row">' +
        '<span class="detail-label">CPU</span>' +
        (m.cpu_name||'?') + ' (' + (m.cpu_count||0) + ' cores)' +
        '  usr=' + (m.cpu_user_pct||0).toFixed(1) + '%' +
        '  sys=' + (m.cpu_system_pct||0).toFixed(1) + '%' +
        '  idle=' + (m.cpu_idle_pct||0).toFixed(1) + '%</div>';

    el.innerHTML = html;
}

window.addEventListener('resize', function() {
    drawChart();
    // In wide mode, auto-expand all collapsible sections
    const isWide = window.innerWidth >= 600;
    document.querySelectorAll('.collapsible').forEach(function(el) {
        if (isWide) el.classList.add('open');
    });
});
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
        self._power_sampler = PowerSampler(interval=max(interval, 1.0))

    def set_window(self, window: object) -> None:
        self._window = window

    def start_polling(self) -> None:
        self._power_sampler.start()
        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()

    def _collect(self) -> dict:
        from darwin_perf import _snapshot
        from darwin_perf._native import (
            cpu_time_ns,
            proc_info,
            system_gpu_stats,
            system_stats,
            temperatures,
        )

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
                    entry = {
                        "pid": pid,
                        "name": info["name"],
                        "gpu": round(gpu_pct, 1),
                        "cpu": round(cpu_pct, 1),
                        "mem": round(mem_mb, 0),
                    }
                    # Full proc detail
                    if pinfo:
                        entry["detail"] = {
                            "threads": pinfo.get("threads", 0),
                            "disk_read_bytes": pinfo.get("disk_read_bytes", 0),
                            "disk_write_bytes": pinfo.get("disk_write_bytes", 0),
                            "instructions": pinfo.get("instructions", 0),
                            "cycles": pinfo.get("cycles", 0),
                            "peak_memory": pinfo.get("peak_memory", 0),
                            "wired_size": pinfo.get("wired_size", 0),
                            "neural_footprint": pinfo.get("neural_footprint", 0),
                            "idle_wakeups": pinfo.get("idle_wakeups", 0),
                            "interrupt_wakeups": pinfo.get("interrupt_wakeups", 0),
                            "pageins": pinfo.get("pageins", 0),
                            "cpu_user_ns": pinfo.get("cpu_user_ns", 0),
                            "cpu_system_ns": pinfo.get("cpu_system_ns", 0),
                        }
                    procs.append(entry)

            procs.sort(key=lambda p: p["gpu"], reverse=True)
            data["processes"] = procs[:15]
            data["total_gpu_pct"] = round(total_gpu, 1)
            data["total_cpu_pct"] = round(total_cpu, 1)
        else:
            data["total_gpu_pct"] = 0
            data["total_cpu_pct"] = 0

        # System memory + temperatures
        sys = system_stats()
        GB = 1024**3
        data["memory_total_gb"] = round(sys.get("memory_total", 0) / GB, 1)
        data["memory_used_gb"] = round(sys.get("memory_used", 0) / GB, 1)

        temps = temperatures()
        data["cpu_temp"] = round(temps.get("cpu_avg", 0), 1)
        data["gpu_temp"] = round(temps.get("gpu_avg", 0), 1)
        data["sensors"] = {
            "cpu_sensors": temps.get("cpu_sensors", {}),
            "gpu_sensors": temps.get("gpu_sensors", {}),
            "system_sensors": temps.get("system_sensors", {}),
        }

        # Power data (cached from background thread)
        cpu_pwr, gpu_pwr = self._power_sampler.get()
        data["cpu_power"] = cpu_pwr
        data["gpu_power"] = gpu_pwr

        # GPU detail — all system_gpu_stats fields
        data["gpu_detail"] = system_gpu_stats()

        # Memory breakdown — all system_stats fields
        data["memory_breakdown"] = sys

        self._prev_snap = snap
        self._prev_cpu = curr_cpu
        self._prev_time = now
        return data

    def _poll_loop(self) -> None:
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
