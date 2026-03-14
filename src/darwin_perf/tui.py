"""darwin-perf TUI: Rich terminal system monitor with live graphs.

A full-screen terminal app showing per-process GPU utilization,
CPU/GPU power, frequencies, temperatures, and memory — with
sparkline history graphs. No sudo needed.

Usage:
    darwin-perf --tui              # all GPU-active processes
    darwin-perf --tui -i 1         # 1s update interval
    darwin-perf --tui --record f   # monitor + record to JSONL
"""

from __future__ import annotations

import json
import time as _time

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

from . import _snapshot
from ._native import cpu_time_ns, proc_info, system_gpu_stats, temperatures

# ---------------------------------------------------------------------------
# Sparkline renderer (unicode block chars)
# ---------------------------------------------------------------------------

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 40) -> str:
    """Render a sparkline string from a list of 0-100 values."""
    if not values:
        return ""
    tail = values[-width:]
    mx = max(tail) if max(tail) > 0 else 1
    return "".join(
        _SPARK_CHARS[min(int(v / mx * (len(_SPARK_CHARS) - 1)), len(_SPARK_CHARS) - 1)]
        for v in tail
    )


def _fmt_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    if b < 1024:
        return f"{b}B"
    if b < 1024**2:
        return f"{b / 1024:.0f}K"
    if b < 1024**3:
        return f"{b / 1024**2:.0f}M"
    return f"{b / 1024**3:.1f}G"


# ---------------------------------------------------------------------------
# Process row widget
# ---------------------------------------------------------------------------


class ProcessRow(Static):
    """Single process row with name, GPU %, CPU %, memory, energy, bar, sparkline."""

    def __init__(self, pid: int, name: str) -> None:
        super().__init__()
        self.pid = pid
        self.proc_name = name
        self.history: list[float] = []
        self.current_pct: float = 0.0
        self.cpu_pct: float = 0.0
        self.mem_str: str = ""
        self.power_w: float = 0.0

    def update_stats(self, gpu_pct: float, cpu_pct: float, mem_str: str, power_w: float) -> None:
        self.current_pct = gpu_pct
        self.cpu_pct = cpu_pct
        self.mem_str = mem_str
        self.power_w = power_w
        self.history.append(gpu_pct)
        if len(self.history) > 120:
            self.history = self.history[-120:]
        self.refresh_display()

    def refresh_display(self) -> None:
        pct = self.current_pct
        bar_len = int(min(pct, 100) / 2.5)  # 40 chars = 100%
        bar = "[green]" + "━" * bar_len + "[/green]" + "[dim]" + "╌" * (40 - bar_len) + "[/dim]"
        spark = _sparkline(self.history, 20)
        power_str = f"{self.power_w:.1f}W" if self.power_w >= 0.01 else ""
        self.update(
            f" {self.pid:>8}  {pct:>5.1f}%  {self.cpu_pct:>5.1f}%  "
            f"{self.mem_str:>6}  {power_str:>5}  {bar}  "
            f"[cyan]{self.proc_name:<18}[/cyan]  "
            f"[dim]{spark}[/dim]"
        )


# ---------------------------------------------------------------------------
# Summary bar
# ---------------------------------------------------------------------------


class SummaryBar(Static):
    """Top bar showing aggregate GPU stats + system metrics."""

    total_gpu = reactive(0.0)
    process_count = reactive(0)
    peak_gpu = reactive(0.0)
    model_name = reactive("")
    core_count = reactive(0)
    device_util = reactive(0)
    cpu_temp = reactive(0.0)
    gpu_temp = reactive(0.0)
    recording = reactive("")

    def render(self) -> str:
        rec = f"  │  [bold red]● REC {self.recording}[/bold red]" if self.recording else ""
        return (
            f"  [bold]{self.model_name}[/bold] ({self.core_count} cores)"
            f"  │  [bold]GPU:[/bold] [green]{self.device_util}%[/green]"
            f"  │  [bold]Sum:[/bold] [green]{self.total_gpu:5.1f}%[/green]"
            f"  │  [bold]Peak:[/bold] [yellow]{self.peak_gpu:5.1f}%[/yellow]"
            f"  │  [bold]CPU:[/bold] {self.cpu_temp:.0f}°C"
            f"  │  [bold]GPU:[/bold] {self.gpu_temp:.0f}°C"
            f"{rec}"
        )


# ---------------------------------------------------------------------------
# System GPU bar with history
# ---------------------------------------------------------------------------


class SystemGpuBar(Static):
    """System-wide GPU utilization with sparkline."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.history: list[float] = []

    def update_value(self, device_pct: float, alloc_mem: int, used_mem: int) -> None:
        self.history.append(device_pct)
        if len(self.history) > 120:
            self.history = self.history[-120:]
        spark = _sparkline(self.history, 60)
        bar_len = int(min(device_pct, 100) / 1.67)  # 60 chars = 100%
        bar = "[green]" + "█" * bar_len + "[/green]" + "[dim]" + "░" * (60 - bar_len) + "[/dim]"
        self.update(
            f"  [bold]Device GPU[/bold]  {device_pct:5.1f}%  {bar}"
            f"  VRAM: {_fmt_bytes(used_mem)}/{_fmt_bytes(alloc_mem)}\n"
            f"  [dim]History:[/dim]  {spark}"
        )


class TempPanel(Static):
    """Expandable temperature sensor panel."""

    def update_temps(self, temps: dict) -> None:
        lines = []
        for category, label, color in [
            ("cpu_sensors", "CPU", "red"),
            ("gpu_sensors", "GPU", "magenta"),
            ("system_sensors", "SYS", "blue"),
        ]:
            sensors = temps.get(category, {})
            if not sensors:
                continue
            avg_key = category.replace("_sensors", "_avg")
            avg = temps.get(avg_key, 0)
            items = sorted(sensors.items())
            vals = " ".join(f"{n}:{v:.0f}" for n, v in items)
            lines.append(
                f"  [{color}]{label}[/{color}] "
                f"[bold]{avg:.1f}°C[/bold]  "
                f"[dim]{vals}[/dim]"
            )
        self.update("\n".join(lines) if lines else "  [dim]No sensors[/dim]")


# ---------------------------------------------------------------------------
# Main TUI App
# ---------------------------------------------------------------------------


class GpuProcApp(App):
    """Per-process GPU monitor TUI."""

    CSS = """
    Screen {
        background: $surface;
    }
    #summary {
        height: 1;
        background: $primary-background;
        color: $text;
    }
    #system-bar {
        height: 3;
        padding: 0 1;
        border-bottom: solid $primary-lighten-2;
    }
    #header-row {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    #temp-panel {
        padding: 0 1;
        border-bottom: solid $primary-lighten-2;
        display: none;
    }
    #temp-panel.visible {
        display: block;
    }
    #process-list {
        height: 1fr;
        overflow-y: auto;
        padding: 0 1;
    }
    ProcessRow {
        height: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("R", "reset", "Reset history"),
        ("r", "toggle_record", "Record"),
        ("t", "toggle_temps", "Temps"),
    ]

    def __init__(
        self,
        pids: list[int] | None = None,
        interval: float = 2.0,
        top_n: int = 30,
        record_path: str | None = None,
    ) -> None:
        super().__init__()
        self._target_pids = pids
        self._interval = interval
        self._top_n = top_n
        self._prev_snap: dict[int, dict] = {}
        self._prev_cpu: dict[int, int] = {}
        self._prev_energy: dict[int, int] = {}
        self._prev_time: float = 0
        self._rows: dict[int, ProcessRow] = {}
        self._all_totals: list[float] = []
        self._record_path = record_path
        self._record_file: object | None = None
        self._record_count: int = 0
        if record_path:
            self._record_file = open(record_path, "w")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield SummaryBar(id="summary")
        yield SystemGpuBar(id="system-bar")
        yield TempPanel(id="temp-panel")
        yield Static(
            "      PID  GPU %  CPU %    Mem  Power  "
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  "
            "Process             History",
            id="header-row",
        )
        yield Vertical(id="process-list")
        yield Footer()

    def on_mount(self) -> None:
        snap = _snapshot()
        self._prev_snap = snap
        for pid in snap:
            ns = cpu_time_ns(pid)
            self._prev_cpu[pid] = ns if ns >= 0 else 0
            info = proc_info(pid)
            self._prev_energy[pid] = info["energy_nj"] if info else 0
        self._prev_time = _time.monotonic()
        self.set_interval(self._interval, self._refresh)

    def _refresh(self) -> None:
        now = _time.monotonic()
        elapsed_s = now - self._prev_time
        if elapsed_s <= 0:
            return
        elapsed_ns = elapsed_s * 1_000_000_000

        snap = _snapshot()
        curr_cpu: dict[int, int] = {}
        curr_energy: dict[int, int] = {}
        for pid in snap:
            ns = cpu_time_ns(pid)
            curr_cpu[pid] = ns if ns >= 0 else 0
            info = proc_info(pid)
            curr_energy[pid] = info["energy_nj"] if info else 0

        # Filter to specific PIDs if requested
        pids = set(snap.keys())
        if self._target_pids:
            pids = pids & set(self._target_pids)

        active: list[tuple[int, str, float, float, str, float]] = []
        for pid in pids:
            c_gpu = snap.get(pid, {}).get("gpu_ns", 0)
            p_gpu = self._prev_snap.get(pid, {}).get("gpu_ns", c_gpu)
            gpu_delta = c_gpu - p_gpu

            cpu_delta = curr_cpu.get(pid, 0) - self._prev_cpu.get(pid, curr_cpu.get(pid, 0))
            energy_delta = curr_energy.get(pid, 0) - self._prev_energy.get(pid, curr_energy.get(pid, 0))

            gpu_pct = min(gpu_delta / elapsed_ns * 100, 100) if elapsed_ns > 0 else 0
            cpu_pct = cpu_delta / elapsed_ns * 100 if elapsed_ns > 0 else 0
            power_w = energy_delta / (elapsed_s * 1e9) if elapsed_s > 0 else 0

            info = proc_info(pid)
            mem_str = _fmt_bytes(info["memory"]) if info else "0"

            name = snap[pid]["name"]
            if gpu_pct >= 0.05 or gpu_delta > 0:
                active.append((pid, name, gpu_pct, cpu_pct, mem_str, power_w))

        self._prev_snap = snap
        self._prev_cpu = curr_cpu
        self._prev_energy = curr_energy
        self._prev_time = now

        active.sort(key=lambda r: r[2], reverse=True)
        active = active[:self._top_n]

        total_pct = sum(r[2] for r in active)
        self._all_totals.append(total_pct)

        # System-wide GPU stats + temperatures
        sys_stats = system_gpu_stats()
        temps = temperatures()

        summary = self.query_one("#summary", SummaryBar)
        summary.total_gpu = total_pct
        summary.process_count = len(active)
        summary.peak_gpu = max(self._all_totals) if self._all_totals else 0
        summary.model_name = sys_stats.get("model", "?")
        summary.core_count = sys_stats.get("gpu_core_count", 0)
        summary.device_util = sys_stats.get("device_utilization", 0)
        summary.cpu_temp = temps.get("cpu_avg", 0)
        summary.gpu_temp = temps.get("gpu_avg", 0)
        summary.recording = self._record_path if self._record_file else ""

        temp_panel = self.query_one("#temp-panel", TempPanel)
        if temp_panel.has_class("visible"):
            temp_panel.update_temps(temps)

        # Write recording line if active
        if self._record_file:
            from ._native import system_stats as _sys_stats
            record = {
                "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "epoch": _time.time(),
                "interval": self._interval,
                "temperatures": temps,
                "memory": _sys_stats(),
                "gpu_stats": sys_stats,
                "processes": [
                    {"pid": pid, "name": name,
                     "gpu_percent": round(gpu_pct, 1),
                     "cpu_percent": round(cpu_pct, 1),
                     "memory": mem_str,
                     "energy_w": round(power_w, 2)}
                    for pid, name, gpu_pct, cpu_pct, mem_str, power_w in active
                ],
            }
            self._record_file.write(json.dumps(record) + "\n")
            self._record_file.flush()
            self._record_count += 1

        sys_bar = self.query_one("#system-bar", SystemGpuBar)
        sys_bar.update_value(
            sys_stats.get("device_utilization", 0),
            sys_stats.get("alloc_system_memory", 0),
            sys_stats.get("in_use_system_memory", 0),
        )

        container = self.query_one("#process-list")
        active_pids = {r[0] for r in active}

        for pid in list(self._rows.keys()):
            if pid not in active_pids:
                self._rows[pid].remove()
                del self._rows[pid]

        for pid, name, gpu_pct, cpu_pct, mem_str, power_w in active:
            if pid in self._rows:
                self._rows[pid].update_stats(gpu_pct, cpu_pct, mem_str, power_w)
            else:
                row = ProcessRow(pid, name)
                self._rows[pid] = row
                container.mount(row)
                row.update_stats(gpu_pct, cpu_pct, mem_str, power_w)

    def action_reset(self) -> None:
        """Reset all history."""
        self._all_totals.clear()
        for row in self._rows.values():
            row.history.clear()
        sys_bar = self.query_one("#system-bar", SystemGpuBar)
        sys_bar.history.clear()

    def action_toggle_temps(self) -> None:
        """Toggle temperature sensor detail panel."""
        panel = self.query_one("#temp-panel", TempPanel)
        panel.toggle_class("visible")
        if panel.has_class("visible"):
            panel.update_temps(temperatures())

    def action_toggle_record(self) -> None:
        """Toggle recording on/off."""
        if self._record_file:
            self._record_file.close()
            self._record_file = None
            self.notify(
                f"Recording stopped ({self._record_count} samples → {self._record_path})",
                severity="information",
            )
        else:
            import time
            self._record_path = f"darwin-perf-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
            self._record_file = open(self._record_path, "w")
            self._record_count = 0
            self.notify(f"Recording to {self._record_path}", severity="warning")

    def on_unmount(self) -> None:
        if self._record_file:
            self._record_file.close()


def run_tui(
    pids: list[int] | None = None,
    interval: float = 2.0,
    top_n: int = 30,
    record_path: str | None = None,
) -> None:
    """Launch the TUI app."""
    app = GpuProcApp(pids=pids, interval=interval, top_n=top_n, record_path=record_path)
    app.run()
