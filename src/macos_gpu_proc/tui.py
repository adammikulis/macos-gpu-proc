"""gpu-proc TUI: Rich terminal GPU monitor with live graphs.

A full-screen terminal app showing per-process GPU utilization with
sparkline history graphs, sorted by usage. No sudo needed.

Usage:
    gpu-proc --tui              # all GPU-active processes
    gpu-proc --tui -i 1         # 1s update interval
"""

from __future__ import annotations

import time as _time

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

from . import _snapshot
from ._native import cpu_time_ns, proc_info, system_gpu_stats

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
    """Top bar showing aggregate GPU stats."""

    total_gpu = reactive(0.0)
    process_count = reactive(0)
    peak_gpu = reactive(0.0)
    model_name = reactive("")
    core_count = reactive(0)
    device_util = reactive(0)

    def render(self) -> str:
        return (
            f"  [bold]{self.model_name}[/bold] ({self.core_count} cores)"
            f"  │  [bold]GPU:[/bold] [green]{self.device_util}%[/green] (hw)"
            f"  │  [bold]Sum:[/bold] [green]{self.total_gpu:5.1f}%[/green]"
            f"  │  [bold]Peak:[/bold] [yellow]{self.peak_gpu:5.1f}%[/yellow]"
            f"  │  [bold]Clients:[/bold] {self.process_count}"
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
        ("r", "reset", "Reset history"),
    ]

    def __init__(
        self,
        pids: list[int] | None = None,
        interval: float = 2.0,
        top_n: int = 30,
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield SummaryBar(id="summary")
        yield SystemGpuBar(id="system-bar")
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

        # System-wide GPU stats
        sys_stats = system_gpu_stats()

        summary = self.query_one("#summary", SummaryBar)
        summary.total_gpu = total_pct
        summary.process_count = len(active)
        summary.peak_gpu = max(self._all_totals) if self._all_totals else 0
        summary.model_name = sys_stats.get("model", "?")
        summary.core_count = sys_stats.get("gpu_core_count", 0)
        summary.device_util = sys_stats.get("device_utilization", 0)

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


def run_tui(
    pids: list[int] | None = None,
    interval: float = 2.0,
    top_n: int = 30,
) -> None:
    """Launch the TUI app."""
    app = GpuProcApp(pids=pids, interval=interval, top_n=top_n)
    app.run()
