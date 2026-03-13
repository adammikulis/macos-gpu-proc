"""gpu-proc TUI: Rich terminal GPU monitor with live graphs.

A full-screen terminal app showing per-process GPU utilization with
sparkline history graphs, sorted by usage. Better than Activity Monitor
because it shows history, works over SSH, and needs no mouse.

Usage:
    gpu-proc --tui              # all processes (sudo for others)
    gpu-proc --tui --self       # own process only
    gpu-proc --tui -i 1         # 1s update interval
"""

from __future__ import annotations

import os
import time as _time
from collections import defaultdict

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

from ._native import gpu_time_ns_multi

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


# ---------------------------------------------------------------------------
# Process row widget
# ---------------------------------------------------------------------------


class ProcessRow(Static):
    """Single process row with name, GPU %, bar, and sparkline history."""

    def __init__(self, pid: int, name: str) -> None:
        super().__init__()
        self.pid = pid
        self.proc_name = name
        self.history: list[float] = []
        self.current_pct: float = 0.0

    def update_pct(self, pct: float) -> None:
        self.current_pct = pct
        self.history.append(pct)
        if len(self.history) > 120:
            self.history = self.history[-120:]
        self.refresh_display()

    def refresh_display(self) -> None:
        pct = self.current_pct
        bar_len = int(min(pct, 100) / 2.5)  # 40 chars = 100%
        bar = "[green]" + "━" * bar_len + "[/green]" + "[dim]" + "╌" * (40 - bar_len) + "[/dim]"
        spark = _sparkline(self.history, 30)
        pid_str = str(self.pid) if self.pid != 0 else str(os.getpid())
        self.update(
            f" {pid_str:>8}  {pct:>6.1f}%  {bar}  "
            f"[cyan]{self.proc_name:<20}[/cyan]  "
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
    avg_gpu = reactive(0.0)

    def render(self) -> str:
        return (
            f"  [bold]GPU Total:[/bold] [green]{self.total_gpu:5.1f}%[/green]"
            f"  │  [bold]Peak:[/bold] [yellow]{self.peak_gpu:5.1f}%[/yellow]"
            f"  │  [bold]Avg:[/bold] {self.avg_gpu:5.1f}%"
            f"  │  [bold]Processes:[/bold] {self.process_count}"
        )


# ---------------------------------------------------------------------------
# System GPU bar with history
# ---------------------------------------------------------------------------


class SystemGpuBar(Static):
    """System-wide GPU utilization with sparkline."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.history: list[float] = []

    def update_value(self, total_pct: float) -> None:
        self.history.append(total_pct)
        if len(self.history) > 120:
            self.history = self.history[-120:]
        spark = _sparkline(self.history, 60)
        bar_len = int(min(total_pct, 100) / 1.67)  # 60 chars = 100%
        bar = "[green]" + "█" * bar_len + "[/green]" + "[dim]" + "░" * (60 - bar_len) + "[/dim]"
        self.update(
            f"  [bold]System GPU[/bold]  {total_pct:5.1f}%  {bar}\n"
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
        self_only: bool = False,
        interval: float = 2.0,
        top_n: int = 30,
    ) -> None:
        super().__init__()
        self._target_pids = pids
        self._self_only = self_only
        self._interval = interval
        self._top_n = top_n
        self._prev: dict[int, int] = {}
        self._prev_time: float = 0
        self._rows: dict[int, ProcessRow] = {}
        self._all_totals: list[float] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield SummaryBar(id="summary")
        yield SystemGpuBar(id="system-bar")
        yield Static(
            "      PID   GPU %   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  "
            "Process               History",
            id="header-row",
        )
        yield Vertical(id="process-list")
        yield Footer()

    def on_mount(self) -> None:
        # Take baseline sample
        pids = self._resolve_pids()
        self._prev = gpu_time_ns_multi(pids)
        self._prev_time = _time.monotonic()
        # Start periodic refresh
        self.set_interval(self._interval, self._refresh)

    def _resolve_pids(self) -> list[int]:
        if self._self_only:
            return [0]
        if self._target_pids:
            return list(self._target_pids)
        try:
            import psutil
            return [p.pid for p in psutil.process_iter(["pid"])]
        except ImportError:
            return [0]

    def _pid_name(self, pid: int) -> str:
        if pid == 0:
            return "self"
        try:
            import psutil
            return psutil.Process(pid).name()
        except Exception:
            return "?"

    def _refresh(self) -> None:
        now = _time.monotonic()
        elapsed_s = now - self._prev_time
        if elapsed_s <= 0:
            return

        pids = self._resolve_pids()
        curr = gpu_time_ns_multi(pids)
        elapsed_ns = elapsed_s * 1_000_000_000

        # Compute per-process GPU %
        active: list[tuple[int, str, float]] = []
        for pid in set(list(self._prev.keys()) + list(curr.keys())):
            c = curr.get(pid, -1)
            p = self._prev.get(pid, -1)
            if c < 0 or p < 0:
                continue
            delta = c - p
            if delta < 0:
                continue
            pct = min((delta / elapsed_ns) * 100, 100)
            if pct < 0.05:
                continue
            name = self._pid_name(pid)
            active.append((pid, name, pct))

        self._prev = curr
        self._prev_time = now

        # Sort by GPU %, take top N
        active.sort(key=lambda r: r[2], reverse=True)
        active = active[:self._top_n]

        total_pct = sum(r[2] for r in active)
        self._all_totals.append(total_pct)

        # Update summary
        summary = self.query_one("#summary", SummaryBar)
        summary.total_gpu = total_pct
        summary.process_count = len(active)
        summary.peak_gpu = max(self._all_totals) if self._all_totals else 0
        summary.avg_gpu = sum(self._all_totals) / len(self._all_totals) if self._all_totals else 0

        # Update system bar
        sys_bar = self.query_one("#system-bar", SystemGpuBar)
        sys_bar.update_value(total_pct)

        # Update/add/remove process rows
        container = self.query_one("#process-list")
        active_pids = {r[0] for r in active}

        # Remove rows for processes no longer active
        for pid in list(self._rows.keys()):
            if pid not in active_pids:
                self._rows[pid].remove()
                del self._rows[pid]

        # Update or create rows
        for pid, name, pct in active:
            if pid in self._rows:
                self._rows[pid].update_pct(pct)
            else:
                row = ProcessRow(pid, name)
                self._rows[pid] = row
                container.mount(row)
                row.update_pct(pct)

    def action_reset(self) -> None:
        """Reset all history."""
        self._all_totals.clear()
        for row in self._rows.values():
            row.history.clear()
        sys_bar = self.query_one("#system-bar", SystemGpuBar)
        sys_bar.history.clear()


def run_tui(
    pids: list[int] | None = None,
    self_only: bool = False,
    interval: float = 2.0,
    top_n: int = 30,
) -> None:
    """Launch the TUI app."""
    app = GpuProcApp(pids=pids, self_only=self_only, interval=interval, top_n=top_n)
    app.run()
