"""Basic tests for darwin-perf."""

import os
import sys
import time

import pytest


@pytest.fixture(autouse=True)
def _skip_non_macos():
    if sys.platform != "darwin":
        pytest.skip("macOS only")


def test_gpu_time_ns_self():
    from darwin_perf import gpu_time_ns

    ns = gpu_time_ns(0)
    assert ns >= 0, "Should be able to read own process GPU time"


def test_gpu_time_ns_self_explicit_pid():
    from darwin_perf import gpu_time_ns

    ns = gpu_time_ns(os.getpid())
    assert ns >= 0


def test_gpu_time_ns_multi():
    from darwin_perf import gpu_time_ns_multi

    result = gpu_time_ns_multi([0])
    assert 0 in result
    assert result[0] >= 0


def test_gpu_time_ns_multi_invalid_pid():
    from darwin_perf import gpu_time_ns_multi

    result = gpu_time_ns_multi([999999999])
    assert result[999999999] == 0


def test_monitor_self():
    from darwin_perf import GpuMonitor

    mon = GpuMonitor()
    s1 = mon.sample()
    assert s1 == 0.0  # first sample is always 0 (baseline)
    time.sleep(0.1)
    s2 = mon.sample()
    assert isinstance(s2, float)
    assert s2 >= 0.0


def test_monitor_context_manager():
    from darwin_perf import GpuMonitor

    with GpuMonitor() as mon:
        mon.sample()  # baseline
        time.sleep(0.1)
        mon.sample()  # first real sample

    stats = mon.summary()
    assert "gpu_pct_avg" in stats
    assert "gpu_pct_max" in stats
    assert stats["num_samples"] >= 1


def test_monitor_background():
    from darwin_perf import GpuMonitor

    mon = GpuMonitor()
    mon.start(interval=0.1)
    time.sleep(0.5)
    mon.stop()
    stats = mon.summary()
    assert stats["num_samples"] >= 2


def test_system_stats_memory():
    from darwin_perf import system_stats

    s = system_stats()
    assert s["memory_total"] > 0
    assert s["memory_used"] > 0
    assert s["memory_available"] >= 0
    assert s["memory_used"] <= s["memory_total"]
    # Compressed should be included in used
    assert s["memory_compressed"] >= 0


def test_system_stats_cpu_ticks():
    from darwin_perf import system_stats

    s = system_stats()
    assert s["cpu_ticks_user"] > 0
    assert s["cpu_ticks_system"] > 0
    assert s["cpu_ticks_idle"] > 0
    assert s["cpu_count"] > 0
    assert isinstance(s["cpu_name"], str)
    assert len(s["cpu_name"]) > 0


def test_system_stats_cpu_delta():
    from darwin_perf import system_stats

    s1 = system_stats()
    time.sleep(0.5)
    s2 = system_stats()
    # Ticks should advance
    assert s2["cpu_ticks_user"] >= s1["cpu_ticks_user"]
    assert s2["cpu_ticks_idle"] >= s1["cpu_ticks_idle"]
    total_delta = (
        (s2["cpu_ticks_user"] - s1["cpu_ticks_user"])
        + (s2["cpu_ticks_system"] - s1["cpu_ticks_system"])
        + (s2["cpu_ticks_idle"] - s1["cpu_ticks_idle"])
    )
    assert total_delta > 0


def test_proc_info_self():
    from darwin_perf import proc_info

    info = proc_info(os.getpid())
    assert info is not None
    assert info["cpu_ns"] >= 0
    assert info["memory"] > 0
    assert info["real_memory"] > 0
    assert info["threads"] >= 1


def test_system_gpu_stats():
    from darwin_perf import system_gpu_stats

    s = system_gpu_stats()
    assert "device_utilization" in s
    assert "model" in s
    assert "gpu_core_count" in s
    assert s["gpu_core_count"] > 0


def test_cpu_power_returns_dict():
    from darwin_perf import cpu_power

    result = cpu_power(0.1)
    assert isinstance(result, dict)
    assert "cpu_power_w" in result
    assert "cpu_energy_nj" in result
    assert "clusters" in result
    assert isinstance(result["cpu_power_w"], float)
    assert result["cpu_power_w"] >= 0
    assert isinstance(result["cpu_energy_nj"], int)


def test_cpu_power_cluster_states():
    from darwin_perf import cpu_power

    result = cpu_power(0.1)
    clusters = result.get("clusters", {})
    # Should have at least one cluster (ECPU or PCPU)
    assert len(clusters) >= 1, f"Expected at least 1 cluster, got {list(clusters.keys())}"
    for name, data in clusters.items():
        assert name in ("ECPU", "PCPU"), f"Unexpected cluster name: {name}"
        assert "frequency_states" in data
        assert "active_pct" in data
        assert isinstance(data["frequency_states"], list)


def test_gpu_clients_has_api_field():
    from darwin_perf import gpu_clients

    clients = gpu_clients()
    # gpu_clients may be empty if no GPU activity, but if present, check api
    for c in clients:
        assert "api" in c, f"Missing 'api' key in gpu client: {c}"
        assert isinstance(c["api"], str)


def test_gpu_power_expanded_temps():
    from darwin_perf import gpu_power

    result = gpu_power(0.1)
    temps = result.get("temperatures", {})
    assert isinstance(temps, dict)
    assert "gpu_sensors" in temps
    assert "cpu_sensors" in temps
    assert "system_sensors" in temps


def test_temperatures_standalone():
    from darwin_perf import temperatures

    temps = temperatures()
    assert isinstance(temps, dict)
    assert "cpu_sensors" in temps
    assert "gpu_sensors" in temps
    assert "system_sensors" in temps
    # Should find at least some sensors on Apple Silicon
    cpu = temps["cpu_sensors"]
    gpu = temps["gpu_sensors"]
    assert len(cpu) > 0 or len(gpu) > 0, "Expected at least one temperature sensor"
    # Verify values are reasonable temperatures
    for name, val in cpu.items():
        assert 0 < val < 150, f"{name} = {val}°C out of range"
    for name, val in gpu.items():
        assert 0 < val < 150, f"{name} = {val}°C out of range"
    # Averages should exist if sensors exist
    if cpu:
        assert "cpu_avg" in temps
        assert 0 < temps["cpu_avg"] < 150
    if gpu:
        assert "gpu_avg" in temps
        assert 0 < temps["gpu_avg"] < 150


def test_cli_json_output():
    import json
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "darwin_perf.cli", "--json", "--once", "-i", "0.5"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    # Should produce at least one JSON line
    lines = [l for l in result.stdout.strip().split("\n") if l]
    assert len(lines) >= 1
    data = json.loads(lines[0])
    assert "timestamp" in data
    assert "processes" in data


def test_cli_csv_output():
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "darwin_perf.cli", "--csv", "--once", "-i", "0.5"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    lines = result.stdout.strip().split("\n")
    assert len(lines) >= 1
    header = lines[0]
    assert "timestamp" in header
    assert "pid" in header
    assert "gpu_pct" in header


def test_snapshot_system_mode():
    from darwin_perf import snapshot

    s = snapshot(interval=0.2, system=True)
    assert isinstance(s, dict)
    assert "processes" in s
    assert "cpu" in s
    assert "gpu" in s
    assert "temperatures" in s
    assert "memory" in s
    assert "gpu_stats" in s
    # CPU data
    assert "cpu_power_w" in s["cpu"]
    assert "clusters" in s["cpu"]
    # GPU data
    assert "gpu_power_w" in s["gpu"]
    # Temperatures
    assert "cpu_avg" in s["temperatures"]
    # Memory
    assert "memory_total" in s["memory"]
    assert s["memory"]["memory_total"] > 0


def test_record_creates_jsonl(tmp_path):
    import json
    import subprocess

    outfile = tmp_path / "test_record.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "darwin_perf.cli", "--record", str(outfile), "--once", "-i", "0.5"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert outfile.exists()
    content = outfile.read_text().strip()
    assert len(content) > 0
    data = json.loads(content.split("\n")[0])
    assert "timestamp" in data
    assert "processes" in data
    # Full system recording should include system-level data
    assert "cpu" in data
    assert "gpu" in data
    assert "temperatures" in data
    assert "memory" in data


def test_export_creates_csvs(tmp_path):
    import json
    import subprocess

    # Create a fixture JSONL
    fixture = tmp_path / "fixture.jsonl"
    record = {
        "timestamp": "2026-03-14T12:00:00",
        "epoch": 1773580800.0,
        "interval": 1.0,
        "processes": [
            {"pid": 1, "name": "test", "gpu_percent": 50.0, "cpu_percent": 10.0,
             "memory_mb": 100.0, "energy_w": 1.5, "threads": 4}
        ],
        "cpu": {"cpu_power_w": 5.0, "cpu_energy_nj": 5000000000, "clusters": {
            "ECPU": {"freq_mhz": 600, "active_pct": 80.0, "frequency_states": []},
            "PCPU": {"freq_mhz": 1200, "active_pct": 50.0, "frequency_states": []},
        }},
        "gpu": {"gpu_power_w": 3.0, "gpu_freq_mhz": 1000, "throttled": False},
        "temperatures": {"cpu_avg": 45.0, "gpu_avg": 40.0, "system_avg": 35.0,
                         "cpu_sensors": {}, "gpu_sensors": {}, "system_sensors": {}},
        "memory": {"memory_total": 137438953472, "memory_used": 50000000000,
                   "memory_available": 87438953472, "memory_compressed": 1000000000},
        "gpu_stats": {"device_utilization": 25, "model": "Apple M4 Max"},
    }
    fixture.write_text(json.dumps(record) + "\n")

    result = subprocess.run(
        [sys.executable, "-m", "darwin_perf.cli", "--export", str(fixture)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0

    sys_csv = tmp_path / "fixture_system.csv"
    proc_csv = tmp_path / "fixture_processes.csv"
    assert sys_csv.exists()
    assert proc_csv.exists()

    sys_lines = sys_csv.read_text().strip().split("\n")
    assert len(sys_lines) == 2  # header + 1 data row
    assert "cpu_power_w" in sys_lines[0]
    assert "5.0" in sys_lines[1]

    proc_lines = proc_csv.read_text().strip().split("\n")
    assert len(proc_lines) == 2
    assert "test" in proc_lines[1]


def test_replay_reads_jsonl(tmp_path):
    import json
    import subprocess

    # Create a fixture JSONL file
    fixture = tmp_path / "fixture.jsonl"
    record = {
        "timestamp": "2026-03-14T12:00:00+0000",
        "epoch": 1773580800.0,
        "interval": 1.0,
        "processes": [
            {"pid": 1, "name": "test", "gpu_percent": 50.0, "cpu_percent": 10.0, "memory_mb": 100.0}
        ],
    }
    fixture.write_text(json.dumps(record) + "\n")

    result = subprocess.run(
        [sys.executable, "-m", "darwin_perf.cli", "--replay", str(fixture), "--once"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert "test" in result.stdout
    assert "replay" in result.stdout
