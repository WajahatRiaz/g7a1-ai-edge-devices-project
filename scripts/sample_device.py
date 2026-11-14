#!/usr/bin/env python3
"""Sample llama-server memory, and Pi power/thermal when vcgencmd exists.

Wrap the benchmark so the sample covers the same window:

    ./scripts/sample_device.sh -- ./scripts/characterize_baseline.sh --lengths 512,2048,4096

On this host, peak RSS is recorded and energy, temperature, and throttling
stay empty. Those three are filled only on a Raspberry Pi.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "baseline" / "device.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def platform_name() -> tuple[str, bool]:
    model = read_text(Path("/proc/device-tree/model"))
    if model:
        name = model.replace("\x00", "").strip()
        return name, "Raspberry Pi" in name
    try:
        uname = subprocess.run(
            ["uname", "-srm"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown", False
    name = uname.stdout.strip() or "unknown"
    return name, False


def llama_server_hwm_bytes() -> int | None:
    peak = 0
    found = False
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        comm = read_text(entry / "comm")
        if comm is None or comm.strip() != "llama-server":
            continue
        status = read_text(entry / "status")
        if status is None:
            continue
        for line in status.splitlines():
            if line.startswith("VmHWM:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    peak = max(peak, int(parts[1]) * 1024)
                    found = True
    return peak if found else None


def run_vcgencmd(vcgencmd: str, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            [vcgencmd, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def rail_power_w(adc_text: str) -> float | None:
    currents: dict[str, float] = {}
    volts: dict[str, float] = {}
    pattern = re.compile(
        r"(\S+)\s+(current|volt)\(\d+\)=([0-9.eE+-]+)([AV])"
    )
    for line in adc_text.splitlines():
        match = pattern.search(line.strip())
        if match is None:
            continue
        name, kind, value, _unit = match.groups()
        if kind == "current" and name.endswith("_A"):
            currents[name[:-2]] = float(value)
        elif kind == "volt" and name.endswith("_V"):
            volts[name[:-2]] = float(value)
    paired = [volts[rail] * amps for rail, amps in currents.items() if rail in volts]
    if not paired:
        return None
    return float(sum(paired))


def parse_temp_c(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"temp=([0-9.]+)", text)
    if match is None:
        return None
    return float(match.group(1))


THROTTLE_BITS = {
    0: "under_voltage",
    1: "freq_capped",
    2: "throttled",
    3: "soft_temp_limit",
    16: "under_voltage_occurred",
    17: "freq_capped_occurred",
    18: "throttled_occurred",
    19: "soft_temp_limit_occurred",
}


def parse_throttled(text: str | None) -> dict:
    if not text:
        return {"raw": None, "now": [], "since_boot": []}
    match = re.search(r"(0x[0-9a-fA-F]+)", text)
    raw = match.group(1) if match else (text.strip() or None)
    if raw is None:
        return {"raw": None, "now": [], "since_boot": []}
    value = int(raw, 16)
    now = [name for bit, name in THROTTLE_BITS.items() if bit < 16 and value & (1 << bit)]
    since_boot = [name for bit, name in THROTTLE_BITS.items() if bit >= 16 and value & (1 << bit)]
    return {"raw": raw, "now": now, "since_boot": since_boot}


def integrate_energy(samples: list[tuple[float, float]]) -> float | None:
    if len(samples) < 2:
        return None
    energy = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        energy += (p0 + p1) * 0.5 * max(0.0, t1 - t0)
    return energy


def sample_once(vcgencmd: str | None) -> dict:
    power_w = None
    temp_c = None
    if vcgencmd:
        adc = run_vcgencmd(vcgencmd, "pmic_read_adc")
        if adc:
            power_w = rail_power_w(adc)
        temp_c = parse_temp_c(run_vcgencmd(vcgencmd, "measure_temp"))
    return {
        "t": time.perf_counter(),
        "power_w": power_w,
        "temp_c": temp_c,
        "rss_bytes": llama_server_hwm_bytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.cmd)
    if command and command[0] == "--":
        command = command[1:]

    platform, is_pi = platform_name()
    vcgencmd = shutil.which("vcgencmd")
    started = now_iso()
    t_start = time.perf_counter()
    points = [sample_once(vcgencmd)]
    exit_code = 0

    if command:
        child = subprocess.Popen(command)
        while child.poll() is None:
            time.sleep(max(0.1, args.interval))
            points.append(sample_once(vcgencmd))
        exit_code = int(child.returncode or 0)
        points.append(sample_once(vcgencmd))

    throttled = {"raw": None, "now": [], "since_boot": []}
    if vcgencmd:
        throttled = parse_throttled(run_vcgencmd(vcgencmd, "get_throttled"))

    temps = [point["temp_c"] for point in points if point["temp_c"] is not None]
    rss = [point["rss_bytes"] for point in points if point["rss_bytes"] is not None]
    power_samples = [(point["t"], point["power_w"]) for point in points if point["power_w"] is not None]
    notes = []
    if vcgencmd is None:
        notes.append("vcgencmd not found; energy, temperature, and throttling were not sampled.")
    if not rss:
        notes.append("llama-server was not running, so peak RSS is empty.")
    if not is_pi:
        notes.append("Host RSS is a debug sample. Report memory, energy, and thermal from the Pi.")
    if throttled["now"]:
        notes.append(
            "Throttle bits were set during this run: " + ", ".join(throttled["now"]) + "."
        )
    elif throttled["since_boot"]:
        notes.append(
            "No throttle bit was set at the end of the run. Since boot: "
            + ", ".join(throttled["since_boot"])
            + "."
        )

    record = {
        "platform": platform,
        "is_raspberry_pi": is_pi,
        "report_as": "device" if is_pi else "host_debug",
        "started_at": started,
        "ended_at": now_iso(),
        "duration_s": round(time.perf_counter() - t_start, 3),
        "command": command,
        "peak_memory_bytes": max(rss) if rss else None,
        "energy_j": integrate_energy(power_samples),
        "temp_c_start": temps[0] if temps else None,
        "temp_c_end": temps[-1] if temps else None,
        "temp_c_max": max(temps) if temps else None,
        "throttled": throttled["raw"],
        "throttled_now": throttled["now"],
        "throttled_since_boot": throttled["since_boot"],
        "power_samples": len(power_samples),
        "notes": " ".join(notes),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    runs = out.parent / "device_runs.jsonl"
    with open(runs, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(f"wrote {out}")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
