#!/usr/bin/env python3
"""Record Pi power and temperature in parallel with a benchmark.

Each wrapped command is stored by date and is not overwritten:

    ./scripts/sample_device.sh -- ./scripts/run_harness.sh --split long_extra --task-id long_extra_03_fund_after_dump
    ./scripts/sample_device.sh -- ./scripts/run_harness.sh --split test --limit 3
    ./scripts/sample_device.sh -- ./scripts/characterize_baseline.sh --lengths 512,2048,4096

The sampler thread runs for the whole command. Logs land in
results/baseline/history/<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "results" / "baseline"

TRACE_NAME = "device_trace.jsonl"
SUMMARY_NAME = "device.json"
PID_NAME = "device_sampler.pid"
LOG_NAME = "device_sampler.log"

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
    pattern = re.compile(r"(\S+)\s+(current|volt)\(\d+\)=([0-9.eE+-]+)([AV])")
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


def sample_once(vcgencmd: str | None, pool: ThreadPoolExecutor | None) -> dict:
    power_w = None
    temp_c = None
    throttled = {"raw": None, "now": [], "since_boot": []}
    if vcgencmd and pool is not None:
        adc_f = pool.submit(run_vcgencmd, vcgencmd, "pmic_read_adc")
        temp_f = pool.submit(run_vcgencmd, vcgencmd, "measure_temp")
        thr_f = pool.submit(run_vcgencmd, vcgencmd, "get_throttled")
        adc = adc_f.result()
        if adc:
            power_w = rail_power_w(adc)
        temp_c = parse_temp_c(temp_f.result())
        throttled = parse_throttled(thr_f.result())
    return {
        "kind": "sample",
        "t_iso": now_iso(),
        "power_w": power_w,
        "temp_c": temp_c,
        "rss_bytes": llama_server_hwm_bytes(),
        "throttled": throttled["raw"],
        "throttled_now": throttled["now"],
        "throttled_since_boot": throttled["since_boot"],
    }


def pid_path(directory: Path) -> Path:
    return directory / PID_NAME


def trace_path(directory: Path) -> Path:
    return directory / TRACE_NAME


def summary_path(directory: Path) -> Path:
    return directory / SUMMARY_NAME


def read_pid(directory: Path) -> int | None:
    text = read_text(pid_path(directory))
    if text is None:
        return None
    text = text.strip()
    if not text.isdigit():
        return None
    return int(text)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()


def load_trace(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _iso_age(origin: datetime, stamp: str) -> float | None:
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return (moment - origin).total_seconds()


def integrate_energy(samples: list[dict]) -> float | None:
    points = [
        (sample["t_s"], sample["power_w"])
        for sample in samples
        if sample.get("power_w") is not None and sample.get("t_s") is not None
    ]
    if len(points) < 2:
        return None
    energy = 0.0
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        energy += (p0 + p1) * 0.5 * max(0.0, t1 - t0)
    return energy


def assign_time(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    samples = [row for row in rows if row.get("kind", "sample") == "sample"]
    marks = [row for row in rows if row.get("kind") == "mark"]
    if not samples:
        return [], marks
    origin = None
    if samples[0].get("t_iso"):
        try:
            origin = datetime.fromisoformat(samples[0]["t_iso"])
        except ValueError:
            origin = None
    for sample in samples:
        if sample.get("t_s") is None and origin is not None and sample.get("t_iso"):
            sample["t_s"] = round(_iso_age(origin, sample["t_iso"]) or 0.0, 3)
    t0 = samples[0].get("t_s") or 0.0
    for mark in marks:
        if mark.get("t_s") is not None or origin is None or not mark.get("t_iso"):
            continue
        age = _iso_age(origin, mark["t_iso"])
        if age is not None:
            mark["t_s"] = round(age + t0, 3)
    return samples, marks


def summarize_trace(rows: list[dict], *, interval_s: float | None = None) -> dict | None:
    samples, marks = assign_time(rows)
    if not samples:
        return None
    platform, is_pi = platform_name()
    temps = [sample["temp_c"] for sample in samples if sample.get("temp_c") is not None]
    rss = [sample["rss_bytes"] for sample in samples if sample.get("rss_bytes") is not None]
    during: list[str] = []
    for sample in samples:
        for name in sample.get("throttled_now") or []:
            if name not in during:
                during.append(name)
    last = samples[-1]
    notes = []
    if all(sample.get("power_w") is None for sample in samples):
        notes.append("vcgencmd did not return rail power; energy is empty.")
    if not rss:
        notes.append("llama-server was not running, so peak RSS is empty.")
    if not is_pi:
        notes.append("Host RSS is a debug sample. Report memory, energy, and thermal from the Pi.")
    if during:
        notes.append("Throttle bits were active during the recording: " + ", ".join(during) + ".")
    elif last.get("throttled_since_boot"):
        notes.append(
            "No throttle bit was active during the recording. Since boot: "
            + ", ".join(last["throttled_since_boot"])
            + "."
        )
    duration = samples[-1].get("t_s")
    if duration is None:
        duration = 0.0
    return {
        "platform": platform,
        "is_raspberry_pi": is_pi,
        "report_as": "device" if is_pi else "host_debug",
        "started_at": samples[0].get("t_iso"),
        "ended_at": samples[-1].get("t_iso"),
        "duration_s": round(float(duration), 3),
        "interval_s": interval_s,
        "n_samples": len(samples),
        "marks": [{"t_s": mark.get("t_s"), "label": mark.get("label")} for mark in marks],
        "peak_memory_bytes": max(rss) if rss else None,
        "energy_j": integrate_energy(samples),
        "temp_c_start": temps[0] if temps else None,
        "temp_c_end": temps[-1] if temps else None,
        "temp_c_max": max(temps) if temps else None,
        "temp_c_min": min(temps) if temps else None,
        "throttled": last.get("throttled"),
        "throttled_now": last.get("throttled_now") or [],
        "throttled_during": during,
        "throttled_since_boot": last.get("throttled_since_boot") or [],
        "power_samples": sum(1 for sample in samples if sample.get("power_w") is not None),
        "notes": " ".join(notes),
    }


def downsample(samples: list[dict], max_points: int = 480) -> list[dict]:
    if len(samples) <= max_points:
        keep = samples
    else:
        stride = max(1, len(samples) // max_points)
        chosen = set(range(0, len(samples), stride))
        chosen.add(len(samples) - 1)
        for index, sample in enumerate(samples):
            if sample.get("throttled_now"):
                chosen.add(index)
        keep = [samples[index] for index in sorted(chosen)]
    slim = []
    for sample in keep:
        item = {
            "t_s": sample.get("t_s"),
            "power_w": sample.get("power_w"),
            "temp_c": sample.get("temp_c"),
            "rss_bytes": sample.get("rss_bytes"),
        }
        if sample.get("throttled_now"):
            item["throttled_now"] = sample["throttled_now"]
        slim.append(item)
    return slim


def load_experiment(directory: Path) -> dict | None:
    rows = load_trace(trace_path(directory))
    samples, marks = assign_time(rows)
    summary = None
    path = summary_path(directory)
    if path.is_file():
        with open(path, encoding="utf-8") as handle:
            summary = json.load(handle)
    if summary is None:
        summary = summarize_trace(rows)
    if summary is None:
        return None
    summary["series"] = downsample(samples)
    summary["marks"] = [{"t_s": mark.get("t_s"), "label": mark.get("label")} for mark in marks]
    return summary


def write_summary(directory: Path, rows: list[dict], interval_s: float | None) -> dict | None:
    summary = summarize_trace(rows, interval_s=interval_s)
    if summary is None:
        return None
    path = summary_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def cmd_run(args: argparse.Namespace) -> None:
    directory = Path(args.dir)
    directory.mkdir(parents=True, exist_ok=True)
    stop = False

    def handle_stop(signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    platform, _is_pi = platform_name()
    vcgencmd = shutil.which("vcgencmd")
    print(
        f"sampler pid={os.getpid()} interval={args.interval}s platform={platform} "
        f"vcgencmd={'yes' if vcgencmd else 'no'}",
        flush=True,
    )
    t0 = time.perf_counter()
    path = trace_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=3) as pool:
        while not stop:
            point = sample_once(vcgencmd, pool)
            point["t_s"] = round(time.perf_counter() - t0, 3)
            handle.write(json.dumps(point) + "\n")
            handle.flush()
            deadline = time.perf_counter() + max(0.2, args.interval)
            while not stop and time.perf_counter() < deadline:
                time.sleep(0.05)
    rows = load_trace(path)
    summary = write_summary(directory, rows, args.interval)
    pid = read_pid(directory)
    if pid == os.getpid():
        pid_path(directory).unlink(missing_ok=True)
    if summary is None:
        print("sampler stopped with no samples", flush=True)
        return
    print(
        f"sampler stopped samples={summary['n_samples']} "
        f"energy_j={summary['energy_j']} temp_max={summary['temp_c_max']}",
        flush=True,
    )


def infer_kind(command: list[str]) -> str:
    text = " ".join(command)
    if "characterize_baseline" in text:
        return "sweep"
    if "--split" in command:
        index = command.index("--split")
        split = command[index + 1] if index + 1 < len(command) else ""
        if split == "long_extra":
            return "long_task"
        if split == "test":
            return "held_out"
        if split == "train":
            return "train"
    if "run_harness" in text:
        return "harness"
    return "power"


def write_meta(directory: Path, meta: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")


def archive_directory(directory: Path, *, kind: str, command: list[str], exit_code: int, started_at: str) -> Path:
    """Copy a finished live recording into history/<timestamp>/, unless it is already there."""
    if directory.parent.name == "history":
        dest = directory
        stamp = directory.name
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        dest = directory / "history" / stamp
        dest.mkdir(parents=True, exist_ok=True)
        for name in (TRACE_NAME, SUMMARY_NAME):
            source = directory / name
            if source.is_file():
                shutil.copyfile(source, dest / name)
    write_meta(
        dest,
        {
            "id": stamp,
            "started_at": started_at,
            "ended_at": now_iso(),
            "kind": kind,
            "command": command,
            "exit_code": exit_code,
        },
    )
    return dest


def cmd_wrap(args: argparse.Namespace) -> None:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("usage: ./scripts/sample_device.sh -- <command>")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = Path(args.dir) / "history" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    kind = infer_kind(command)
    stop_event = threading.Event()

    def loop() -> None:
        vcgencmd = shutil.which("vcgencmd")
        t0 = time.perf_counter()
        path = trace_path(run_dir)
        with open(path, "a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=3) as pool:
            while not stop_event.is_set():
                point = sample_once(vcgencmd, pool)
                point["t_s"] = round(time.perf_counter() - t0, 3)
                handle.write(json.dumps(point) + "\n")
                handle.flush()
                deadline = time.perf_counter() + max(0.2, args.interval)
                while not stop_event.is_set() and time.perf_counter() < deadline:
                    time.sleep(0.05)

    print(f"recording {kind} {stamp}", flush=True)
    worker = threading.Thread(target=loop, name="device-sampler", daemon=True)
    worker.start()
    env = os.environ.copy()
    env["A1_RESULTS_DIR"] = str(run_dir)
    completed = subprocess.run(command, env=env, cwd=str(ROOT))
    stop_event.set()
    worker.join(timeout=15)
    rows = load_trace(trace_path(run_dir))
    write_summary(run_dir, rows, args.interval)
    if kind == "power":
        if (run_dir / "characterization.json").is_file():
            kind = "sweep"
        elif (run_dir / "long_extra").is_dir():
            kind = "long_task"
        elif (run_dir / "test").is_dir():
            kind = "held_out"
        elif (run_dir / "train").is_dir():
            kind = "train"
    archive_directory(
        run_dir,
        kind=kind,
        command=command,
        exit_code=int(completed.returncode or 0),
        started_at=started,
    )
    print(f"archived {run_dir}")
    raise SystemExit(int(completed.returncode or 0))


def cmd_start(args: argparse.Namespace) -> None:
    directory = Path(args.dir)
    directory.mkdir(parents=True, exist_ok=True)
    pid = read_pid(directory)
    if pid is not None and pid_alive(pid):
        raise SystemExit(f"sampler already running (pid {pid})")
    trace = trace_path(directory)
    if trace.exists():
        trace.unlink()
    summary = summary_path(directory)
    if summary.exists():
        summary.unlink()
    log_path = directory / LOG_NAME
    log_handle = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "run",
            "--dir",
            str(directory),
            "--interval",
            str(args.interval),
        ],
        cwd=str(ROOT),
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    log_handle.close()
    pid_path(directory).write_text(f"{proc.pid}\n", encoding="utf-8")
    print(f"sampling pid {proc.pid} every {args.interval}s")
    print(f"trace {trace}")
    print("run the benchmarks, then: ./scripts/sample_device.sh stop")


def cmd_stop(args: argparse.Namespace) -> None:
    directory = Path(args.dir)
    pid = read_pid(directory)
    if pid is None or not pid_alive(pid):
        print("sampler is not running")
    else:
        os.kill(pid, signal.SIGTERM)
        for _ in range(150):
            if not pid_alive(pid):
                break
            time.sleep(0.1)
        else:
            os.kill(pid, signal.SIGKILL)
        pid_path(directory).unlink(missing_ok=True)
    rows = load_trace(trace_path(directory))
    if not summary_path(directory).is_file():
        write_summary(directory, rows, None)
    summary = load_experiment(directory)
    if summary is None:
        raise SystemExit("no samples recorded")
    print(
        f"samples={summary['n_samples']} duration_s={summary['duration_s']} "
        f"energy_j={summary['energy_j']} temp_c_max={summary['temp_c_max']} "
        f"throttled_during={summary['throttled_during']}"
    )
    archived = archive_directory(
        directory,
        kind="power",
        command=[],
        exit_code=0,
        started_at=summary.get("started_at") or now_iso(),
    )
    print(f"archived {archived}")


def cmd_mark(args: argparse.Namespace) -> None:
    directory = Path(args.dir)
    label = " ".join(args.label).strip()
    if not label:
        raise SystemExit("mark needs a label")
    pid = read_pid(directory)
    if pid is None or not pid_alive(pid):
        raise SystemExit("sampler is not running; start it before marking a phase")
    append_jsonl(
        trace_path(directory),
        {"kind": "mark", "t_iso": now_iso(), "label": label},
    )
    print(f"marked {label}")


def cmd_status(args: argparse.Namespace) -> None:
    directory = Path(args.dir)
    pid = read_pid(directory)
    running = pid is not None and pid_alive(pid)
    rows = load_trace(trace_path(directory))
    samples = [row for row in rows if row.get("kind", "sample") == "sample"]
    print(f"running={running} pid={pid if running else None} samples={len(samples)}")
    if samples:
        last = samples[-1]
        print(
            f"last t_s={last.get('t_s')} power_w={last.get('power_w')} "
            f"temp_c={last.get('temp_c')} throttled_now={last.get('throttled_now')}"
        )


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dir", default=str(DEFAULT_DIR))

    parser = argparse.ArgumentParser(description="Continuous Pi power and thermal sampler")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", parents=[common], help="sample until SIGTERM")
    run.add_argument("--interval", type=float, default=1.0)
    run.set_defaults(func=cmd_run)

    start = sub.add_parser("start", parents=[common], help="start the sampler in the background")
    start.add_argument("--interval", type=float, default=1.0)
    start.set_defaults(func=cmd_start)

    stop = sub.add_parser("stop", parents=[common], help="stop the sampler and write the summary")
    stop.set_defaults(func=cmd_stop)

    mark = sub.add_parser("mark", parents=[common], help="label a phase inside the running trace")
    mark.add_argument("label", nargs="+")
    mark.set_defaults(func=cmd_mark)

    status = sub.add_parser("status", parents=[common], help="show whether the sampler is running")
    status.set_defaults(func=cmd_status)

    wrap = sub.add_parser("wrap", parents=[common], help="sample one command and archive it by date")
    wrap.add_argument("--interval", type=float, default=1.0)
    wrap.add_argument("command", nargs=argparse.REMAINDER)
    wrap.set_defaults(func=cmd_wrap)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
