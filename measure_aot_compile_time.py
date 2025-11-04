#!/usr/bin/env python3
"""
Measure ahead-of-time compilation and link times for hecbench frontends.

The script builds each requested benchmark/frontend combination multiple times,
parsing the `/usr/bin/time` output emitted by the benchmark Makefiles to
separate compile and link durations. Results are written to a CSV via pandas.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd
import tomllib

# Supported frontends in the order we typically report them.
FRONTENDS: tuple[str, ...] = ("proteus", "aot", "cpp", "dsl")

# Regex capturing entries like "compile 0:01.95 real" or "link 0:00.37 real => foo".
TIME_LINE_RE = re.compile(
    r"^(?P<label>compile|link)\s+(?P<elapsed>[0-9:.]+)\s+real(?:\s+=>\s+.+)?$",
    re.IGNORECASE,
)


class CommandError(RuntimeError):
    """Raised when a subprocess command fails."""

    def __init__(self, message: str, output: str, returncode: int):
        super().__init__(message)
        self.output = output
        self.returncode = returncode


def parse_elapsed(value: str) -> float:
    """
    Convert a %E-style time string (H:MM:SS or M:SS with optional decimals)
    into total seconds.
    """
    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Unrecognized elapsed time format: {value}")


@dataclass
class TimingStats:
    """Aggregate compile/link timing statistics for a single build."""

    total_seconds: float = 0.0
    samples: int = 0

    def add(self, seconds: float) -> None:
        self.total_seconds += seconds
        self.samples += 1


def parse_compile_output(lines: Iterable[str]) -> dict[str, TimingStats]:
    """
    Scan command output and accumulate compile/link timings.
    """
    stats: dict[str, TimingStats] = {
        "compile": TimingStats(),
        "link": TimingStats(),
    }
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        match = TIME_LINE_RE.match(line)
        if not match:
            continue
        label = match.group("label").lower()
        elapsed = parse_elapsed(match.group("elapsed"))
        stats[label].add(elapsed)
    return stats


def normalize_commands(command_spec: object) -> list[str]:
    """
    Build commands in the descriptors can be strings or arrays of strings.
    Normalize to a list of shell command strings.
    """
    if isinstance(command_spec, str):
        return [command_spec]
    if isinstance(command_spec, Sequence):
        return [str(cmd) for cmd in command_spec]
    raise TypeError(f"Unsupported build command type: {type(command_spec)!r}")


def execute(cmd: str, *, cwd: Path, env: Mapping[str, str]) -> str:
    """
    Execute a shell command, streaming output to stdout while also capturing it.
    """
    print(f"=> Execute {cmd}")
    process = subprocess.Popen(
        ["bash", "-lc", cmd],
        cwd=str(cwd),
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None  # for type checkers
    output_lines: list[str] = []
    for line in process.stdout:
        output_lines.append(line)
        print(line, end="")
    returncode = process.wait()
    output = "".join(output_lines)
    if returncode != 0:
        raise CommandError(f"Command failed: {cmd}", output=output, returncode=returncode)
    return output


def load_hecbench_descriptor(path: Path) -> tuple[dict, dict[str, dict]]:
    """
    Load and split the hecbench descriptor into (config, benchmarks).
    """
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if len(raw) != 1:
        raise ValueError("Expected a single top-level table in descriptor.")
    (group_name, group_data), = raw.items()
    if group_name != "hecbench":
        raise ValueError(f"Descriptor {path} does not describe hecbench.")
    config = group_data.get("config", {})
    benchmarks = {k: v for k, v in group_data.items() if k != "config"}
    return config, benchmarks


def resolve_build_command(config: dict, machine: str, frontend: str) -> list[str]:
    """
    Extract the build command array for the specified machine/frontend.
    """
    machine_cfg = config.get("build", {}).get(machine, {})
    cmd = machine_cfg.get(frontend, {}).get("command")
    if cmd is None:
        cmd = machine_cfg.get("command")
    if cmd is None:
        raise KeyError(f"No build command for machine={machine} frontend={frontend}")
    return normalize_commands(cmd)


def resolve_clean_command(config: dict, machine: str) -> list[str]:
    clean = config.get("build", {}).get(machine, {}).get("clean", {}).get("command")
    if clean is None:
        return []
    return normalize_commands(clean)


def resolve_bench_entry(
    benchmark: dict,
    machine: str,
    frontend: str,
) -> dict:
    """
    Return the descriptor subsection for the given benchmark/machine/frontend.
    """
    machine_section = benchmark.get(machine, {})
    entry = machine_section.get(frontend)
    if not entry:
        raise KeyError(
            f"Benchmark does not declare frontend '{frontend}' for machine '{machine}'."
        )
    return entry


def ensure_env(args: argparse.Namespace) -> tuple[str, str]:
    """
    Determine compiler and proteus paths, preferring CLI overrides but ensuring
    the environment variables exist as a fallback.
    """
    compiler = args.compiler or os.environ.get("PROTEUS_CC")
    proteus_path = args.proteus_path or os.environ.get("PROTEUS_PATH")
    missing = []
    if not compiler:
        missing.append("PROTEUS_CC / --compiler")
    if not proteus_path:
        missing.append("PROTEUS_PATH / --proteus-path")
    if missing:
        raise RuntimeError(
            "Missing required environment settings: " + ", ".join(missing)
        )
    return compiler, proteus_path


def build_single(
    *,
    build_commands: Sequence[str],
    clean_commands: Sequence[str],
    build_dir: Path,
    env: Mapping[str, str],
    logs_dir: Path,
    log_name: str,
) -> tuple[str, dict[str, TimingStats]]:
    """
    Run clean + build commands (in order) and return combined output + stats.
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    combined_output: list[str] = []
    for cmd in clean_commands:
        try:
            combined_output.append(execute(cmd, cwd=build_dir, env=env))
        except CommandError as err:
            raise CommandError(
                f"Clean command failed in {build_dir}: {cmd}", err.output, err.returncode
            )
    for cmd in build_commands:
        try:
            combined_output.append(execute(cmd, cwd=build_dir, env=env))
        except CommandError as err:
            raise CommandError(
                f"Build command failed in {build_dir}: {cmd}", err.output, err.returncode
            )
    full_output = "".join(combined_output)
    stats = parse_compile_output(full_output.splitlines())

    log_path = logs_dir / f"{log_name}.log"
    log_path.write_text(full_output, encoding="utf-8")
    return full_output, stats


def assemble_record(
    *,
    machine: str,
    benchmark: str,
    frontend: str,
    rep: int,
    stats: Mapping[str, TimingStats],
    build_dir: Path,
    build_commands: Sequence[str],
    compiler: str,
    proteus_path: str,
) -> dict[str, object]:
    """
    Package a single repetition's measurements for pandas ingestion.
    """
    compile_stats = stats["compile"]
    link_stats = stats["link"]
    return {
        "timestamp": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "machine": machine,
        "benchmark": benchmark,
        "frontend": frontend,
        "rep": rep,
        "compile_time_s": compile_stats.total_seconds if compile_stats.samples else float("nan"),
        "link_time_s": link_stats.total_seconds if link_stats.samples else float("nan"),
        "compile_samples": compile_stats.samples,
        "link_samples": link_stats.samples,
        "build_path": str(build_dir),
        "build_command": " && ".join(build_commands),
        "compiler": compiler,
        "proteus_path": proteus_path,
        "enable_proteus": frontend in {"proteus", "cpp", "dsl"},
    }


def create_results_dir(path: Path | None) -> Path:
    """
    Use provided results directory or create a timestamped one.
    """
    if path is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path("results") / f"compile-time-{stamp}"
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(exist_ok=True)
    return path


def flatten_groups(groups: Sequence[Sequence[str]] | None) -> list[str]:
    if not groups:
        return []
    flattened: list[str] = []
    for group in groups:
        flattened.extend(group)
    return flattened


def filter_requested_frontends(requested: Sequence[str] | None) -> list[str]:
    if not requested:
        return list(FRONTENDS)
    invalid = sorted(set(requested) - set(FRONTENDS))
    if invalid:
        raise ValueError(f"Unsupported frontends requested: {invalid}")
    # Preserve requested order but deduplicate.
    seen: set[str] = set()
    normalized: list[str] = []
    for frontend in requested:
        if frontend not in seen:
            normalized.append(frontend)
            seen.add(frontend)
    return normalized


def filter_requested_benchmarks(
    requested: Sequence[str] | None, available: Sequence[str]
) -> list[str]:
    if not requested:
        return list(available)
    missing = sorted(set(requested) - set(available))
    if missing:
        raise KeyError(f"Requested benchmarks not found: {missing}")
    seen: set[str] = set()
    ordered: list[str] = []
    for name in requested:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure hecbench ahead-of-time compile/link timings."
    )
    parser.add_argument("--toml", default="hecbench.toml", type=Path)
    parser.add_argument(
        "--machine",
        required=True,
        choices=("amd", "nvidia"),
        help="Target platform build configuration.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        required=True,
        help="Number of clean builds per benchmark/frontend.",
    )
    parser.add_argument(
        "--bench",
        nargs="+",
        action="append",
        default=None,
        help="Optional list of benchmarks to build (defaults to all).",
    )
    parser.add_argument(
        "--frontends",
        nargs="+",
        dest="frontends",
        default=None,
        help="Subset of frontends to measure (default: proteus aot cpp dsl).",
    )
    parser.add_argument(
        "--compiler",
        help="Path to the compiler wrapper (defaults to $PROTEUS_CC).",
    )
    parser.add_argument(
        "--proteus-path",
        dest="proteus_path",
        help="Path to Proteus install (defaults to $PROTEUS_PATH).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory to store CSV results and logs (default: results/compile-time-<timestamp>).",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip benchmark/frontend repetitions already present in the output CSV.",
    )
    args = parser.parse_args(argv)

    if args.reps <= 0:
        raise ValueError("--reps must be a positive integer.")

    compiler, proteus_path = ensure_env(args)

    config, benchmarks = load_hecbench_descriptor(args.toml)
    requested_frontends = filter_requested_frontends(args.frontends)
    all_bench_names = list(benchmarks.keys())
    requested_bench_names = filter_requested_benchmarks(
        flatten_groups(args.bench), all_bench_names
    )

    selected_benchmarks: Mapping[str, dict] = {
        name: benchmarks[name] for name in requested_bench_names
    }

    clean_commands = resolve_clean_command(config, args.machine)

    results_dir = create_results_dir(args.results_dir)
    logs_dir = results_dir / "logs"

    csv_path = results_dir / f"{args.machine}-compile-times.csv"
    existing_rows: pd.DataFrame | None = (
        pd.read_csv(csv_path) if csv_path.exists() else None
    )

    records: list[dict[str, object]] = []

    for bench_name, bench_cfg in selected_benchmarks.items():
        print(f"== Benchmark: {bench_name}")
        for frontend in requested_frontends:
            try:
                entry = resolve_bench_entry(bench_cfg, args.machine, frontend)
            except KeyError:
                print(f"   - Skipping frontend '{frontend}' (not configured).")
                continue
            build_path = (Path.cwd() / entry["path"]).resolve()
            build_commands = resolve_build_command(config, args.machine, frontend)

            enable_proteus = frontend in {"proteus", "cpp", "dsl"}
            env = os.environ.copy()
            env["CC"] = compiler
            env["ENABLE_PROTEUS"] = "yes" if enable_proteus else "no"
            env["PROTEUS_PATH"] = proteus_path

            for rep in range(args.reps):
                if args.skip_completed and existing_rows is not None:
                    mask = (
                        (existing_rows["benchmark"] == bench_name)
                        & (existing_rows["frontend"] == frontend)
                        & (existing_rows.get("machine", args.machine) == args.machine)
                        & (existing_rows["rep"] == rep)
                    )
                    if mask.any():
                        print(
                            f"   - Frontend {frontend}, rep {rep} (skipped: already in CSV)"
                        )
                        continue
                print(f"   - Frontend {frontend}, rep {rep}")
                log_name = f"{args.machine}-{bench_name}-{frontend}-rep{rep}"
                try:
                    _, stats = build_single(
                        build_commands=build_commands,
                        clean_commands=clean_commands,
                        build_dir=build_path,
                        env=env,
                        logs_dir=logs_dir,
                        log_name=log_name,
                    )
                except CommandError as err:
                    log_path = logs_dir / f"{log_name}-error.log"
                    log_path.write_text(err.output, encoding="utf-8")
                    print(f"     ! Build failed; see {log_path}")
                    raise

                record = assemble_record(
                    machine=args.machine,
                    benchmark=bench_name,
                    frontend=frontend,
                    rep=rep,
                    stats=stats,
                    build_dir=build_path,
                    build_commands=build_commands,
                    compiler=compiler,
                    proteus_path=proteus_path,
                )
                records.append(record)

    if not records:
        if args.skip_completed and existing_rows is not None and not existing_rows.empty:
            print(f"=> No new measurements; existing results remain at {csv_path}")
            return 0
        print("No measurements collected; nothing to write.", file=sys.stderr)
        return 1

    new_df = pd.DataFrame.from_records(records)
    if existing_rows is not None:
        combined_df = pd.concat([existing_rows, new_df], ignore_index=True)
        combined_df.drop_duplicates(
            subset=["machine", "benchmark", "frontend", "rep"],
            keep="last",
            inplace=True,
        )
    else:
        combined_df = new_df

    combined_df.to_csv(csv_path, index=False)

    print(f"=> Appended {len(new_df)} rows; CSV now has {len(combined_df)} total rows at {csv_path}")
    summary = (
        new_df.groupby(["benchmark", "frontend"])
        [["compile_time_s", "link_time_s"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    print("=> Mean compile/link times (seconds):")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
