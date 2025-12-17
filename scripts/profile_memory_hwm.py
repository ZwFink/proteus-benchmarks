#!/usr/bin/env python3
"""
Profile CPU and GPU memory high-water mark (HWM) for benchmark/frontend combinations.

Uses valgrind massif for CPU memory and rocprofv3 for GPU memory on AMD systems.

Usage:
    python profile_memory_hwm.py --output results.csv
    python profile_memory_hwm.py --output results.csv --benchmarks 3mm,gemm --frontends aot,proteus
    python profile_memory_hwm.py --output results.csv --skip-completed  # Resume from previous run
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


# Default benchmarks and frontends
DEFAULT_BENCHMARKS = [
    "3mm", "adam", "attention", "bezier-surface",
    "conv3d", "floyd-warshall", "gemm", "minibude"
]
DEFAULT_FRONTENDS = ["aot", "proteus", "dsl", "cpp"]
DEFAULT_MACHINE = "amd"

CSV_FIELDNAMES = ["benchmark", "frontend", "cpu_hwm_bytes", "gpu_hwm_bytes"]


def load_benchmark_configs(toml_path: Path) -> tuple[dict, dict]:
    """Load benchmark descriptor and extract group config."""
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    # Expect single top-level key (e.g., "hecbench")
    assert len(data.keys()) == 1, f"Expected single top-level key, got {list(data.keys())}"
    group_name = list(data.keys())[0]
    benchmarks = data[group_name].copy()
    group_config = benchmarks.pop("config", {})
    return benchmarks, group_config


def resolve_build_command(
    group_config: dict, machine: str, frontend: str, bench_path: str
) -> str | None:
    """Get build command with fallback: frontend-specific -> machine default.

    Also handles special cases:
    - All builds need CC=${PROTEUS_CC}
    - hip/ directories need ENABLE_PROTEUS and GPU_BACKEND flags
    """
    build_cfg = group_config.get("build", {}).get(machine, {})
    cmd = build_cfg.get(frontend, {}).get("command")
    if cmd is None:
        cmd = build_cfg.get("command")

    if cmd is None:
        return None

    # Always add CC=${PROTEUS_CC} for the compiler
    cmd = f"{cmd} CC=${{PROTEUS_CC}}"

    # For hip/ directories, we need to set ENABLE_PROTEUS and GPU_BACKEND flags
    if "/hip/" in bench_path:
        cmd = f"{cmd} GPU_BACKEND=hip"
        if frontend == "proteus":
            cmd = f"{cmd} ENABLE_PROTEUS=yes"
        elif frontend == "aot":
            cmd = f"{cmd} ENABLE_PROTEUS=no"

    return cmd


def resolve_clean_command(group_config: dict, machine: str) -> str | None:
    """Get clean command for machine."""
    build_cfg = group_config.get("build", {}).get(machine, {})
    clean_cfg = build_cfg.get("clean", {})
    return clean_cfg.get("command")


def resolve_bench_entry(bench_cfg: dict, machine: str, frontend: str) -> dict:
    """Get benchmark entry (path, exe) for machine/frontend."""
    return bench_cfg.get(machine, {}).get(frontend, {})


def get_inputs(bench_cfg: dict) -> str:
    """Get default input arguments."""
    inputs = bench_cfg.get("inputs", {})
    return inputs.get("default", "")


def get_completed_combos(output_path: Path) -> set[tuple[str, str]]:
    """Read existing CSV and return set of (benchmark, frontend) already done."""
    if not output_path.exists():
        return set()
    completed = set()
    with open(output_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((row["benchmark"], row["frontend"]))
    return completed


def write_result(output_path: Path, result: dict, write_header: bool):
    """Append single result row to CSV."""
    mode = "w" if write_header else "a"
    with open(output_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def clean_proteus_cache(bench_path: Path, verbose: bool, dry_run: bool):
    """Remove .proteus cache directory to ensure fresh JIT compilation."""
    cache_dir = bench_path / ".proteus"
    if dry_run:
        print(f"  [DRY-RUN] Would remove cache: {cache_dir}")
        return
    if cache_dir.exists():
        if verbose:
            print(f"  Cleaning cache: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)


def build_benchmark(
    bench_path: Path, clean_cmd: str | None, build_cmd: str, verbose: bool, dry_run: bool
) -> bool:
    """Build the benchmark. Returns True on success."""
    if dry_run:
        if clean_cmd:
            print(f"  [DRY-RUN] Would run: {clean_cmd}")
        print(f"  [DRY-RUN] Would run: {build_cmd}")
        return True

    # Run clean first
    if clean_cmd:
        if verbose:
            print(f"  Running: {clean_cmd}")
        result = subprocess.run(
            clean_cmd, shell=True, cwd=bench_path,
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  Warning: clean failed: {result.stderr}", file=sys.stderr)

    # Build
    if verbose:
        print(f"  Running: {build_cmd}")
    result = subprocess.run(
        build_cmd, shell=True, cwd=bench_path,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  Error: build failed: {result.stderr}", file=sys.stderr)
        return False
    return True


def extract_massif_hwm(massif_file: Path) -> int | None:
    """Extract the maximum mem_heap_B value from a massif file."""
    max_heap = None
    pattern = re.compile(r"^mem_heap_B=(\d+)$")

    try:
        with open(massif_file, "r") as f:
            for line in f:
                match = pattern.match(line.strip())
                if match:
                    value = int(match.group(1))
                    if max_heap is None or value > max_heap:
                        max_heap = value
    except (IOError, OSError) as e:
        print(f"  Warning: Could not read {massif_file}: {e}", file=sys.stderr)
        return None

    return max_heap


def run_massif(
    bench_path: Path, exe: str, inputs: str, verbose: bool, dry_run: bool
) -> int | None:
    """Run valgrind massif and return CPU HWM in bytes."""
    # Create temp file for massif output
    with tempfile.NamedTemporaryFile(prefix="massif_", suffix=".out", delete=False) as tmp:
        massif_out = Path(tmp.name)

    cmd = f"valgrind --tool=massif --massif-out-file={massif_out} ./{exe} {inputs}"

    if dry_run:
        print(f"  [DRY-RUN] Would run: {cmd}")
        massif_out.unlink(missing_ok=True)
        return 12345678  # Dummy value for dry-run

    exe_path = bench_path / exe
    if not exe_path.exists():
        print(f"  Warning: Executable not found: {exe_path}", file=sys.stderr)
        massif_out.unlink(missing_ok=True)
        return None

    if verbose:
        print(f"  Running massif: {cmd}")

    try:
        result = subprocess.run(
            cmd, shell=True, cwd=bench_path,
            capture_output=True, text=True, timeout=600  # 10 minute timeout
        )
        if result.returncode != 0:
            print(f"  Warning: massif run failed: {result.stderr}", file=sys.stderr)
            return None

        hwm = extract_massif_hwm(massif_out)
        return hwm
    except subprocess.TimeoutExpired:
        print(f"  Warning: massif run timed out", file=sys.stderr)
        return None
    finally:
        massif_out.unlink(missing_ok=True)


def compute_gpu_hwm(agent_file: Path, trace_file: Path) -> int | None:
    """Parse rocprofv3 CSV files, compute GPU-only HWM."""
    try:
        # Load agent info to identify GPU agents
        agents = {}
        with open(agent_file) as f:
            for row in csv.DictReader(f):
                agents[int(row["Node_Id"])] = {
                    "type": row["Agent_Type"],
                    "name": row["Name"]
                }

        # Track allocations and compute HWM per agent
        addr_to_info = {}  # addr -> (size, agent_id)
        current_by_agent = {}
        hwm_by_agent = {}

        with open(trace_file) as f:
            for row in csv.DictReader(f):
                addr = row["Address"]
                op = row["Operation"]
                agent = int(row["Agent_Id"])

                if agent not in current_by_agent:
                    current_by_agent[agent] = 0
                    hwm_by_agent[agent] = 0

                if op == "MEMORY_ALLOCATION_ALLOCATE":
                    size = int(row["Allocation_Size"])
                    addr_to_info[addr] = (size, agent)
                    current_by_agent[agent] += size
                    hwm_by_agent[agent] = max(hwm_by_agent[agent], current_by_agent[agent])
                elif op == "MEMORY_ALLOCATION_FREE":
                    if addr in addr_to_info:
                        size, alloc_agent = addr_to_info.pop(addr)
                        current_by_agent[alloc_agent] -= size

        # Sum HWM for GPU agents only
        gpu_total = 0
        for agent_id, hwm in hwm_by_agent.items():
            info = agents.get(agent_id, {"type": "???", "name": "Unknown"})
            if info["type"] != "CPU":
                gpu_total += hwm

        return gpu_total

    except (IOError, OSError, KeyError, ValueError) as e:
        print(f"  Warning: Could not parse rocprofv3 output: {e}", file=sys.stderr)
        return None


def run_rocprofv3(
    bench_path: Path, exe: str, inputs: str, verbose: bool, dry_run: bool
) -> int | None:
    """Run rocprofv3 memory allocation trace and return GPU HWM in bytes."""
    # Create temp directory for rocprofv3 output
    tmpdir = tempfile.mkdtemp(prefix="rocprof_")

    cmd = (
        f"rocprofv3 --memory-allocation-trace --output-format csv "
        f"--output-directory {tmpdir} -- ./{exe} {inputs}"
    )

    if dry_run:
        print(f"  [DRY-RUN] Would run: {cmd}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return 536870912  # Dummy value for dry-run

    exe_path = bench_path / exe
    if not exe_path.exists():
        print(f"  Warning: Executable not found: {exe_path}", file=sys.stderr)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None

    if verbose:
        print(f"  Running rocprofv3: {cmd}")

    try:
        result = subprocess.run(
            cmd, shell=True, cwd=bench_path,
            capture_output=True, text=True, timeout=600  # 10 minute timeout
        )
        if result.returncode != 0:
            print(f"  Warning: rocprofv3 run failed: {result.stderr}", file=sys.stderr)
            return None

        # Find the output files (rocprofv3 creates a subdirectory with hostname)
        tmpdir_path = Path(tmpdir)
        agent_files = list(tmpdir_path.glob("**/*_agent_info.csv"))
        trace_files = list(tmpdir_path.glob("**/*_memory_allocation_trace.csv"))

        if not agent_files or not trace_files:
            print(f"  Warning: rocprofv3 output files not found in {tmpdir}", file=sys.stderr)
            return None

        hwm = compute_gpu_hwm(agent_files[0], trace_files[0])
        return hwm

    except subprocess.TimeoutExpired:
        print(f"  Warning: rocprofv3 run timed out", file=sys.stderr)
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Profile CPU and GPU memory HWM for benchmarks"
    )
    parser.add_argument(
        "--toml", "-t",
        type=Path,
        default=Path("hecbench.toml"),
        help="Path to benchmark descriptor TOML (default: hecbench.toml)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--machine", "-m",
        default=DEFAULT_MACHINE,
        help=f"Machine type (default: {DEFAULT_MACHINE})"
    )
    parser.add_argument(
        "--benchmarks", "-b",
        default=",".join(DEFAULT_BENCHMARKS),
        help=f"Comma-separated benchmark list (default: {','.join(DEFAULT_BENCHMARKS)})"
    )
    parser.add_argument(
        "--frontends", "-f",
        default=",".join(DEFAULT_FRONTENDS),
        help=f"Comma-separated frontend list (default: {','.join(DEFAULT_FRONTENDS)})"
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip benchmark/frontend combos already in output CSV"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress and commands"
    )
    args = parser.parse_args()

    # Parse benchmark and frontend lists
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    frontends = [f.strip() for f in args.frontends.split(",")]

    # Load TOML config
    if not args.toml.exists():
        print(f"Error: TOML file not found: {args.toml}", file=sys.stderr)
        sys.exit(1)

    bench_configs, group_config = load_benchmark_configs(args.toml)

    # Get completed combos if resuming
    completed = set()
    if args.skip_completed:
        completed = get_completed_combos(args.output)
        if completed:
            print(f"Found {len(completed)} completed combos, will skip them")

    # Determine if we need to write header
    write_header = not args.output.exists() or (not args.skip_completed)

    # Get base directory (where script is run from, should contain benchmarks/)
    base_dir = Path.cwd()

    # Process each benchmark/frontend combo
    total = len(benchmarks) * len(frontends)
    done = 0

    for benchmark in benchmarks:
        bench_cfg = bench_configs.get(benchmark)
        if not bench_cfg:
            print(f"Warning: Benchmark '{benchmark}' not found in TOML, skipping")
            continue

        inputs = get_inputs(bench_cfg)
        clean_cmd = resolve_clean_command(group_config, args.machine)

        for frontend in frontends:
            done += 1

            # Skip if already completed
            if (benchmark, frontend) in completed:
                print(f"[{done}/{total}] {benchmark}/{frontend}: skipped (already completed)")
                continue

            print(f"[{done}/{total}] {benchmark}/{frontend}")

            # Resolve benchmark entry
            entry = resolve_bench_entry(bench_cfg, args.machine, frontend)
            if not entry:
                print(f"  Warning: No entry for {args.machine}/{frontend}, skipping")
                continue

            bench_path = base_dir / entry.get("path", "")
            exe = entry.get("exe", "")

            if not bench_path.exists():
                print(f"  Warning: Path not found: {bench_path}, skipping")
                continue

            # Get build command
            build_cmd = resolve_build_command(
                group_config, args.machine, frontend, str(bench_path)
            )
            if not build_cmd:
                print(f"  Warning: No build command for {args.machine}/{frontend}, skipping")
                continue

            # Build
            if args.verbose:
                print(f"  Building in {bench_path}")
            if not build_benchmark(bench_path, clean_cmd, build_cmd, args.verbose, args.dry_run):
                print(f"  Skipping profiling due to build failure")
                continue

            # For JIT frontends, clean .proteus cache before each profiler run
            # to ensure both profilers measure with fresh JIT compilation
            is_jit_frontend = frontend in ("proteus", "dsl", "cpp")

            # Run massif for CPU HWM
            if is_jit_frontend:
                clean_proteus_cache(bench_path, args.verbose, args.dry_run)
            cpu_hwm = run_massif(bench_path, exe, inputs, args.verbose, args.dry_run)

            # Run rocprofv3 for GPU HWM
            if is_jit_frontend:
                clean_proteus_cache(bench_path, args.verbose, args.dry_run)
            gpu_hwm = run_rocprofv3(bench_path, exe, inputs, args.verbose, args.dry_run)

            # Write result
            result = {
                "benchmark": benchmark,
                "frontend": frontend,
                "cpu_hwm_bytes": cpu_hwm if cpu_hwm is not None else "",
                "gpu_hwm_bytes": gpu_hwm if gpu_hwm is not None else "",
            }

            if args.verbose:
                print(f"  CPU HWM: {cpu_hwm}, GPU HWM: {gpu_hwm}")

            if not args.dry_run:
                write_result(args.output, result, write_header)
                write_header = False  # Only write header once

    print(f"Done. Results written to {args.output}")


if __name__ == "__main__":
    main()
