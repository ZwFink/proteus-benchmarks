#!/usr/bin/env python3
"""
Profile CPU and GPU memory high-water mark (HWM) for benchmark/frontend combinations.

Uses valgrind massif for CPU memory, rocprofv3 for GPU memory on AMD systems,
and nsys (Nsight Systems) for GPU memory on NVIDIA systems. Also extracts static
GPU memory (global/static variables) from the device code using llvm tools.

Output columns:
- cpu_hwm_bytes: CPU heap memory HWM from valgrind massif
- gpu_hwm_bytes: GPU runtime allocation HWM from rocprofv3/nsys
- gpu_static_bytes: GPU static/global memory from device symbols

Usage:
    python profile_memory_hwm.py --output results.csv
    python profile_memory_hwm.py --output results.csv --benchmarks 3mm,gemm --frontends aot,proteus
    python profile_memory_hwm.py --output results.csv --skip-completed  # Resume from previous run
    python profile_memory_hwm.py --output results.csv --machine nvidia  # Profile on NVIDIA GPUs
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

CSV_FIELDNAMES = ["benchmark", "frontend", "cpu_hwm_bytes", "gpu_hwm_bytes", "gpu_static_bytes"]


def find_nsys() -> str | None:
    """Find nsys executable, preferring NVHPC bundled version.

    Search order:
    1. NSYS environment variable
    2. Common NVHPC installation paths
    3. Fall back to PATH lookup
    """
    # Check environment variable first
    if "NSYS" in os.environ:
        nsys_path = os.environ["NSYS"]
        if os.path.isfile(nsys_path) and os.access(nsys_path, os.X_OK):
            return nsys_path

    # Try common NVHPC paths (adjust for your system)
    nvhpc_paths = [
        "/collab/usr/global/tools/nvidia/nvhpc/toss_4_x86_64_ib/nvhpc-25.9/Linux_x86_64/25.9/profilers/12.9/Nsight_Systems/bin/nsys",
        "/collab/usr/global/tools/nvidia/nvhpc/toss_4_x86_64_ib/nvhpc-24.9/Linux_x86_64/24.9/profilers/12.6/Nsight_Systems/bin/nsys",
    ]
    for path in nvhpc_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    # Fall back to PATH lookup
    nsys_in_path = shutil.which("nsys")
    if nsys_in_path:
        return nsys_in_path

    return None


def extract_static_gpu_memory(
    exe_path: Path, machine: str, verbose: bool, dry_run: bool
) -> int | None:
    """Extract static/global GPU memory from device code in executable.

    For AMD (HIP):
    1. llvm-objcopy --dump-section=.hip_fatbin to extract fat binary
    2. clang-offload-bundler --unbundle to get device object
    3. llvm-readelf -s to read symbols and sum OBJECT sizes

    For NVIDIA (CUDA):
    1. cuobjdump --extract-elf all to extract device cubins
    2. llvm-readelf -s on each cubin to read symbols and sum OBJECT sizes

    Returns total bytes of static GPU memory, or None on failure.
    """
    if dry_run:
        print(f"  [DRY-RUN] Would extract static GPU memory from {exe_path}")
        return 1048576  # Dummy value

    if not exe_path.exists():
        return None

    tmpdir = tempfile.mkdtemp(prefix="static_gpu_")
    try:
        if machine == "nvidia":
            return _extract_nvidia_static_memory(exe_path, tmpdir, verbose)
        elif machine == "amd":
            return _extract_amd_static_memory(exe_path, tmpdir, verbose)
        else:
            if verbose:
                print(f"  Warning: Unknown machine type for static memory: {machine}")
            return None
    except Exception as e:
        if verbose:
            print(f"  Warning: Static memory extraction failed: {e}")
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _extract_nvidia_static_memory(
    exe_path: Path, tmpdir: str, verbose: bool
) -> int | None:
    """Extract static GPU memory from NVIDIA CUDA executable using cuobjdump."""
    # Use cuobjdump to extract device ELF (cubin) files
    # Must use absolute path since we run in tmpdir
    extract_cmd = ["cuobjdump", "--extract-elf", "all", str(exe_path.resolve())]
    if verbose:
        print(f"  Running: {' '.join(extract_cmd)}")

    result = subprocess.run(
        extract_cmd, capture_output=True, text=True, cwd=tmpdir
    )

    if result.returncode != 0:
        if verbose:
            print(f"  Warning: cuobjdump failed: {result.stderr}")
        # Check if it's just "no elf" (no device code)
        if "no elf" in result.stderr.lower() or "no cubin" in result.stderr.lower():
            return 0
        return None

    # Find extracted cubin files
    tmpdir_path = Path(tmpdir)
    cubin_files = list(tmpdir_path.glob("*.cubin"))

    if not cubin_files:
        if verbose:
            print(f"  No cubin files extracted (runtime-generated code?)")
        return 0

    # Read symbols from each cubin and sum OBJECT sizes
    return _sum_allocatable_sections(cubin_files, verbose)


def _extract_amd_static_memory(
    exe_path: Path, tmpdir: str, verbose: bool
) -> int | None:
    """Extract static GPU memory from AMD HIP executable using clang tools."""
    section_name = ".hip_fatbin"
    # Common AMD GPU targets - try gfx942 first (MI300), then others
    targets = [
        "hipv4-amdgcn-amd-amdhsa--gfx942",
        "hipv4-amdgcn-amd-amdhsa--gfx90a",
        "hipv4-amdgcn-amd-amdhsa--gfx908",
    ]

    fatbin_path = Path(tmpdir) / "device.fatbin"
    device_obj = Path(tmpdir) / "device.bin"

    # Step 1: Extract fat binary section
    extract_cmd = [
        "llvm-objcopy",
        f"--dump-section={section_name}={fatbin_path}",
        str(exe_path)
    ]
    if verbose:
        print(f"  Running: {' '.join(extract_cmd)}")

    result = subprocess.run(extract_cmd, capture_output=True, text=True)
    if result.returncode != 0 or not fatbin_path.exists():
        if verbose:
            print(f"  No {section_name} section found in {exe_path.name}")
        # No fatbin means no static GPU memory (e.g., DSL generates code at runtime)
        return 0

    # Step 2: Unbundle - try each target until one works
    unbundle_success = False
    for target in targets:
        unbundle_cmd = [
            "clang-offload-bundler",
            "--unbundle",
            "--type=o",
            f"--output={device_obj}",
            f"--targets={target}",
            f"--input={fatbin_path}"
        ]
        if verbose:
            print(f"  Trying target: {target}")

        result = subprocess.run(unbundle_cmd, capture_output=True, text=True)
        if result.returncode == 0 and device_obj.exists():
            unbundle_success = True
            if verbose:
                print(f"  Successfully unbundled with target: {target}")
            break

    if not unbundle_success:
        if verbose:
            print(f"  Warning: Could not unbundle fat binary (no matching target)")
        # Can't extract symbols, but fatbin exists - report 0 rather than failing
        return 0

    # Step 3: Read symbols from device object
    return _sum_allocatable_sections([device_obj], verbose)


def _sum_allocatable_sections(device_files: list[Path], verbose: bool) -> int | None:
    """Sum sizes of allocatable sections from device object files.

    Uses section sizes (not symbol sizes) for accurate memory measurement.
    Counts sections with 'A' (alloc) flag, which are loaded into GPU memory:
    - .text: kernel machine code
    - .rodata: kernel descriptors, read-only data
    - .data: initialized static data
    - .bss: uninitialized static data
    - .jit.bitcode.*: embedded LLVM IR for JIT compilation

    This gives the true memory footprint including alignment padding.
    """
    total = 0
    seen_sections = {}  # Track sections for verbose output

    for device_file in device_files:
        readelf_cmd = ["llvm-readelf", "-S", str(device_file)]
        if verbose:
            print(f"  Running: {' '.join(readelf_cmd)}")

        result = subprocess.run(readelf_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if verbose:
                print(f"  Warning: llvm-readelf failed on {device_file.name}: {result.stderr}")
            continue

        # Parse section header output
        # Format: [Nr] Name              Type            Address          Off    Size   ES Flg
        # Example: [ 6] .rodata           PROGBITS        0000000000001480 001480 0000c0 00   A
        for line in result.stdout.splitlines():
            # Match section header lines: [ N] .name TYPE ...
            if "[" not in line or "]" not in line:
                continue

            # Skip header line
            if "Name" in line and "Type" in line:
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            try:
                # Find section name - it's the first part starting with '.'
                # Format: [ 6] .rodata PROGBITS ... -> parts = ['[', '6]', '.rodata', ...]
                name = None
                name_idx = -1
                for i, p in enumerate(parts):
                    if p.startswith(".") or p == "NULL":
                        name = p
                        name_idx = i
                        break

                if name is None:
                    continue

                # Skip metadata sections that don't contain user data
                if name in ("NULL", ".shstrtab", ".strtab", ".symtab",
                            ".comment", ".note", ".dynsym", ".dynstr",
                            ".gnu.hash", ".hash", ".dynamic", ".relro_padding"):
                    continue
                # Skip relocation sections
                if name.startswith(".rela"):
                    continue

                # Parse the rest: Type Address Off Size ES Flg
                # After name, we have: PROGBITS addr off size es flg
                remaining = parts[name_idx + 1:]
                if len(remaining) < 6:
                    continue

                # Type is first, then Address (16 hex), Off (6 hex), Size (6 hex)
                # Size is at index 3 (after Type, Address, Off)
                size_hex = remaining[3]
                size = int(size_hex, 16)

                # Flags are at index 5
                flags = remaining[5] if len(remaining) > 5 else ""

                # Only count allocatable sections (have 'A' flag)
                # These are sections that get loaded into memory
                if "A" in flags and size > 0:
                    # Build unique key for deduplication across files
                    key = f"{device_file.name}:{name}"
                    if key not in seen_sections:
                        seen_sections[key] = (name, size)
                        total += size

            except (ValueError, IndexError):
                continue

    if verbose:
        for key, (name, size) in seen_sections.items():
            print(f"    Section {name}: {size} bytes")

    return total


def extract_jit_cache_static_memory(
    bench_path: Path, verbose: bool, dry_run: bool
) -> int | None:
    """Extract static GPU memory from JIT cache files in .proteus directory.

    JIT cache files are already device object files (ELF format for AMDGPU or CUDA),
    so we can read them directly with llvm-readelf without extraction/unbundling.

    Returns total bytes of static GPU memory from cached kernels, or None on failure.
    """
    if dry_run:
        print(f"  [DRY-RUN] Would extract static GPU memory from JIT cache in {bench_path}")
        return 524288  # Dummy value

    cache_dir = bench_path / ".proteus"
    if not cache_dir.exists():
        if verbose:
            print(f"  No JIT cache directory found: {cache_dir}")
        return 0

    cache_files = list(cache_dir.glob("*.o"))
    if not cache_files:
        if verbose:
            print(f"  No cache files found in {cache_dir}")
        return 0

    if verbose:
        print(f"  Found {len(cache_files)} JIT cache file(s) in {cache_dir}")

    return _sum_allocatable_sections(cache_files, verbose)


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

    # For hip/ and cuda/ directories, we need to set ENABLE_PROTEUS and GPU_BACKEND flags
    if "/hip/" in bench_path:
        cmd = f"{cmd} GPU_BACKEND=hip"
        if frontend == "proteus":
            cmd = f"{cmd} ENABLE_PROTEUS=yes"
        elif frontend == "aot":
            cmd = f"{cmd} ENABLE_PROTEUS=no"
    elif "/cuda/" in bench_path:
        cmd = f"{cmd} GPU_BACKEND=cuda"
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


def compute_nsys_gpu_hwm(sqlite_file: Path) -> int | None:
    """Parse nsys SQLite database and compute GPU memory HWM.

    Uses SQL query to track cumulative memory from CUDA_GPU_MEMORY_USAGE_EVENTS.
    """
    query = """
WITH running_total AS (
  SELECT
    SUM(CASE WHEN memoryOperationType = 0 THEN bytes ELSE -bytes END)
      OVER (ORDER BY start) AS cumulative_bytes
  FROM CUDA_GPU_MEMORY_USAGE_EVENTS
)
SELECT MAX(cumulative_bytes) AS hwm_bytes FROM running_total;
"""
    try:
        result = subprocess.run(
            ["sqlite3", str(sqlite_file), query],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"  Warning: sqlite3 query failed: {result.stderr}", file=sys.stderr)
            return None

        # Parse output - should be a single integer
        output = result.stdout.strip()
        if not output:
            print("  Warning: No memory events found in nsys trace", file=sys.stderr)
            return None

        return int(output)
    except subprocess.TimeoutExpired:
        print("  Warning: sqlite3 query timed out", file=sys.stderr)
        return None
    except ValueError as e:
        print(f"  Warning: Could not parse HWM value: {e}", file=sys.stderr)
        return None


def run_nsys(
    bench_path: Path, exe: str, inputs: str, verbose: bool, dry_run: bool
) -> int | None:
    """Run nsys profiler and return GPU HWM in bytes (NVIDIA GPUs)."""
    nsys = find_nsys()
    if nsys is None:
        print("  Warning: nsys not found. Set NSYS env var or install NVHPC.", file=sys.stderr)
        return None

    # Create temp directory for output files
    tmpdir = tempfile.mkdtemp(prefix="nsys_")
    report_path = Path(tmpdir) / "mem_trace"

    profile_cmd = (
        f"{nsys} profile --trace=cuda --cuda-memory-usage=true "
        f"-o {report_path} ./{exe} {inputs}"
    )

    if dry_run:
        print(f"  [DRY-RUN] Would run: {profile_cmd}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return 536870912  # Dummy value for dry-run

    exe_path = bench_path / exe
    if not exe_path.exists():
        print(f"  Warning: Executable not found: {exe_path}", file=sys.stderr)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None

    if verbose:
        print(f"  Running nsys profile: {profile_cmd}")

    try:
        # Step 1: Profile the application
        result = subprocess.run(
            profile_cmd, shell=True, cwd=bench_path,
            capture_output=True, text=True, timeout=600  # 10 minute timeout
        )
        if result.returncode != 0:
            print(f"  Warning: nsys profile failed: {result.stderr}", file=sys.stderr)
            return None

        # Step 2: Export to SQLite
        nsys_rep = Path(f"{report_path}.nsys-rep")
        if not nsys_rep.exists():
            print(f"  Warning: nsys report not found: {nsys_rep}", file=sys.stderr)
            return None

        sqlite_file = Path(f"{report_path}.sqlite")
        export_cmd = f"{nsys} export --type=sqlite --output={sqlite_file} {nsys_rep}"
        if verbose:
            print(f"  Running nsys export: {export_cmd}")

        result = subprocess.run(
            export_cmd, shell=True, cwd=bench_path,
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"  Warning: nsys export failed: {result.stderr}", file=sys.stderr)
            return None

        # Step 3: Query the SQLite database for HWM
        if not sqlite_file.exists():
            print(f"  Warning: SQLite file not found: {sqlite_file}", file=sys.stderr)
            return None

        hwm = compute_nsys_gpu_hwm(sqlite_file)
        return hwm

    except subprocess.TimeoutExpired:
        print("  Warning: nsys run timed out", file=sys.stderr)
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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

            # Extract static GPU memory from compiled executable (AOT kernels + JIT bitcode)
            # For AOT: this is all the static GPU memory
            # For proteus: this includes original kernel + embedded JIT bitcode
            # For dsl/cpp: no fatbin in binary, returns 0
            exe_path = bench_path / exe
            gpu_static_binary = extract_static_gpu_memory(
                exe_path, args.machine, args.verbose, args.dry_run
            )

            # For JIT frontends, clean .proteus cache before each profiler run
            # to ensure both profilers measure with fresh JIT compilation
            is_jit_frontend = frontend in ("proteus", "dsl", "cpp")

            # Run massif for CPU HWM
            if is_jit_frontend:
                clean_proteus_cache(bench_path, args.verbose, args.dry_run)
            cpu_hwm = run_massif(bench_path, exe, inputs, args.verbose, args.dry_run)

            # For JIT frontends, extract static memory from JIT cache after massif run
            # (massif run populates the cache)
            # For proteus: add to binary static (original kernel + specialized kernel)
            # For dsl/cpp: this is the only source of static GPU memory
            gpu_static_jit = None
            if is_jit_frontend:
                gpu_static_jit = extract_jit_cache_static_memory(
                    bench_path, args.verbose, args.dry_run
                )

            # Compute total static GPU memory
            if is_jit_frontend:
                # Sum binary static (if any) + JIT cache static
                binary_val = gpu_static_binary if gpu_static_binary is not None else 0
                jit_val = gpu_static_jit if gpu_static_jit is not None else 0
                gpu_static = binary_val + jit_val
                if args.verbose:
                    print(f"  Static GPU: binary={binary_val}, jit_cache={jit_val}, total={gpu_static}")
            else:
                gpu_static = gpu_static_binary

            # Run GPU memory profiler (rocprofv3 for AMD, nsys for NVIDIA)
            if is_jit_frontend:
                clean_proteus_cache(bench_path, args.verbose, args.dry_run)
            if args.machine == "amd":
                gpu_hwm = run_rocprofv3(bench_path, exe, inputs, args.verbose, args.dry_run)
            elif args.machine == "nvidia":
                gpu_hwm = run_nsys(bench_path, exe, inputs, args.verbose, args.dry_run)
            else:
                print(f"  Warning: No GPU profiler for machine '{args.machine}'", file=sys.stderr)
                gpu_hwm = None

            # Write result
            result = {
                "benchmark": benchmark,
                "frontend": frontend,
                "cpu_hwm_bytes": cpu_hwm if cpu_hwm is not None else "",
                "gpu_hwm_bytes": gpu_hwm if gpu_hwm is not None else "",
                "gpu_static_bytes": gpu_static if gpu_static is not None else "",
            }

            if args.verbose:
                print(f"  CPU HWM: {cpu_hwm}, GPU HWM: {gpu_hwm}, GPU Static: {gpu_static}")

            if not args.dry_run:
                write_result(args.output, result, write_header)
                write_header = False  # Only write header once

    print(f"Done. Results written to {args.output}")


if __name__ == "__main__":
    main()
