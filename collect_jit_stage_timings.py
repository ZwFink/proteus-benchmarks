#!/usr/bin/env python3
"""
Aggregate JIT compilation stage timings from Proteus benchmark stdout logs.

The script scans a results directory for files named like
``<platform>-<benchmark>-<approach>-stdouterr.txt``. For each file it captures
the first `=> Execute ...` block (the uncached run), sums the timings for three
high-level stages, and emits a CSV with one row per file.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

_VALUE_RE = r"(?P<value>[-+]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)"

# Stage regexes
_SPECIALIZED_KERNEL = re.compile(
    r"\[proteus\]\s*specialized kernel construction.*?" + _VALUE_RE + r"\s*ms",
    re.IGNORECASE,
)
_PARSE_IR = re.compile(
    r"\[proteus\]\s*parse ir.*?" + _VALUE_RE + r"\s*ms", re.IGNORECASE
)
_CLONING = re.compile(
    r"\[proteus\]\s*cloning cross-clone.*?" + _VALUE_RE + r"\s*ms", re.IGNORECASE
)
_EXTRACT_KERNEL = re.compile(
    r"\[proteus\]\s*extract kernel module.*?" + _VALUE_RE + r"\s*ms", re.IGNORECASE
)
_OPTIMIZE_IR = re.compile(
    r"\[proteus\]\s*optimizeir.*?" + _VALUE_RE + r"\s*ms", re.IGNORECASE
)
_COMPILE_CPP_TO_IR = re.compile(
    r"\[proteus\]\s*compile\s+c\+\+\s*to\s*ir.*?" + _VALUE_RE + r"\s*ms",
    re.IGNORECASE,
)
_CODEGEN_OBJECT = re.compile(
    r"\[proteus\]\s*codegen object.*?" + _VALUE_RE + r"\s*ms", re.IGNORECASE
)
_CODEGEN_LINKING = re.compile(
    r"\[proteus\]\s*codegen linking.*?" + _VALUE_RE + r"\s*ms", re.IGNORECASE
)
_CODEGEN_PTX = re.compile(
    r"\[proteus\]\s*codegen ptx.*?" + _VALUE_RE + r"\s*ms", re.IGNORECASE
)
_CODEGEN_CUDA_RTC = re.compile(
    r"\[proteus\]\s*codegen cuda rtc.*?" + _VALUE_RE + r"\s*ms", re.IGNORECASE
)


class StageConfig(NamedTuple):
    specialized: Sequence[re.Pattern[str]]
    optimized_ir: Sequence[re.Pattern[str]]


_APPROACH_CONFIGS: Dict[str, StageConfig] = {
    "dsl": StageConfig(
        specialized=[_SPECIALIZED_KERNEL],
        optimized_ir=[_OPTIMIZE_IR],
    ),
    "cpp": StageConfig(
        specialized=[_SPECIALIZED_KERNEL],
        optimized_ir=[_COMPILE_CPP_TO_IR],
    ),
    "proteus": StageConfig(
        specialized=[_PARSE_IR, _CLONING, _EXTRACT_KERNEL, _SPECIALIZED_KERNEL],
        optimized_ir=[_OPTIMIZE_IR],
    ),
}

_DEVICE_PATTERNS = {
    "amd": [_CODEGEN_OBJECT, _CODEGEN_LINKING],
    "nvidia": [_CODEGEN_PTX, _CODEGEN_CUDA_RTC],
}


class ParsedName(NamedTuple):
    platform: str
    benchmark: str
    approach: str


class StageStats(NamedTuple):
    mean: float
    minimum: float
    maximum: float
    median: float
    p25: float
    p75: float


def _sum_patterns(lines: Iterable[str], patterns: Sequence[re.Pattern[str]]) -> float:
    """Sum all numeric values matched by the provided regex patterns."""
    total = 0.0
    for line in lines:
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                try:
                    total += float(match.group("value"))
                except ValueError:
                    # Should not happen, but continue gracefully.
                    continue
                break
    return total


def _is_run_command(line: str) -> bool:
    """Return True if the execute line launches the benchmark binary."""
    if not line.startswith("=> Execute "):
        return False
    command = line[len("=> Execute ") :].strip()
    if not command:
        return False
    if command.startswith("make"):
        return False
    if command.startswith(("rm ", "cp ", "mv ")):
        return False
    if ".x" in command or ".exe" in command:
        return True
    # Fall back to excluding obvious build invocations.
    return "/benchmarks/" in command


def _extract_run_blocks(path: Path) -> List[List[str]]:
    """Return all benchmark execution blocks (one per rep)."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        lines = [raw.rstrip("\n") for raw in handle]

    blocks: List[List[str]] = []
    awaiting_run = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "=== End of Experiment ===":
            awaiting_run = True
            i += 1
            continue

        if awaiting_run and line.startswith("=> Execute "):
            if not _is_run_command(line):
                # Still waiting for the benchmark execution within this rep.
                i += 1
                continue

            block: List[str] = [line]
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.startswith("=> Execute ") or nxt.strip() == "=== Experiment ===":
                    break
                block.append(nxt)
                i += 1
            blocks.append(block)
            awaiting_run = False
            continue

        i += 1

    return blocks


def _parse_name(path: Path) -> Optional[ParsedName]:
    stem = path.stem  # e.g. amd-3mm-dsl-stdouterr
    parts = stem.split("-")
    if len(parts) < 4:
        print(f"Skipping {path}: unexpected filename format.", file=sys.stderr)
        return None
    platform = parts[0].lower()
    approach = parts[-2].lower()
    benchmark = "-".join(parts[1:-2])
    if not benchmark:
        print(f"Skipping {path}: benchmark name missing.", file=sys.stderr)
        return None
    return ParsedName(platform=platform, benchmark=benchmark, approach=approach)


def _collect_for_file(
    path: Path,
) -> Optional[Tuple[ParsedName, StageStats, StageStats, StageStats]]:
    parsed = _parse_name(path)
    if not parsed:
        return None
    stage_config = _APPROACH_CONFIGS.get(parsed.approach)
    if not stage_config:
        print(
            f"Skipping {path}: no stage configuration for approach '{parsed.approach}'.",
            file=sys.stderr,
        )
        return None
    device_patterns = _DEVICE_PATTERNS.get(parsed.platform)
    if not device_patterns:
        print(
            f"Skipping {path}: unrecognised platform '{parsed.platform}'.",
            file=sys.stderr,
        )
        return None

    blocks = _extract_run_blocks(path)
    if not blocks:
        print(f"No execute blocks found in {path}.", file=sys.stderr)
        return None

    specialized_totals: List[float] = []
    optimized_totals: List[float] = []
    device_totals: List[float] = []

    for block in blocks:
        specialized_totals.append(_sum_patterns(block, stage_config.specialized))
        optimized_totals.append(_sum_patterns(block, stage_config.optimized_ir))
        device_totals.append(_sum_patterns(block, device_patterns))

    if not any(value > 0.0 for value in specialized_totals):
        print(
            f"Warning: specialized stage not detected in {path} (all reps).",
            file=sys.stderr,
        )
    if not any(value > 0.0 for value in optimized_totals):
        print(
            f"Warning: optimized IR stage not detected in {path} (all reps).",
            file=sys.stderr,
        )
    if not any(value > 0.0 for value in device_totals):
        print(
            f"Warning: device binary stage not detected in {path} (all reps).",
            file=sys.stderr,
        )

    specialized_stats = _compute_stats(specialized_totals)
    optimized_stats = _compute_stats(optimized_totals)
    device_stats = _compute_stats(device_totals)

    return parsed, specialized_stats, optimized_stats, device_stats


def _compute_stats(values: Sequence[float]) -> StageStats:
    """Compute summary statistics for the provided series."""
    if not values:
        return StageStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    arr = sorted(values)
    mean_val = sum(arr) / len(arr)
    min_val = arr[0]
    max_val = arr[-1]
    median_val = _percentile(arr, 0.5)
    p25_val = _percentile(arr, 0.25)
    p75_val = _percentile(arr, 0.75)
    return StageStats(mean_val, min_val, max_val, median_val, p25_val, p75_val)


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Compute percentile using linear interpolation on sorted data."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    index = fraction * (n - 1)
    lower = int(index)
    upper = min(lower + 1, n - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _write_csv(
    rows: Sequence[Tuple[ParsedName, StageStats, StageStats, StageStats]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "platform",
                "benchmark",
                "jit_approach",
                "specialized_mean_ms",
                "specialized_min_ms",
                "specialized_max_ms",
                "specialized_median_ms",
                "specialized_p25_ms",
                "specialized_p75_ms",
                "optimized_mean_ms",
                "optimized_min_ms",
                "optimized_max_ms",
                "optimized_median_ms",
                "optimized_p25_ms",
                "optimized_p75_ms",
                "device_mean_ms",
                "device_min_ms",
                "device_max_ms",
                "device_median_ms",
                "device_p25_ms",
                "device_p75_ms",
            ]
        )
        for parsed, specialized, optimized, device in rows:
            writer.writerow(
                [
                    parsed.platform,
                    parsed.benchmark,
                    parsed.approach,
                    specialized.mean,
                    specialized.minimum,
                    specialized.maximum,
                    specialized.median,
                    specialized.p25,
                    specialized.p75,
                    optimized.mean,
                    optimized.minimum,
                    optimized.maximum,
                    optimized.median,
                    optimized.p25,
                    optimized.p75,
                    device.mean,
                    device.minimum,
                    device.maximum,
                    device.median,
                    device.p25,
                    device.p75,
                ]
            )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Proteus JIT stage timings from stdouterr logs."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory that contains <platform>-<benchmark>-<approach>-stdouterr.txt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the CSV to write (default: <results_dir>/jit-stage-timings.csv).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    results_dir: Path = args.results_dir.expanduser()
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory.", file=sys.stderr)
        return 1

    files = sorted(results_dir.glob("*-stdouterr.txt"))
    if not files:
        print(f"No stdouterr logs found in {results_dir}.", file=sys.stderr)
        return 1

    rows: List[Tuple[ParsedName, StageStats, StageStats, StageStats]] = []
    for path in files:
        collected = _collect_for_file(path)
        if collected:
            rows.append(collected)

    if not rows:
        print("No stage timings collected; exiting without writing CSV.", file=sys.stderr)
        return 1

    output_path = args.output
    if output_path is None:
        output_path = results_dir / "jit-stage-timings.csv"
    else:
        output_path = output_path.expanduser()

    _write_csv(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
