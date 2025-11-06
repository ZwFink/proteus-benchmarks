#!/usr/bin/env python3
"""
Plot per-benchmark slowdowns relative to cached Proteus executions.

The script compares uncached direct-mode runs (Proteus annotations/DSL/CPP)
against cached Proteus baselines and emits bar plots per platform.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import pandas as pd
from matplotlib import use as mpl_use
from plotnine import (
    aes,
    element_blank,
    element_text,
    geom_col,
    geom_hline,
    ggplot,
    labs,
    position_dodge,
    scale_fill_manual,
    theme,
    theme_seaborn,
)

mpl_use("Agg")

# Frontends we consider for slowdown reporting.
TARGET_FRONTENDS: Tuple[str, ...] = ("proteus", "dsl", "cpp")

# Shared labeling conventions.
FRONTEND_LABELS: Dict[str, str] = {
    "proteus": "PJ-Annot.",
    "dsl": "PJ-DSL",
    "cpp": "PJ-CPP",
}

FRONTEND_COLORS: Dict[str, str] = {
    "PJ-Annot.": "#000000",
    "PJ-CPP": "#FF0066",
    "PJ-DSL": "#107F80",
}

BENCHMARK_LABELS: Dict[str, str] = {
    "3mm": "3mm",
    "adam": "Adam",
    "attention": "Attention",
    "bezier-surface": "Bezier-Surface",
    "conv3d": "Conv3D",
    "floyd-warshall": "Floyd-Warshall",
    "gemm": "GEMM",
    "minibude": "MiniBUDE",
}


@dataclass(frozen=True)
class RuntimeKey:
    platform: str
    benchmark: str
    frontend: str


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create slowdown plots relative to cached Proteus baselines."
    )
    parser.add_argument(
        "--uncached-dir",
        required=True,
        type=Path,
        help="Directory containing *-results-direct.csv from uncached runs.",
    )
    parser.add_argument(
        "--cached-dir",
        required=True,
        type=Path,
        help="Directory containing *-results-direct.csv for cached Proteus runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store generated plots (defaults to --uncached-dir).",
    )
    parser.add_argument(
        "--output-basename",
        default="slowdown-vs-cache",
        help="Base filename for saved plots (extensions and platform suffixes added).",
    )
    parser.add_argument(
        "--platform",
        choices=("amd", "nvidia"),
        help="Optional platform filter; plots only the specified platform when set.",
    )
    return parser.parse_args(argv)


def discover_result_files(results_dir: Path) -> list[Path]:
    pattern = "*-results-direct.csv"
    return sorted(results_dir.glob(pattern))


def parse_metadata(csv_path: Path) -> RuntimeKey:
    """
    Extract metadata from filenames like 'amd-3mm-proteus-results-direct.csv'.
    """
    stem = csv_path.stem
    suffix = "-results-direct"
    if not stem.endswith(suffix):
        raise ValueError(f"Unexpected filename format: {csv_path.name}")
    payload = stem[: -len(suffix)]
    parts = payload.split("-")
    if len(parts) < 3:
        raise ValueError(f"Filename '{csv_path.name}' is too short to parse.")

    platform = parts[0]
    frontend = parts[-1]
    benchmark = "-".join(parts[1:-1])
    if not benchmark:
        raise ValueError(f"Filename '{csv_path.name}' is missing benchmark token(s).")
    return RuntimeKey(platform=platform, benchmark=benchmark, frontend=frontend)


def load_average_runtime(csv_path: Path) -> float:
    df = pd.read_csv(csv_path)
    if "ExeTime" not in df.columns:
        raise KeyError(f"'ExeTime' column missing from {csv_path}")
    runtimes = df.copy()
    if "repeat" in runtimes.columns:
        runtimes = runtimes[runtimes["repeat"] != 0]
    runtimes = pd.to_numeric(runtimes["ExeTime"], errors="coerce").dropna()
    if runtimes.empty:
        raise ValueError(f"No valid runtimes found in {csv_path}")
    return float(runtimes.mean())


def collect_uncached_runtimes(results_dir: Path) -> Dict[RuntimeKey, float]:
    runtimes: Dict[RuntimeKey, float] = {}
    for csv_path in discover_result_files(results_dir):
        try:
            meta = parse_metadata(csv_path)
        except ValueError as exc:
            print(f"Skipping '{csv_path.name}': {exc}", file=sys.stderr)
            continue

        if meta.frontend not in TARGET_FRONTENDS:
            continue

        try:
            runtimes[meta] = load_average_runtime(csv_path)
        except (KeyError, ValueError) as exc:
            print(f"Skipping '{csv_path.name}': {exc}", file=sys.stderr)
    return runtimes


def collect_cached_baselines(results_dir: Path) -> Dict[Tuple[str, str], float]:
    """
    Collect cached Proteus runtimes keyed by (platform, benchmark).
    """
    baselines: Dict[Tuple[str, str], float] = {}
    for csv_path in discover_result_files(results_dir):
        try:
            meta = parse_metadata(csv_path)
        except ValueError as exc:
            print(f"Skipping '{csv_path.name}': {exc}", file=sys.stderr)
            continue

        if meta.frontend != "proteus":
            continue

        key = (meta.platform, meta.benchmark)
        try:
            baselines[key] = load_average_runtime(csv_path)
        except (KeyError, ValueError) as exc:
            print(f"Skipping cached baseline '{csv_path.name}': {exc}", file=sys.stderr)
    return baselines


def compute_slowdowns(
    uncached: Mapping[RuntimeKey, float],
    baselines: Mapping[Tuple[str, str], float],
) -> pd.DataFrame:
    rows = []
    missing_baselines: set[Tuple[str, str]] = set()

    for meta, runtime in uncached.items():
        baseline = baselines.get((meta.platform, meta.benchmark))
        if baseline is None:
            missing_baselines.add((meta.platform, meta.benchmark))
            continue
        if baseline <= 0 or runtime <= 0:
            print(
                f"Skipping {meta} due to non-positive runtimes "
                f"(uncached={runtime}, baseline={baseline})",
                file=sys.stderr,
            )
            continue
        slowdown = runtime / baseline
        rows.append(
            {
                "platform": meta.platform,
                "benchmark": meta.benchmark,
                "frontend": meta.frontend,
                "slowdown": slowdown,
            }
        )

    if missing_baselines:
        missing_text = ", ".join(
            sorted(f"{plat}:{bench}" for plat, bench in missing_baselines)
        )
        print(
            f"Warning: missing cached Proteus baselines for {missing_text}",
            file=sys.stderr,
        )

    if not rows:
        raise RuntimeError(
            "No slowdown data computed. Ensure cached and uncached directories align."
        )

    df = pd.DataFrame(rows)
    df["frontend_label"] = df["frontend"].map(FRONTEND_LABELS)
    df = df[df["frontend_label"].notna()].copy()

    fallback_labels = (
        df["benchmark"]
        .str.replace("-", " ", regex=False)
        .str.title()
        .str.replace(" ", "-", regex=False)
    )
    df["benchmark_label"] = (
        df["benchmark"].map(BENCHMARK_LABELS).fillna(fallback_labels)
    )

    return df


def build_plot(df: pd.DataFrame) -> ggplot:
    df = df.copy()
    df["benchmark_label"] = pd.Categorical(
        df["benchmark_label"], categories=sorted(df["benchmark_label"].unique()), ordered=True
    )
    frontend_order = [FRONTEND_LABELS[name] for name in TARGET_FRONTENDS]
    df["frontend_label"] = pd.Categorical(
        df["frontend_label"], categories=frontend_order, ordered=True
    )

    dodge = position_dodge(width=0.75)
    return (
        ggplot(df, aes("benchmark_label", "slowdown", fill="frontend_label"))
        + geom_col(position=dodge, width=0.6)
        + geom_hline(yintercept=1.0, linetype="dashed", color="#333333", size=1.2)
        + scale_fill_manual(values=FRONTEND_COLORS)
        + labs(x="", y="Uncached Slowdown", fill="")
        + theme_seaborn(style="whitegrid")
        + theme(
            axis_text_x=element_text(rotation=45, ha="right", size=14),
            axis_text_y=element_text(size=16),
            axis_title_y=element_text(size=18),
            axis_title_x=element_text(size=16),
            legend_position="top",
            legend_direction="horizontal",
            legend_text=element_text(size=14),
            panel_grid_major_x=element_blank(),
            panel_grid_minor_x=element_blank(),
            panel_grid_major_y=element_blank(),
            figure_size=(6.4, 4.8),
        )
    )


def save_plot(plot: ggplot, output_dir: Path, basename: str, platform: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{basename}-{platform}"
    plot.save(f"{stem}.png", dpi=300, bbox_inches="tight")
    plot.save(f"{stem}.pdf", dpi=300, bbox_inches="tight")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    uncached_dir: Path = args.uncached_dir
    cached_dir: Path = args.cached_dir
    output_dir: Path = args.output_dir or uncached_dir

    if not uncached_dir.is_dir():
        print(f"Uncached directory '{uncached_dir}' is not a directory.", file=sys.stderr)
        return 2
    if not cached_dir.is_dir():
        print(f"Cached directory '{cached_dir}' is not a directory.", file=sys.stderr)
        return 2

    uncached = collect_uncached_runtimes(uncached_dir)
    baselines = collect_cached_baselines(cached_dir)

    try:
        slowdown_df = compute_slowdowns(uncached, baselines)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.platform is not None:
        slowdown_df = slowdown_df[slowdown_df["platform"] == args.platform]
        if slowdown_df.empty:
            print(
                f"No slowdown data for platform '{args.platform}'.",
                file=sys.stderr,
            )
            return 1

        plot = build_plot(slowdown_df)
        save_plot(plot, output_dir, args.output_basename, args.platform)
        print(
            f"Wrote slowdown plots for {args.platform} to "
            f"{output_dir / (args.output_basename + '-' + args.platform)}.[png|pdf]"
        )
        return 0

    for platform, platform_df in slowdown_df.groupby("platform"):
        plot = build_plot(platform_df)
        save_plot(plot, output_dir, args.output_basename, platform)
        print(
            f"Wrote slowdown plots for {platform} to "
            f"{output_dir / (args.output_basename + '-' + platform)}.[png|pdf]"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
