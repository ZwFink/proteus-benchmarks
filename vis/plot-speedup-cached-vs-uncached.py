#!/usr/bin/env python3
"""
Plot cached vs. uncached Proteus speedups with Matplotlib but plotnine-like styling.

Cached bars are solid; uncached bars reuse the same colors but add hatching.
Both states are normalized to cached AoT runtimes.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import use as mpl_use
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

mpl_use("Agg")

TARGET_FRONTENDS: Tuple[str, ...] = ("proteus", "dsl", "cpp")

FRONTEND_LABELS: Dict[str, str] = {
    "proteus": "PJ-Annot.",
    "dsl": "PJ-DSL",
    "cpp": "PJ-CPP",
}

FRONTEND_COLORS: Dict[str, str] = {
    "PJ-Annot.": "#0077BB",
    "PJ-DSL": "#009988",
    "PJ-CPP": "#33BBEE",
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

PLOTNINE_PANEL_FACE = "#EBEBEB"
PLOTNINE_FIGURE_FACE = "#FFFFFF"
PLOTNINE_STRIP_FACE = "#D9D9D9"
PLOTNINE_STRIP_EDGE = "#B6B6B6"
PLOTNINE_GRID_MAJOR = "#FFFFFF"
PLOTNINE_AXIS_COLOR = "#333333"
PLOTNINE_TEXT_COLOR = "#1A1A1A"


def configure_plotnine_defaults() -> None:
    """Align Matplotlib defaults with plotnine's gray theme."""

    plt.rcParams.update(
        {
            "axes.edgecolor": PLOTNINE_AXIS_COLOR,
            "axes.labelcolor": PLOTNINE_TEXT_COLOR,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "figure.facecolor": PLOTNINE_FIGURE_FACE,
            "grid.color": PLOTNINE_GRID_MAJOR,
            "grid.linestyle": "-",
            "grid.linewidth": 0.9,
            "legend.fontsize": 11,
            "text.color": PLOTNINE_TEXT_COLOR,
            "xtick.color": PLOTNINE_AXIS_COLOR,
            "ytick.color": PLOTNINE_AXIS_COLOR,
            # Preserve editable text in vector formats for Illustrator workflows.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            # 'hatch.linewidth': 2,
        }
    )


configure_plotnine_defaults()


@dataclass(frozen=True)
class RuntimeKey:
    platform: str
    benchmark: str
    frontend: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create cached vs. uncached speedup plots normalized to cached AoT baselines."
    )
    parser.add_argument(
        "--cached-dir",
        required=True,
        type=Path,
        help="Directory containing cached *-results-direct.csv files.",
    )
    parser.add_argument(
        "--uncached-dir",
        required=True,
        type=Path,
        help="Directory containing uncached *-results-direct.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store generated plots (defaults to --uncached-dir).",
    )
    parser.add_argument(
        "--output-basename",
        default="cached-vs-uncached-speedup",
        help="Base filename for saved plots (platform suffix + extensions appended).",
    )
    parser.add_argument(
        "--platform",
        choices=("amd", "nvidia"),
        help="Optional platform filter; plots only the specified platform when set.",
    )
    return parser.parse_args()


def discover_result_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*-results-direct.csv"))


def parse_metadata(csv_path: Path) -> RuntimeKey:
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


def collect_runtimes(results_dir: Path) -> Dict[RuntimeKey, float]:
    runtimes: Dict[RuntimeKey, float] = {}
    for csv_path in discover_result_files(results_dir):
        try:
            meta = parse_metadata(csv_path)
        except ValueError:
            continue

        try:
            runtimes[meta] = load_average_runtime(csv_path)
        except (KeyError, ValueError):
            continue
    return runtimes


def collect_cached_aot_baselines(
    runtimes: Dict[RuntimeKey, float]
) -> Dict[Tuple[str, str], float]:
    baselines: Dict[Tuple[str, str], float] = {}
    for meta, runtime in runtimes.items():
        if meta.frontend != "aot" or runtime <= 0:
            continue
        baselines[(meta.platform, meta.benchmark)] = runtime
    return baselines


def label_benchmark(name: str) -> str:
    return BENCHMARK_LABELS.get(
        name,
        name.replace("-", " ").title().replace(" ", "-"),
    )


def build_dataset(
    cached_runtimes: Dict[RuntimeKey, float],
    uncached_runtimes: Dict[RuntimeKey, float],
    cached_baselines: Dict[Tuple[str, str], float],
) -> pd.DataFrame:
    rows = []
    skips: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    for cache_state, runtimes in (
        ("cached", cached_runtimes),
        ("uncached", uncached_runtimes),
    ):
        for meta, runtime in runtimes.items():
            if meta.frontend not in TARGET_FRONTENDS:
                continue
            baseline = cached_baselines.get((meta.platform, meta.benchmark))
            if baseline is None:
                skips[(meta.platform, meta.benchmark)].add("missing cached AoT baseline")
                continue
            if baseline <= 0:
                skips[(meta.platform, meta.benchmark)].add("non-positive cached AoT baseline")
                continue
            if runtime <= 0:
                skips[(meta.platform, meta.benchmark)].add(f"non-positive {cache_state} runtime")
                continue
            frontend_label = FRONTEND_LABELS.get(meta.frontend)
            if frontend_label is None:
                skips[(meta.platform, meta.benchmark)].add(f"unsupported frontend '{meta.frontend}'")
                continue
            rows.append(
                {
                    "platform": meta.platform,
                    "benchmark": meta.benchmark,
                    "benchmark_label": label_benchmark(meta.benchmark),
                    "frontend": meta.frontend,
                    "frontend_label": frontend_label,
                    "cache_state": cache_state,
                    "runtime_seconds": runtime,
                    "cached_aot_runtime_seconds": baseline,
                    "speedup": baseline / runtime,
                }
            )
    if skips:
        for (platform, benchmark), reasons in sorted(skips.items()):
            reason_str = "; ".join(sorted(reasons))
            print(
                f"[INFO] Skipping {platform}:{benchmark} -> {reason_str}",
                file=sys.stderr,
            )
    return pd.DataFrame(rows)


def write_results_csv(df: pd.DataFrame, output_dir: Path, basename: str) -> Path:
    """Persist the combined dataset so plot inputs are reproducible."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{basename}.csv"
    export_columns = [
        "platform",
        "benchmark",
        "benchmark_label",
        "frontend",
        "frontend_label",
        "cache_state",
        "runtime_seconds",
        "cached_aot_runtime_seconds",
        "speedup",
    ]
    available_columns = [col for col in export_columns if col in df.columns]
    ordered_df = df.sort_values(["platform", "benchmark_label", "frontend_label", "cache_state"]).copy()
    ordered_df.to_csv(csv_path, index=False, columns=available_columns)
    return csv_path


def prepare_platform_frame(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    platform_df = df[df["platform"] == platform].copy()
    if platform_df.empty:
        raise RuntimeError(f"No data available for platform '{platform}'.")

    present_frontends = platform_df["frontend_label"].unique().tolist()
    ordered_frontends = [
        FRONTEND_LABELS[name]
        for name in TARGET_FRONTENDS
        if FRONTEND_LABELS.get(name) in present_frontends
    ]
    if not ordered_frontends:
        raise RuntimeError(f"No supported frontends found for platform '{platform}'.")

    benchmark_labels = sorted(platform_df["benchmark_label"].unique())
    platform_df["benchmark_label"] = pd.Categorical(
        platform_df["benchmark_label"], categories=benchmark_labels, ordered=True
    )
    platform_df["frontend_label"] = pd.Categorical(
        platform_df["frontend_label"], categories=ordered_frontends, ordered=True
    )
    platform_df["cache_state"] = pd.Categorical(
        platform_df["cache_state"], categories=["cached", "uncached"], ordered=True
    )
    return platform_df


def _lookup_speedup(
    df: pd.DataFrame, benchmark: str, frontend: str, cache_state: str
) -> float | None:
    subset = df[
        (df["benchmark_label"] == benchmark)
        & (df["frontend_label"] == frontend)
        & (df["cache_state"] == cache_state)
    ]
    if subset.empty:
        return None
    return float(subset["speedup"].iloc[0])


def render_platform_plot(platform_df: pd.DataFrame, platform: str, destination: Path) -> None:
    ordered_frontends = list(platform_df["frontend_label"].cat.categories)
    benchmarks = list(platform_df["benchmark_label"].cat.categories)
    if not benchmarks:
        raise RuntimeError(f"No benchmarks present for platform '{platform}'.")

    n_benchmarks = len(benchmarks)
    nrows = 1
    ncols = n_benchmarks

    # Match narrow facet layout similar to plotnine grid (about 1.5" per facet).
    fig_width = max(10, 1.55 * ncols + 1.5)
    fig_height = 4.5
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharey=False,
    )
    fig.patch.set_facecolor(PLOTNINE_FIGURE_FACE)
    fig.subplots_adjust(wspace=0.02)

    bar_colors = {label: FRONTEND_COLORS.get(label, "#888888") for label in ordered_frontends}
    x_spacing = 1.25
    x_positions = np.arange(len(ordered_frontends)) * x_spacing
    x_margin = x_spacing * 0.75
    bar_width = 0.95
    bar_edge_width = 1.05

    for idx, benchmark in enumerate(benchmarks):
        ax = axes.flat[idx]
        ax.set_facecolor("#FFFFFF")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=PLOTNINE_GRID_MAJOR, linestyle="-", linewidth=0.9, zorder=0)
        ax.xaxis.grid(False)

        for spine_name, spine in ax.spines.items():
            if spine_name in ("top", "right"):
                spine.set_visible(False)
            else:
                spine.set_color(PLOTNINE_AXIS_COLOR)
                spine.set_linewidth(0.8)

        for x_pos, frontend in zip(x_positions, ordered_frontends):
            color = bar_colors[frontend]
            cached_speedup = _lookup_speedup(platform_df, benchmark, frontend, "cached")
            if cached_speedup is not None:
                ax.bar(
                    x_pos,
                    cached_speedup,
                    width=bar_width,
                    color=color,
                    edgecolor=color,
                    linewidth=bar_edge_width,
                    zorder=2,
                )

            uncached_speedup = _lookup_speedup(platform_df, benchmark, frontend, "uncached")
            if uncached_speedup is not None:
                ax.bar(
                    x_pos,
                    uncached_speedup,
                    width=bar_width,
                    color=color,
                    edgecolor="#111111",
                    linewidth=bar_edge_width,
                    hatch="//",
                    zorder=3,
                )

        ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.6, zorder=4)

        strip_height = 0.12
        ax.add_patch(
            Rectangle(
                (0.0, 1.0),
                1.0,
                strip_height,
                transform=ax.transAxes,
                facecolor=PLOTNINE_STRIP_FACE,
                edgecolor="none",
                clip_on=False,
                zorder=4,
            )
        )
        ax.text(
            0.5,
            1.0 + strip_height / 2,
            benchmark,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=PLOTNINE_TEXT_COLOR,
            zorder=5,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels([])
        last_pos = x_positions[-1] if len(x_positions) else 0
        ax.set_xlim(-x_margin, last_pos + x_margin)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=11, colors=PLOTNINE_AXIS_COLOR)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        if idx % ncols == 0:
            ax.set_ylabel("Speedup", fontsize=13)
            ax.yaxis.label.set_color(PLOTNINE_AXIS_COLOR)

    # Remove unused axes if benchmarks do not fill the grid.
    total_axes = nrows * ncols
    for idx in range(n_benchmarks, total_axes):
        fig.delaxes(axes.flat[idx])

    frontend_handles = [
        Patch(facecolor=bar_colors[label], edgecolor="none", label=label) for label in ordered_frontends
    ]
    cache_handles = [
        Patch(facecolor="#CFCFCF", edgecolor="none", label="Cached"),
        Patch(facecolor="#CFCFCF", edgecolor="#111111", hatch="//", label="Uncached"),
    ]

    legend_handles = frontend_handles + cache_handles
    legend_labels = [handle.get_label() for handle in legend_handles]
    legend = fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=max(2, len(frontend_handles)),
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    fig.suptitle(
        f"{platform.upper()} Cached vs. Uncached Speedups",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    destination.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        output_path = destination.with_suffix(f".{extension}")
        save_kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            save_kwargs["dpi"] = 300
        fig.savefig(output_path, **save_kwargs)

    plt.close(fig)


def main() -> None:
    args = parse_args()
    cached_dir = args.cached_dir
    uncached_dir = args.uncached_dir

    if not cached_dir.is_dir():
        raise FileNotFoundError(f"{cached_dir} is not a directory.")
    if not uncached_dir.is_dir():
        raise FileNotFoundError(f"{uncached_dir} is not a directory.")

    cached_runtimes = collect_runtimes(cached_dir)
    uncached_runtimes = collect_runtimes(uncached_dir)
    if not cached_runtimes:
        raise RuntimeError("No cached runtimes found.")
    if not uncached_runtimes:
        raise RuntimeError("No uncached runtimes found.")

    cached_baselines = collect_cached_aot_baselines(cached_runtimes)
    if not cached_baselines:
        raise RuntimeError("Cached AoT baselines are required but missing.")

    dataset = build_dataset(cached_runtimes, uncached_runtimes, cached_baselines)
    if dataset.empty:
        raise RuntimeError("Combined dataset is empty after filtering.")

    if args.platform is not None:
        dataset = dataset[dataset["platform"] == args.platform]
        if dataset.empty:
            raise RuntimeError(f"No data available for platform '{args.platform}'.")

    output_dir = args.output_dir or uncached_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / args.output_basename
    csv_path = write_results_csv(dataset, output_dir, args.output_basename)
    print(f"[INFO] Wrote dataset CSV to {csv_path}")

    for platform in sorted(dataset["platform"].unique()):
        platform_df = prepare_platform_frame(dataset, platform)
        destination = Path(f"{base}-{platform}")
        render_platform_plot(platform_df, platform, destination)


if __name__ == "__main__":
    main()
