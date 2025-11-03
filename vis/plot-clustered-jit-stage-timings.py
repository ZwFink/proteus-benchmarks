#!/usr/bin/env python3
"""
Plot clustered + stacked bar charts for Proteus JIT stage timings.

The generated figure groups benchmarks along the x-axis. Each benchmark
contains one stacked bar per JIT approach, with the stack segments showing
median times for the specialized, optimized IR, and device stages.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from plotnine import *

mpl_use("Agg")

FRONTEND_LABELS: dict[str, str] = {
    "aot": "AoT",
    "proteus": "PJ-Annot.",
    "dsl": "PJ-DSL",
    "cpp": "PJ-CPP",
}

BENCHMARK_LABELS: dict[str, str] = {
    "3mm": "3mm",
    "adam": "Adam",
    "attention": "Attention",
    "bezier-surface": "Bezier-Surf.",
    "conv3d": "Conv3D",
    "floyd-warshall": "Floyd-Warsh.",
    "gemm": "GEMM",
    "minibude": "MiniBUDE",
}


FRONTEND_ORDER: list[str] = ["aot", "proteus", "dsl", "cpp"]

AXIS_TEXT_SIZE: int = 16
AXIS_TITLE_X_SIZE: int = 18
AXIS_TITLE_Y_SIZE: int = 25

STAGE_DEFINITIONS: OrderedDict[str, tuple[str, str]] = OrderedDict(
    [
        ("specialized_median_ms", ("SKC", "#FFB000")),
        ("optimized_median_ms", ("Optim. IR Gen.", "#AA66FF")),
        ("device_median_ms", ("Dev. Kernel Gen.", "#66CCFE")),
    ]
)

HATCH_PATTERNS: tuple[str, ...] = ("///", "\\\\", "..", "++", "xx", "--", "**")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a clustered + stacked bar chart from jit-stage-timings.csv."
        )
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to jit-stage-timings.csv produced by collect_jit_stage_timings.py.",
    )
    parser.add_argument(
        "--platform",
        help="Platform to plot (amd|nvidia). If omitted, inferred from the CSV when unique.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store the generated plots (defaults to csv_path.parent).",
    )
    parser.add_argument(
        "--output-basename",
        default="jit-stage-timings-clustered",
        help="Base name for saved plots (extensions are appended automatically).",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=11.0,
        help="Figure width in inches (default: 11.0).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=6.0,
        help="Figure height in inches (default: 6.0).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for the raster output (default: 300).",
    )
    return parser.parse_args()


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def infer_or_validate_platform(df: pd.DataFrame, requested: str | None) -> str:
    platforms = df["platform"].dropna().astype(str).unique()
    if requested:
        matched = df["platform"].astype(str) == requested
        if not matched.any():
            available = ", ".join(sorted(platforms)) or "<none>"
            raise ValueError(
                f"No rows found for platform '{requested}'. Available platforms: {available}."
            )
        return requested

    if len(platforms) == 1:
        return platforms[0]

    raise ValueError(
        "Multiple platforms present in the CSV; please pass --platform to select one."
    )


def prepare_dataframe(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    required = set(STAGE_DEFINITIONS.keys()) | {"platform", "benchmark", "jit_approach"}
    missing = required - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {missing_list}")

    df = df[df["platform"] == platform].copy()
    if df.empty:
        raise ValueError(f"No rows remain after filtering for platform '{platform}'.")

    value_vars = list(STAGE_DEFINITIONS.keys())
    melted = df.melt(
        id_vars=["benchmark", "jit_approach"],
        value_vars=value_vars,
        var_name="stage_column",
        value_name="time_ms",
    )
    melted = melted.dropna(subset=["time_ms"])
    if melted.empty:
        raise ValueError("All stage timing values are NaN after filtering.")

    melted["stage_label"] = melted["stage_column"].map(
        lambda column: STAGE_DEFINITIONS[column][0]
    )
    melted["stage_label"] = pd.Categorical(
        melted["stage_label"],
        categories=[entry[0] for entry in STAGE_DEFINITIONS.values()],
        ordered=True,
    )

    melted["approach"] = melted["jit_approach"].astype(str)
    melted["approach_key"] = melted["approach"].map(lambda value: value.strip().lower())
    melted["approach_label"] = melted["approach"].map(_format_approach_label)

    benchmarks = unique_preserving_order(melted["benchmark"])
    benchmark_label_map = {
        benchmark: _format_benchmark_label(benchmark) for benchmark in benchmarks
    }
    benchmark_labels = [benchmark_label_map[benchmark] for benchmark in benchmarks]
    melted["benchmark_label"] = melted["benchmark"].map(benchmark_label_map)
    melted["benchmark_label"] = pd.Categorical(
        melted["benchmark_label"], categories=benchmark_labels, ordered=True
    )

    approach_key_order = _order_approach_keys(unique_preserving_order(melted["approach_key"]))
    key_to_label: dict[str, str] = {}
    for key, label in zip(melted["approach_key"], melted["approach_label"]):
        key_to_label.setdefault(key, label)
    approaches = [key_to_label[key] for key in approach_key_order if key in key_to_label]

    melted["approach_label"] = pd.Categorical(
        melted["approach_label"], categories=approaches, ordered=True
    )

    hatch_map = _build_hatch_map(approaches)
    melted["approach_hatch"] = melted["approach_label"].map(hatch_map)

    melted = melted.sort_values(["benchmark_label", "approach_label", "stage_label"]).reset_index(drop=True)

    positions = _compute_positions(melted, benchmark_labels, approaches)
    melted["x_pos"] = positions["x_pos"]
    melted["bar_width"] = positions["bar_width"]
    melted["benchmark_center"] = positions["benchmark_center"]

    return melted


def _format_approach_label(value: str) -> str:
    normalized = value.strip().lower()
    overrides = {
        "cpp": "CPP",
        "dsl": "DSL",
        "proteus": "Proteus",
        "jitify": "Jitify",
        "aot": "AoT",
    }
    if normalized in FRONTEND_LABELS:
        return FRONTEND_LABELS[normalized]
    return overrides.get(normalized, value.upper())


def _order_approach_keys(keys: list[str]) -> list[str]:
    ordered: list[str] = [key for key in FRONTEND_ORDER if key in keys]
    ordered.extend([key for key in keys if key not in FRONTEND_ORDER])
    return ordered


def _format_benchmark_label(name: str) -> str:
    label = BENCHMARK_LABELS.get(name)
    if label is not None:
        return label
    fallback = name.replace("-", " ")
    fallback = fallback.title().replace(" ", "-")
    return fallback


def _compute_positions(
    df: pd.DataFrame, benchmarks: list[str], approaches: list[str]
) -> dict[str, np.ndarray | float]:
    bench_to_index = {name: idx for idx, name in enumerate(benchmarks)}
    num_approaches = len(approaches)
    if num_approaches <= 0:
        raise ValueError("At least one JIT approach is required to plot the chart.")

    if num_approaches == 1:
        bar_width = 0.6
        offsets = np.array([0.0])
    else:
        cluster_width = min(0.9, 0.6 + 0.15 * (num_approaches - 1))
        bar_width = cluster_width / num_approaches * 0.9
        offsets = np.linspace(
            -cluster_width / 2 + bar_width / 2,
            cluster_width / 2 - bar_width / 2,
            num_approaches,
        )

    approach_to_offset = {
        approach: offsets[idx] for idx, approach in enumerate(approaches)
    }

    benchmark_centers = (
        df["benchmark_label"].astype(str).map(bench_to_index).astype(float)
    )
    approach_offsets = df["approach_label"].astype(str).map(approach_to_offset).astype(float)
    x_positions = benchmark_centers + approach_offsets

    return {
        "x_pos": x_positions,
        "bar_width": float(bar_width),
        "benchmark_center": benchmark_centers,
    }


def build_plot(df: pd.DataFrame, figure_size: tuple[float, float]) -> ggplot:
    stage_labels = [entry[0] for entry in STAGE_DEFINITIONS.values()]
    stage_colors = [entry[1] for entry in STAGE_DEFINITIONS.values()]

    plot = (
        ggplot(df, aes("x_pos", "time_ms", fill="stage_label"))
        + geom_col(
            width=df["bar_width"].iloc[0] if not df.empty else 0.6,
        )
        + scale_fill_manual(name="", values=stage_colors, breaks=stage_labels)
        + scale_x_continuous(
            breaks=sorted(df["benchmark_center"].unique()),
            labels=[str(cat) for cat in df["benchmark_label"].cat.categories],
            expand=(0.02, 0.02),
        )
        + labs(x="", y="Time (ms)")
        + theme_seaborn(style="whitegrid")
        + theme(
            figure_size=figure_size,
            axis_text_x=element_text(rotation=45, ha="right", size=AXIS_TEXT_SIZE),
            axis_text_y=element_text(size=AXIS_TEXT_SIZE),
            axis_title_x=element_text(size=AXIS_TITLE_X_SIZE),
            axis_title_y=element_text(size=AXIS_TITLE_Y_SIZE),
            legend_title=element_blank(),
            legend_text=element_text(size=AXIS_TEXT_SIZE),
            axis_ticks_minor_x=element_blank(),
            panel_grid_major_x=element_blank(),
            panel_grid_minor_x=element_blank(),
            panel_grid_major_y=element_blank(),
            legend_direction="horizontal",
            legend_position="top",
        )
        + guides(
            fill=guide_legend(title=None, reverse=False),
        )
    )

    return plot


def _apply_hatching_and_legends(fig, df: pd.DataFrame) -> None:
    if not fig.axes:
        return

    ax = fig.axes[0]
    stage_legend = ax.legend_
    if stage_legend is not None:
        stage_legend.set_title(None)
        stage_legend.set_frame_on(False)

    ordered = df.sort_values(["benchmark_label", "approach_label", "stage_label"]).reset_index(drop=True)
    if ordered.empty:
        return

    grouped = (
        ordered.groupby(["benchmark_label", "approach_label"], observed=True)
        .agg(
            x_pos=("x_pos", "first"),
            bar_width=("bar_width", "first"),
            top=("time_ms", "sum"),
            hatch=("approach_hatch", "first"),
        )
        .reset_index()
    )

    for _, row in grouped.iterrows():
        x_left = float(row["x_pos"]) - float(row["bar_width"]) / 2.0
        rect = Rectangle(
            (x_left, 0.0),
            float(row["bar_width"]),
            float(row["top"]),
            facecolor=(1.0, 1.0, 1.0, 0.0),
            edgecolor="#333333",
            linewidth=0.4,
            hatch=row["hatch"],
        )
        rect.set_zorder(10)
        ax.add_patch(rect)

    approach_categories = [str(cat) for cat in df["approach_label"].cat.categories]
    ordered_approaches = ordered["approach_label"].astype(str)
    approach_hatches: dict[str, str] = {}
    for label in approach_categories:
        mask = ordered_approaches == label
        if mask.any():
            approach_hatches[label] = ordered.loc[mask, "approach_hatch"].iloc[0]

    handles = [
        Patch(
            facecolor="white",
            edgecolor="#333333",
            hatch=approach_hatches[label],
            label=label,
        )
        for label in approach_categories
        if label in approach_hatches
    ]

    if handles:
        approach_legend = ax.legend(
            handles,
            [handle.get_label() for handle in handles],
            title="",
            loc="upper left",
            bbox_to_anchor=(1.12, 0.3),
            frameon=False,
        )
        ax.add_artist(approach_legend)

    if stage_legend is not None:
        ax.add_artist(stage_legend)


def _build_hatch_map(approaches: list[str]) -> dict[str, str]:
    pattern_cycle = HATCH_PATTERNS
    return {
        approach: pattern_cycle[idx % len(pattern_cycle)]
        for idx, approach in enumerate(approaches)
    }


def save_plot(
    plot: ggplot,
    data: pd.DataFrame,
    output_dir: Path,
    basename: str,
    width: float,
    height: float,
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        destination = output_dir / f"{basename}.{extension}"
        fig = plot.draw(show=False)
        _apply_hatching_and_legends(fig, data)
        save_kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(destination, **save_kwargs)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path: Path = args.csv_path.expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"{csv_path} does not exist.")

    df = pd.read_csv(csv_path)
    platform = infer_or_validate_platform(df, args.platform)
    plot_df = prepare_dataframe(df, platform)

    output_dir = args.output_dir or csv_path.parent
    plot = build_plot(plot_df, (args.figure_width, args.figure_height))

    save_plot(
        plot,
        plot_df,
        output_dir,
        args.output_basename,
        args.figure_width,
        args.figure_height,
        args.dpi,
    )


if __name__ == "__main__":
    main()
