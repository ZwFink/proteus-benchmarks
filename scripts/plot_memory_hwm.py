#!/usr/bin/env python3
"""
Plot CPU memory overhead relative to AOT baseline from memory HWM profiling results.

Usage:
    python plot_memory_hwm.py --input results.csv --output plot.png
    python plot_memory_hwm.py --input results.csv --output plot  # Saves .png and .pdf
"""

import argparse
from pathlib import Path

import pandas as pd
from plotnine import *

# Frontend labels and colors (matching speedup plots)
FRONTEND_LABELS = {
    "proteus": "PJ-Annot.",
    "dsl": "PJ-DSL",
    "cpp": "PJ-CPP",
}

FRONTEND_COLORS = {
    "PJ-Annot.": "#0077BB",
    "PJ-DSL": "#009988",
    "PJ-CPP": "#33BBEE",
}

FRONTEND_ORDER = ["PJ-Annot.", "PJ-DSL", "PJ-CPP"]

BENCHMARK_LABELS = {
    "3mm": "3mm",
    "adam": "Adam",
    "attention": "Attention",
    "bezier-surface": "Bezier Surface",
    "conv3d": "Conv3D",
    "floyd-warshall": "Floyd Warshall",
    "gemm": "GEMM",
    "minibude": "MiniBUDE",
}

BENCHMARK_ORDER = [
    "3mm", "Adam", "Attention", "Bezier Surface",
    "Conv3D", "Floyd Warshall", "GEMM", "MiniBUDE"
]


def compute_overhead(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CPU and GPU memory overhead in KiB relative to AOT baseline."""
    # Get AOT baselines for each benchmark
    aot_df = df[df["frontend"] == "aot"].set_index("benchmark")
    cpu_baseline = aot_df["cpu_hwm_bytes"]
    gpu_baseline = aot_df["gpu_hwm_bytes"]

    # Filter to JIT frontends only
    jit_df = df[df["frontend"].isin(["proteus", "dsl", "cpp"])].copy()

    # Compute overhead in KiB
    jit_df["cpu_overhead_kib"] = (
        (jit_df["cpu_hwm_bytes"] - jit_df["benchmark"].map(cpu_baseline)) / 1024
    )
    jit_df["gpu_overhead_kib"] = (
        (jit_df["gpu_hwm_bytes"] - jit_df["benchmark"].map(gpu_baseline)) / 1024
    )

    # Apply frontend labels
    jit_df["frontend_label"] = jit_df["frontend"].map(FRONTEND_LABELS)
    jit_df["frontend_label"] = pd.Categorical(
        jit_df["frontend_label"], categories=FRONTEND_ORDER, ordered=True
    )

    # Apply benchmark labels
    jit_df["benchmark_label"] = jit_df["benchmark"].map(BENCHMARK_LABELS)
    jit_df["benchmark_label"] = pd.Categorical(
        jit_df["benchmark_label"], categories=BENCHMARK_ORDER, ordered=True
    )

    return jit_df


def plot_overhead(df: pd.DataFrame) -> ggplot:
    """Create lollipop chart with CPU and GPU as separate lollipops."""
    df = df.copy()

    # Add 1 to all values so 0 overhead maps to 1 (plottable on log scale)
    df["cpu_plot"] = df["cpu_overhead_kib"] + 1
    df["gpu_plot"] = df["gpu_overhead_kib"] + 1

    # Baseline at 1 (represents 0 overhead)
    y_base = 1

    # Create dummy data for shape legend
    legend_df = pd.DataFrame({
        "benchmark_label": [df["benchmark_label"].iloc[0]] * 2,
        "frontend_label": [df["frontend_label"].iloc[0]] * 2,
        "y": [float("nan")] * 2,
        "metric": ["CPU", "GPU"]
    })

    plot = (
        ggplot(df, aes(x="benchmark_label", color="frontend_label"))
        # CPU lollipop stem
        + geom_segment(
            aes(xend="benchmark_label", y=y_base, yend="cpu_plot"),
            position=position_dodge(width=0.6),
            size=1.5
        )
        # GPU lollipop stem
        + geom_segment(
            aes(xend="benchmark_label", y=y_base, yend="gpu_plot"),
            position=position_dodge(width=0.6),
            size=1.5
        )
        # CPU points (circles)
        + geom_point(
            aes(y="cpu_plot"),
            position=position_dodge(width=0.6),
            size=4,
            shape="o"
        )
        # GPU points (squares)
        + geom_point(
            aes(y="gpu_plot"),
            position=position_dodge(width=0.6),
            size=4,
            shape="s"
        )
        # Invisible points for shape legend
        + geom_point(
            data=legend_df,
            mapping=aes(y="y", shape="metric"),
            size=4
        )
        + scale_y_continuous(trans="log2")
        + scale_color_manual(values=FRONTEND_COLORS)
        + scale_shape_manual(values={"CPU": "o", "GPU": "s"})
        + guides(color=guide_legend(override_aes={"shape": "s", "size": 6}))
        + labs(
            x="Benchmark",
            y="Memory Overhead vs AOT (KiB + 1)",
            color="Frontend",
            shape="Metric",
        )
        + theme(
            panel_background=element_rect(fill="white"),
            panel_grid_major_y=element_line(color="white", size=0.9),
            panel_grid_major_x=element_blank(),
            panel_grid_minor=element_blank(),
            axis_text_x=element_text(rotation=45, hjust=1, size=11),
            axis_text_y=element_text(size=11),
            axis_title=element_text(size=13),
            legend_title=element_text(size=12),
            legend_text=element_text(size=11),
            figure_size=(10, 6),
        )
    )
    return plot


def main():
    parser = argparse.ArgumentParser(
        description="Plot CPU memory overhead from HWM profiling results"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input CSV file from profile_memory_hwm.py"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output file path (without extension for both PNG and PDF)"
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Compute overhead
    overhead_df = compute_overhead(df)

    # Create plot
    plot = plot_overhead(overhead_df)

    # Save outputs
    output_stem = args.output.with_suffix("")
    plot.save(f"{output_stem}.png", dpi=300, width=10, height=6)
    plot.save(f"{output_stem}.pdf", width=10, height=6)

    print(f"Saved: {output_stem}.png")
    print(f"Saved: {output_stem}.pdf")


if __name__ == "__main__":
    main()
