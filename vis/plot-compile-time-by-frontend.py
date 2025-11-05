#!/usr/bin/env python3
"""
Aggregate benchmark results and plot compile times per frontend.
"""

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import use as mpl_use
from plotnine import (
    aes,
    element_blank,
    element_text,
    geom_col,
    ggplot,
    labs,
    position_dodge,
    scale_fill_manual,
    theme,
    theme_seaborn,
)

mpl_use("Agg")

FRONTEND_LABELS = {
    "aot": "AoT",
    "proteus": "PJ-Annot.",
    "dsl": "PJ-DSL",
    "cpp": "PJ-CPP",
}

FRONTEND_ORDER = ["aot", "proteus", "dsl", "cpp"]

FRONTEND_COLORS = {
    "PJ-Annot.": "#000000",
    "PJ-CPP": "#FF0066",
    "PJ-DSL": "#107F80",
    "AoT": '#40007F'
}

BENCHMARK_LABELS = {
    "3mm": "3mm",
    "adam": "Adam",
    "attention": "Attention",
    "bezier-surface": "Bezier-Surf.",
    "conv3d": "Conv3D",
    "floyd-warshall": "Floyd-Warsh.",
    "gemm": "GEMM",
    "minibude": "MiniBUDE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a bar chart of compile times grouped by benchmark."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing *-compile-times.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store the generated plots (defaults to results_dir).",
    )
    parser.add_argument(
        "--output-basename",
        default="compile-time-by-frontend",
        help="Base filename for saved plots (extensions are appended automatically).",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="Optional platform prefix to filter on (for example 'amd').",
    )
    return parser.parse_args()


def discover_result_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*-compile-times.csv"))


def load_results(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def aggregate_compile_times(files: list[Path]) -> pd.DataFrame:
    rows = []
    for csv_path in files:
        df = load_results(csv_path)
        required_cols = {
            "machine",
            "benchmark",
            "frontend",
            "compile_time_s",
            "link_time_s",
            "rep",
            "timestamp",
        }
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise KeyError(
                f"{csv_path} missing required columns: {', '.join(sorted(missing))}"
            )

        filtered = df[df["frontend"].isin(FRONTEND_LABELS)].copy()
        if filtered.empty:
            continue

        filtered["timestamp"] = pd.to_datetime(
            filtered["timestamp"], errors="coerce"
        )
        trimmed_groups = []
        for _, group in filtered.groupby(
            ["machine", "benchmark", "frontend"], sort=False
        ):
            group = group.sort_values(
                by=["rep", "timestamp"], kind="stable"
            )
            if len(group) <= 1:
                continue
            trimmed_groups.append(group.iloc[1:])

        if not trimmed_groups:
            continue

        filtered = pd.concat(trimmed_groups, ignore_index=True)

        filtered["total_time_s"] = filtered["compile_time_s"] + filtered["link_time_s"]

        grouped = (
            filtered.groupby(["machine", "benchmark", "frontend"], as_index=False)[
                "total_time_s"
            ].mean()
        )
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "platform": str(row["machine"]),
                    "benchmark": str(row["benchmark"]),
                    "frontend": str(row["frontend"]),
                    "compile_time": float(row["total_time_s"]),
                }
            )

    if not rows:
        raise RuntimeError(
            "No compile time samples remain after filtering; all groups were removed by warmup exclusion or frontend filtering."
        )

    df = pd.DataFrame(rows)
    return df.groupby(
        ["platform", "benchmark", "frontend"], as_index=False
    )["compile_time"].mean()


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["frontend_label"] = df["frontend"].map(FRONTEND_LABELS)
    df = df[df["frontend_label"].notna()].copy()

    fallback_labels = (
        df["benchmark"]
        .str.replace("-", " ", regex=False)
        .str.title()
        .str.replace(" ", "-", regex=False)
    )
    df["benchmark_label"] = df["benchmark"].map(BENCHMARK_LABELS).fillna(fallback_labels)
    return df


def build_plot(df: pd.DataFrame) -> ggplot:
    if df.empty:
        raise RuntimeError("No compile time data available after filtering.")

    benchmarks = sorted(df["benchmark_label"].unique())
    df["benchmark_label"] = pd.Categorical(
        df["benchmark_label"], categories=benchmarks, ordered=True
    )
    df["frontend_label"] = pd.Categorical(
        df["frontend_label"],
        categories=[
            FRONTEND_LABELS[name] for name in FRONTEND_ORDER #if name != "aot"
        ],
        ordered=True,
    )

    return (
        ggplot(df, aes("benchmark_label", "compile_time", fill="frontend_label"))
        + geom_col(position=position_dodge(width=0.8), width=0.7)
        + scale_fill_manual(values=FRONTEND_COLORS)
        + labs(x="", y="Compile Time (s)", fill="")
        + theme_seaborn(style="whitegrid")
        + theme(
            axis_text_x=element_text(rotation=45, ha="right", size=16),
            figure_size=(6.4, 4.8),
            legend_position="top",
            legend_direction="horizontal",
            panel_grid_major_x=element_blank(),
            panel_grid_minor_x=element_blank(),
            panel_grid_major_y=element_blank(),
            axis_title_y=element_text(size=25),
            axis_title_x=element_text(size=18),
            axis_text_y=element_text(size=16),
        )
    )


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    if not results_dir.is_dir():
        raise FileNotFoundError(f"{results_dir} is not a directory.")

    files = discover_result_files(results_dir)
    if not files:
        raise RuntimeError(
            f"No *-compile-times.csv files found in {results_dir}."
        )

    df = aggregate_compile_times(files)

    if args.platform is not None:
        df = df[df["platform"] == args.platform]

    unique_platforms = df["platform"].unique()
    if len(unique_platforms) == 0:
        raise RuntimeError("No results available after applying platform filter.")

    if len(unique_platforms) > 1:
        raise RuntimeError(
            "Multiple platforms detected; use --platform to select one before plotting."
        )

    df = prepare_labels(df)
    if df.empty:
        raise RuntimeError("Compile time aggregation produced no data.")

    plot = build_plot(df)

    output_dir = args.output_dir or results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / args.output_basename

    plot.save(f"{base}.png", dpi=300, bbox_inches="tight")
    plot.save(f"{base}.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
