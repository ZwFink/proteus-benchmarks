#!/usr/bin/env python3
"""
Aggregate benchmark results and plot AoT-relative speedups per frontend.
"""

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import use as mpl_use
from plotnine import *

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
        description="Create a bar chart of AoT-relative speedups grouped by benchmark."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing *-results-direct.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store the generated plots (defaults to results_dir).",
    )
    parser.add_argument(
        "--output-basename",
        default="speedup-by-frontend",
        help="Base filename for saved plots (extensions are appended automatically).",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="Optional platform prefix to filter on (for example 'amd').",
    )
    return parser.parse_args()


def discover_result_files(results_dir: Path) -> list[Path]:
    pattern = "*-results-direct.csv"
    return sorted(results_dir.glob(pattern))


def parse_metadata(csv_path: Path) -> tuple[str, str, str]:
    """
    Extract (platform, benchmark, frontend) from a results CSV filename.
    """
    stem = csv_path.stem  # e.g., amd-3mm-aot-results-direct
    if not stem.endswith("-results-direct"):
        raise ValueError(f"Unexpected filename format: {csv_path.name}")
    payload = stem[: -len("-results-direct")]
    for frontend in FRONTEND_LABELS:
        suffix = f"-{frontend}"
        if payload.endswith(suffix):
            platform, benchmark = payload[: -len(suffix)].split("-", 1)
            return platform, benchmark, frontend
    raise ValueError(f"Unrecognized frontend in filename: {csv_path.name}")


def load_results(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def aggregate_execution_times(files: list[Path]) -> pd.DataFrame:
    rows = []
    for csv_path in files:
        platform, benchmark, frontend = parse_metadata(csv_path)

        if frontend not in FRONTEND_LABELS:
            continue

        df = load_results(csv_path)
        if "ExeTime" not in df.columns:
            raise KeyError(f"'ExeTime' column missing from {csv_path}")

        exe_time = df["ExeTime"].mean()
        rows.append(
            {
                "platform": platform,
                "benchmark": benchmark,
                "frontend": frontend,
                "exe_time": exe_time,
            }
        )

    if not rows:
        raise RuntimeError("No matching result files found for the requested frontends.")

    return pd.DataFrame(rows)


def compute_speedups(df: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        df[df["frontend"] == "aot"]
        .set_index(["platform", "benchmark"])["exe_time"]
        .to_dict()
    )

    def speedup(row: pd.Series) -> float:
        base = baseline.get((row["platform"], row["benchmark"]))
        if base is None or pd.isna(base) or base == 0:
            return float("nan")
        return base / row["exe_time"]

    df = df.copy()
    df["speedup"] = df.apply(speedup, axis=1)
    df = df[~df["speedup"].isna()]
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
    df = df[df["frontend"] != "aot"].copy()
    if df.empty:
        raise RuntimeError("No non-AoT frontends present after filtering.")

    benchmarks = sorted(df["benchmark_label"].unique())
    df["benchmark_label"] = pd.Categorical(
        df["benchmark_label"], categories=benchmarks, ordered=True
    )
    df["frontend_label"] = pd.Categorical(
        df["frontend_label"],
        categories=[
            FRONTEND_LABELS[name] for name in FRONTEND_ORDER if name != "aot"
        ],
        ordered=True,
    )

    return (
        ggplot(df, aes("benchmark_label", "speedup", fill="frontend_label"))
        + geom_col(position=position_dodge(width=0.8), width=0.7)
        + geom_hline(yintercept=1, linetype="dashed", color="#333333", size=2)
        + scale_fill_manual(values=FRONTEND_COLORS)
        + labs(x="", y="Speedup", fill="")
        + theme_seaborn(style='whitegrid')
        + theme(
            axis_text_x=element_text(rotation=45, ha="right", size=16),
            figure_size=(10, 5),
            legend_position = "top",
            legend_direction = "horizontal",
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
        raise RuntimeError(f"No *-results-direct.csv files found in {results_dir}.")

    df = aggregate_execution_times(files)

    if args.platform is not None:
        df = df[df["platform"] == args.platform]

    unique_platforms = df["platform"].unique()
    if len(unique_platforms) == 0:
        raise RuntimeError("No results available after applying platform filter.")

    if len(unique_platforms) > 1:
        raise RuntimeError(
            "Multiple platforms detected; use --platform to select one before plotting."
        )

    df = compute_speedups(df)
    if df.empty:
        raise RuntimeError("Speedup calculation produced no data.")

    plot = build_plot(df)

    output_dir = args.output_dir or results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / args.output_basename

    plot.save(f"{base}.png", dpi=300, bbox_inches="tight")
    plot.save(f"{base}.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
