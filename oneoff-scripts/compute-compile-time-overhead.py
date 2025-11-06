#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_RESULTS_ROOT = (
    Path(__file__).resolve().parents[1] / "2025-11-04_ctime_experiment" / "results"
)
BASELINE_FRONTEND = "aot"
DEFAULT_JIT_FRONTENDS: Sequence[str] = ("proteus", "dsl", "jitify")
METRIC_COLUMNS = {
    "total": "mean_total_time_s",
    "compile": "mean_compile_time_s",
    "link": "mean_link_time_s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute compile+link slowdowns (relative to AOT) across AMD and NVIDIA "
            "platforms."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing <platform>-compile-times.csv (default: %(default)s).",
    )
    parser.add_argument(
        "--baseline",
        default=BASELINE_FRONTEND,
        help=f"Frontend used as the slowdown baseline (default: {BASELINE_FRONTEND}).",
    )
    parser.add_argument(
        "--jit-frontends",
        nargs="+",
        default=list(DEFAULT_JIT_FRONTENDS),
        help="Frontends treated as JIT for per-benchmark slowdowns.",
    )
    return parser.parse_args()


def load_compile_data(results_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for platform in ("amd", "nvidia"):
        csv_path = results_dir / f"{platform}-compile-times.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        if "platform" not in frame.columns:
            frame["platform"] = frame.get("machine", platform)
        frame["platform"] = frame["platform"].fillna(platform)
        frame["total_time_s"] = frame["compile_time_s"] + frame["link_time_s"]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No compile time CSVs found under {results_dir}. "
            "Expected amd-compile-times.csv and/or nvidia-compile-times.csv."
        )
    return pd.concat(frames, ignore_index=True)


def compute_mean_times(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["platform", "benchmark", "frontend"], as_index=False)
        .agg(
            mean_compile_time_s=("compile_time_s", "mean"),
            mean_link_time_s=("link_time_s", "mean"),
        )
    )
    grouped["mean_total_time_s"] = (
        grouped["mean_compile_time_s"] + grouped["mean_link_time_s"]
    )
    return grouped


def attach_baseline(
    mean_df: pd.DataFrame, baseline_frontend: str
) -> pd.DataFrame:
    baseline = (
        mean_df[mean_df["frontend"] == baseline_frontend]
        .rename(
            columns={
                column: f"{column}_baseline"
                for column in METRIC_COLUMNS.values()
            }
        )
        .drop(columns=["frontend"])
    )
    merged = mean_df.merge(
        baseline,
        on=["platform", "benchmark"],
        how="inner",
        validate="many_to_one",
    )
    for metric_name, metric_col in METRIC_COLUMNS.items():
        baseline_col = f"{metric_col}_baseline"
        slowdown_col = f"{metric_name}_slowdown"
        merged[slowdown_col] = merged[metric_col] / merged[baseline_col].replace(
            0.0, np.nan
        )
    return merged


def geomean(values: Iterable[float]) -> float:
    arr = np.fromiter((v for v in values if v > 0 and not math.isnan(v)), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(arr))))


def print_overall_geomean(slowdown_df: pd.DataFrame) -> None:
    print("Geomean slowdown vs baseline (all platforms):")
    for metric_name in METRIC_COLUMNS:
        slowdown_col = f"{metric_name}_slowdown"
        geos = (
            slowdown_df.groupby("frontend")[slowdown_col]
            .apply(geomean)
            .sort_values()
        )
        print(f"  [{metric_name}]")
        for frontend, value in geos.items():
            print(f"    {frontend}: {value:.3f}×")


def print_platform_geomean(slowdown_df: pd.DataFrame) -> None:
    print("\nGeomean slowdown vs baseline by platform:")
    for platform, platform_df in sorted(slowdown_df.groupby("platform")):
        print(f"  {platform}:")
        for metric_name in METRIC_COLUMNS:
            slowdown_col = f"{metric_name}_slowdown"
            geos = (
                platform_df.groupby("frontend")[slowdown_col]
                .apply(geomean)
                .sort_values()
            )
            print(f"    [{metric_name}]")
            for frontend, value in geos.items():
                print(f"      {frontend}: {value:.3f}×")


def print_per_benchmark_jit(
    slowdown_df: pd.DataFrame, jit_frontends: Sequence[str]
) -> None:
    active_frontends = sorted(
        set(jit_frontends).intersection(slowdown_df["frontend"].unique())
    )
    if not active_frontends:
        print("\nNo JIT frontends present in the dataset.")
        return

    formatter = lambda x: f"{x:.3f}" if pd.notna(x) else "nan"

    print("\nPer-benchmark slowdowns for JIT frontends (vs baseline):")
    for platform, platform_df in sorted(slowdown_df.groupby("platform")):
        subset = platform_df[platform_df["frontend"].isin(active_frontends)]
        if subset.empty:
            continue
        print(f"\n  {platform}:")
        for metric_name in METRIC_COLUMNS:
            slowdown_col = f"{metric_name}_slowdown"
            pivot = (
                subset.pivot_table(
                    index="benchmark",
                    columns="frontend",
                    values=slowdown_col,
                    aggfunc=geomean,
                )
                .reindex(columns=active_frontends)
                .sort_index()
            )
            if pivot.empty:
                continue
            print(f"    [{metric_name}]")
            print(
                pivot.to_string(
                    index=True,
                    formatters={col: formatter for col in pivot.columns},
                )
            )


def main() -> None:
    args = parse_args()
    data = load_compile_data(args.results_dir)
    mean_times = compute_mean_times(data)
    slowdown_df = attach_baseline(mean_times, args.baseline)

    print_overall_geomean(slowdown_df)
    print_platform_geomean(slowdown_df)
    print_per_benchmark_jit(slowdown_df, args.jit_frontends)


if __name__ == "__main__":
    main()
