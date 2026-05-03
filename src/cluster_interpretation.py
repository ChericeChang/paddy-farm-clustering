"""Cluster interpretation utilities for the Paddy Dataset project."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing import DEFAULT_TARGET_COLUMN, load_paddy_dataset


@dataclass(frozen=True)
class ClusterProfileResult:
    """Container for cluster interpretation outputs."""

    profiled_data: pd.DataFrame
    numeric_profile: pd.DataFrame
    categorical_profile: pd.DataFrame
    cluster_summary: pd.DataFrame
    metadata: dict


def project_root_from_file() -> Path:
    """Return the project root assuming this file lives in src/."""

    return Path(__file__).resolve().parents[1]


def load_cluster_labels(path: str | Path, cluster_column: str) -> pd.Series:
    """Load selected cluster labels."""

    labels = pd.read_csv(path)
    if cluster_column not in labels.columns:
        raise ValueError(f"Cluster column not found: {cluster_column}")
    return labels[cluster_column].rename("cluster")


def build_profiled_dataset(
    data_path: str | Path,
    labels_path: str | Path,
    cluster_column: str = "selected_baseline_cluster",
) -> pd.DataFrame:
    """Join raw Paddy data with selected cluster labels."""

    df = load_paddy_dataset(data_path)
    labels = load_cluster_labels(labels_path, cluster_column)
    if len(df) != len(labels):
        raise ValueError("Raw data and labels have different row counts.")
    profiled = df.copy()
    profiled["cluster"] = labels.astype(int)
    return profiled


def summarize_numeric_features(profiled: pd.DataFrame) -> pd.DataFrame:
    """Summarize numeric variables by cluster."""

    numeric_cols = profiled.select_dtypes(include="number").columns.drop("cluster").tolist()
    rows = []
    for cluster, group in profiled.groupby("cluster"):
        for column in numeric_cols:
            rows.append(
                {
                    "cluster": int(cluster),
                    "feature": column,
                    "mean": group[column].mean(),
                    "median": group[column].median(),
                    "std": group[column].std(),
                    "min": group[column].min(),
                    "max": group[column].max(),
                }
            )
    return pd.DataFrame(rows)


def summarize_categorical_features(profiled: pd.DataFrame) -> pd.DataFrame:
    """Find dominant category values within each cluster."""

    categorical_cols = profiled.select_dtypes(exclude="number").columns.tolist()
    rows = []
    for cluster, group in profiled.groupby("cluster"):
        for column in categorical_cols:
            counts = group[column].value_counts(dropna=False)
            top_value = counts.index[0]
            top_count = int(counts.iloc[0])
            rows.append(
                {
                    "cluster": int(cluster),
                    "feature": column,
                    "top_value": top_value,
                    "top_count": top_count,
                    "top_pct": round(top_count / len(group) * 100, 2),
                    "unique_values": int(group[column].nunique(dropna=True)),
                }
            )
    return pd.DataFrame(rows)


def build_cluster_summary(
    profiled: pd.DataFrame,
    categorical_profile: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> pd.DataFrame:
    """Build a compact one-row-per-cluster interpretation table."""

    summary = (
        profiled.groupby("cluster")
        .agg(
            farm_count=("cluster", "size"),
            avg_yield_kg=(target_column, "mean"),
            median_yield_kg=(target_column, "median"),
            avg_hectares=("Hectares", "mean"),
            avg_seedrate_kg=("Seedrate(in Kg)", "mean"),
            avg_dap_20days=("DAP_20days", "mean"),
            avg_urea_40days=("Urea_40Days", "mean"),
            avg_potash_50days=("Potassh_50Days", "mean"),
            avg_pest_60day_ml=("Pest_60Day(in ml)", "mean"),
            avg_trash_bundles=("Trash(in bundles)", "mean"),
        )
        .reset_index()
    )

    for feature in ["Agriblock", "Variety", "Soil Types", "Nursery"]:
        values = categorical_profile[categorical_profile["feature"] == feature][
            ["cluster", "top_value", "top_pct"]
        ].rename(
            columns={
                "top_value": f"dominant_{feature.lower().replace(' ', '_')}",
                "top_pct": f"dominant_{feature.lower().replace(' ', '_')}_pct",
            }
        )
        summary = summary.merge(values, on="cluster", how="left")

    numeric_cols = summary.select_dtypes(include="number").columns
    summary[numeric_cols] = summary[numeric_cols].round(2)
    return summary.sort_values("avg_yield_kg", ascending=False).reset_index(drop=True)


def save_cluster_profile_outputs(
    result: ClusterProfileResult,
    output_dir: str | Path,
    table_dir: str | Path,
    figure_dir: str | Path,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> None:
    """Save cluster interpretation tables and figures."""

    output_path = Path(output_dir)
    table_path = Path(table_dir)
    figure_path = Path(figure_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    table_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)

    result.profiled_data.to_csv(output_path / "cluster_profiled_dataset.csv", index=False)
    result.numeric_profile.to_csv(table_path / "cluster_numeric_profile.csv", index=False)
    result.categorical_profile.to_csv(table_path / "cluster_categorical_profile.csv", index=False)
    result.cluster_summary.to_csv(table_path / "cluster_summary.csv", index=False)
    (table_path / "cluster_interpretation_metadata.json").write_text(
        json.dumps(result.metadata, indent=2),
        encoding="utf-8",
    )

    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(10, 5))
    order = result.cluster_summary.sort_values("cluster")["cluster"]
    sns.boxplot(data=result.profiled_data, x="cluster", y=target_column, order=order, ax=ax)
    ax.set_title("Paddy Yield Distribution by Final K-Means Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Paddy Yield (Kg)")
    fig.tight_layout()
    fig.savefig(figure_path / "cluster_yield_boxplot.png", bbox_inches="tight")
    plt.close(fig)

    heatmap_features = [
        "avg_yield_kg",
        "avg_hectares",
        "avg_seedrate_kg",
        "avg_dap_20days",
        "avg_urea_40days",
        "avg_potash_50days",
        "avg_pest_60day_ml",
        "avg_trash_bundles",
    ]
    heatmap_data = result.cluster_summary.set_index("cluster")[heatmap_features]
    zscore_data = (heatmap_data - heatmap_data.mean()) / heatmap_data.std(ddof=0)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(zscore_data, cmap="vlag", center=0, annot=True, fmt=".1f", ax=ax)
    ax.set_title("Cluster Profiles: Standardized Mean Values")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    fig.savefig(figure_path / "cluster_profile_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_summary = result.cluster_summary.sort_values("avg_yield_kg", ascending=False)
    ax.bar(
        sorted_summary["cluster"].astype(str),
        sorted_summary["avg_yield_kg"],
        color="#2E86AB",
    )
    ax.set_title("Average Paddy Yield by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Average Yield (Kg)")
    fig.tight_layout()
    fig.savefig(figure_path / "cluster_average_yield.png", bbox_inches="tight")
    plt.close(fig)


def run_cluster_interpretation(
    data_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    table_dir: str | Path,
    figure_dir: str | Path,
    cluster_column: str = "selected_baseline_cluster",
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> ClusterProfileResult:
    """Create cluster profiles from raw data and final labels."""

    profiled = build_profiled_dataset(
        data_path=data_path,
        labels_path=labels_path,
        cluster_column=cluster_column,
    )
    numeric_profile = summarize_numeric_features(profiled)
    categorical_profile = summarize_categorical_features(profiled)
    cluster_summary = build_cluster_summary(
        profiled,
        categorical_profile=categorical_profile,
        target_column=target_column,
    )
    metadata = {
        "cluster_column": cluster_column,
        "target_column": target_column,
        "rows_profiled": int(len(profiled)),
        "clusters": sorted(int(cluster) for cluster in profiled["cluster"].unique()),
        "number_of_clusters": int(profiled["cluster"].nunique()),
        "highest_average_yield_cluster": int(cluster_summary.iloc[0]["cluster"]),
        "lowest_average_yield_cluster": int(cluster_summary.iloc[-1]["cluster"]),
    }
    result = ClusterProfileResult(
        profiled_data=profiled,
        numeric_profile=numeric_profile,
        categorical_profile=categorical_profile,
        cluster_summary=cluster_summary,
        metadata=metadata,
    )
    save_cluster_profile_outputs(
        result=result,
        output_dir=output_dir,
        table_dir=table_dir,
        figure_dir=figure_dir,
        target_column=target_column,
    )
    return result


def parse_args() -> argparse.Namespace:
    root = project_root_from_file()
    parser = argparse.ArgumentParser(description="Interpret final Paddy farm clusters.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=root / "data" / "paddydataset.csv",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=root / "outputs" / "processed" / "baseline_cluster_labels.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "processed",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=root / "outputs" / "tables",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=root / "outputs" / "figures",
    )
    parser.add_argument("--cluster-column", default="selected_baseline_cluster")
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_cluster_interpretation(
        data_path=args.data_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        table_dir=args.table_dir,
        figure_dir=args.figure_dir,
        cluster_column=args.cluster_column,
        target_column=args.target_column,
    )
    print("Cluster summary:")
    print(result.cluster_summary.to_string(index=False))
    print("Metadata:")
    print(json.dumps(result.metadata, indent=2))


if __name__ == "__main__":
    main()
