"""Anomaly detection for Paddy farm cluster analysis."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing import DEFAULT_TARGET_COLUMN, load_paddy_dataset


@dataclass(frozen=True)
class AnomalyResult:
    """Container for anomaly scores, top anomalies, and metadata."""

    anomaly_scores: pd.DataFrame
    top_anomalies: pd.DataFrame
    cluster_anomaly_summary: pd.DataFrame
    metadata: dict


def project_root_from_file() -> Path:
    """Return the project root assuming this file lives in src/."""

    return Path(__file__).resolve().parents[1]


def min_max_score(series: pd.Series) -> pd.Series:
    """Scale a score to 0-1, returning zeros when the range is flat."""

    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return pd.Series(0.0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def load_model_inputs(
    coordinates_path: str | Path,
    labels_path: str | Path,
    graph_degree_path: str | Path,
    n_components: int = 6,
    cluster_column: str = "selected_baseline_cluster",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load PCA coordinates, final cluster labels, and graph degree table."""

    coordinates = pd.read_csv(coordinates_path)
    component_columns = [f"PC{i + 1}" for i in range(n_components)]
    coordinates = coordinates[component_columns]

    labels = pd.read_csv(labels_path)
    if cluster_column not in labels.columns:
        raise ValueError(f"Cluster column not found: {cluster_column}")
    clusters = labels[cluster_column].astype(int).rename("cluster")

    degrees = pd.read_csv(graph_degree_path)
    return coordinates, clusters, degrees


def compute_cluster_distance_scores(
    coordinates: pd.DataFrame,
    clusters: pd.Series,
) -> pd.DataFrame:
    """Compute each farm's distance from its assigned cluster centroid."""

    distances = pd.Series(index=coordinates.index, dtype=float)
    for cluster, group_index in clusters.groupby(clusters).groups.items():
        cluster_coordinates = coordinates.loc[group_index]
        centroid = cluster_coordinates.mean(axis=0).to_frame().T
        cluster_distances = pairwise_distances(cluster_coordinates, centroid).ravel()
        distances.loc[group_index] = cluster_distances

    distance_score = pd.DataFrame(
        {
            "node": coordinates.index,
            "cluster": clusters.values,
            "cluster_centroid_distance": distances.values,
        }
    )
    distance_score["cluster_distance_score"] = distance_score.groupby("cluster")[
        "cluster_centroid_distance"
    ].transform(min_max_score)
    return distance_score


def compute_isolation_forest_scores(
    coordinates: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute Isolation Forest anomaly scores in PCA space."""

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(coordinates)
    raw_score = -model.score_samples(coordinates)
    predicted_outlier = (model.predict(coordinates) == -1).astype(int)
    return pd.DataFrame(
        {
            "node": coordinates.index,
            "isolation_forest_raw_score": raw_score,
            "isolation_forest_score": min_max_score(pd.Series(raw_score)).values,
            "isolation_forest_outlier": predicted_outlier,
        }
    )


def compute_graph_degree_scores(degrees: pd.DataFrame) -> pd.DataFrame:
    """Treat low graph connectivity as an anomaly signal."""

    degree_scores = degrees[["node", "degree", "weighted_degree"]].copy()
    degree_scores["low_degree_score"] = 1 - min_max_score(degree_scores["degree"])
    degree_scores["low_weighted_degree_score"] = 1 - min_max_score(
        degree_scores["weighted_degree"]
    )
    return degree_scores


def build_anomaly_scores(
    coordinates: pd.DataFrame,
    clusters: pd.Series,
    degrees: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Combine distance, Isolation Forest, and graph connectivity anomaly signals."""

    distance_scores = compute_cluster_distance_scores(coordinates, clusters)
    isolation_scores = compute_isolation_forest_scores(
        coordinates,
        contamination=contamination,
        random_state=random_state,
    )
    degree_scores = compute_graph_degree_scores(degrees)

    scores = (
        distance_scores.merge(isolation_scores, on="node", how="left")
        .merge(degree_scores, on="node", how="left")
    )
    scores["combined_anomaly_score"] = (
        0.45 * scores["cluster_distance_score"]
        + 0.45 * scores["isolation_forest_score"]
        + 0.10 * scores["low_weighted_degree_score"]
    )
    scores["anomaly_rank"] = scores["combined_anomaly_score"].rank(
        method="first",
        ascending=False,
    ).astype(int)
    scores["is_top_5pct_anomaly"] = (
        scores["anomaly_rank"] <= max(1, int(round(0.05 * len(scores))))
    ).astype(int)
    return scores.sort_values("anomaly_rank")


def attach_raw_context(
    scores: pd.DataFrame,
    data_path: str | Path,
    target_column: str = DEFAULT_TARGET_COLUMN,
    top_n: int = 50,
) -> pd.DataFrame:
    """Attach raw farm context to the highest-ranked anomalies."""

    df = load_paddy_dataset(data_path)
    context_columns = [
        "Hectares",
        "Agriblock",
        "Variety",
        "Soil Types",
        "Seedrate(in Kg)",
        "Nursery",
        "DAP_20days",
        "Urea_40Days",
        "Potassh_50Days",
        "Pest_60Day(in ml)",
        "Trash(in bundles)",
        target_column,
    ]
    context_columns = [column for column in context_columns if column in df.columns]
    context = df[context_columns].copy()
    context.insert(0, "node", df.index)
    return scores.head(top_n).merge(context, on="node", how="left")


def summarize_cluster_anomalies(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize anomaly signal by final cluster."""

    return (
        scores.groupby("cluster")
        .agg(
            farm_count=("node", "size"),
            top_5pct_anomaly_count=("is_top_5pct_anomaly", "sum"),
            avg_combined_anomaly_score=("combined_anomaly_score", "mean"),
            max_combined_anomaly_score=("combined_anomaly_score", "max"),
            avg_cluster_centroid_distance=("cluster_centroid_distance", "mean"),
            isolation_forest_outlier_count=("isolation_forest_outlier", "sum"),
            avg_degree=("degree", "mean"),
            avg_weighted_degree=("weighted_degree", "mean"),
        )
        .reset_index()
        .round(4)
    )


def save_anomaly_outputs(
    result: AnomalyResult,
    output_dir: str | Path,
    table_dir: str | Path,
    figure_dir: str | Path,
) -> None:
    """Save anomaly tables and figures."""

    output_path = Path(output_dir)
    table_path = Path(table_dir)
    figure_path = Path(figure_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    table_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)

    result.anomaly_scores.to_csv(output_path / "anomaly_scores.csv", index=False)
    result.top_anomalies.to_csv(table_path / "top_anomalies.csv", index=False)
    result.cluster_anomaly_summary.to_csv(
        table_path / "cluster_anomaly_summary.csv",
        index=False,
    )
    (table_path / "anomaly_detection_metadata.json").write_text(
        json.dumps(result.metadata, indent=2),
        encoding="utf-8",
    )

    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(result.anomaly_scores["combined_anomaly_score"], bins=35, ax=ax, color="#2E86AB")
    ax.set_title("Combined Anomaly Score Distribution")
    ax.set_xlabel("Combined Anomaly Score")
    ax.set_ylabel("Number of Farms")
    fig.tight_layout()
    fig.savefig(figure_path / "anomaly_score_distribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=result.anomaly_scores, x="cluster", y="combined_anomaly_score", ax=ax)
    ax.set_title("Anomaly Scores by Final Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Combined Anomaly Score")
    fig.tight_layout()
    fig.savefig(figure_path / "anomaly_scores_by_cluster.png", bbox_inches="tight")
    plt.close(fig)


def run_anomaly_detection(
    data_path: str | Path,
    coordinates_path: str | Path,
    labels_path: str | Path,
    graph_degree_path: str | Path,
    output_dir: str | Path,
    table_dir: str | Path,
    figure_dir: str | Path,
    n_components: int = 6,
    cluster_column: str = "selected_baseline_cluster",
    contamination: float = 0.05,
    top_n: int = 50,
    random_state: int = 42,
) -> AnomalyResult:
    """Run the full anomaly detection workflow."""

    coordinates, clusters, degrees = load_model_inputs(
        coordinates_path=coordinates_path,
        labels_path=labels_path,
        graph_degree_path=graph_degree_path,
        n_components=n_components,
        cluster_column=cluster_column,
    )
    scores = build_anomaly_scores(
        coordinates=coordinates,
        clusters=clusters,
        degrees=degrees,
        contamination=contamination,
        random_state=random_state,
    )
    top_anomalies = attach_raw_context(
        scores=scores,
        data_path=data_path,
        top_n=top_n,
    )
    cluster_summary = summarize_cluster_anomalies(scores)
    metadata = {
        "rows_scored": int(len(scores)),
        "pca_components_used": int(n_components),
        "cluster_column": cluster_column,
        "contamination": contamination,
        "top_n_saved": int(top_n),
        "top_5pct_anomaly_count": int(scores["is_top_5pct_anomaly"].sum()),
        "highest_score_node": int(scores.iloc[0]["node"]),
        "highest_score_cluster": int(scores.iloc[0]["cluster"]),
        "highest_combined_anomaly_score": float(scores.iloc[0]["combined_anomaly_score"]),
        "score_weights": {
            "cluster_distance_score": 0.45,
            "isolation_forest_score": 0.45,
            "low_weighted_degree_score": 0.10,
        },
    }
    result = AnomalyResult(
        anomaly_scores=scores,
        top_anomalies=top_anomalies,
        cluster_anomaly_summary=cluster_summary,
        metadata=metadata,
    )
    save_anomaly_outputs(
        result=result,
        output_dir=output_dir,
        table_dir=table_dir,
        figure_dir=figure_dir,
    )
    return result


def parse_args() -> argparse.Namespace:
    root = project_root_from_file()
    parser = argparse.ArgumentParser(description="Detect anomalous Paddy farms.")
    parser.add_argument("--data-path", type=Path, default=root / "data" / "paddydataset.csv")
    parser.add_argument(
        "--coordinates-path",
        type=Path,
        default=root / "outputs" / "processed" / "pca_coordinates_all_components.csv",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=root / "outputs" / "processed" / "baseline_cluster_labels.csv",
    )
    parser.add_argument(
        "--graph-degree-path",
        type=Path,
        default=root / "outputs" / "tables" / "similarity_graph_degrees.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "processed")
    parser.add_argument("--table-dir", type=Path, default=root / "outputs" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=root / "outputs" / "figures")
    parser.add_argument("--n-components", type=int, default=6)
    parser.add_argument("--cluster-column", default="selected_baseline_cluster")
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_anomaly_detection(
        data_path=args.data_path,
        coordinates_path=args.coordinates_path,
        labels_path=args.labels_path,
        graph_degree_path=args.graph_degree_path,
        output_dir=args.output_dir,
        table_dir=args.table_dir,
        figure_dir=args.figure_dir,
        n_components=args.n_components,
        cluster_column=args.cluster_column,
        contamination=args.contamination,
        top_n=args.top_n,
        random_state=args.random_state,
    )
    print("Top anomalies:")
    print(result.top_anomalies.head(15).to_string(index=False))
    print("Metadata:")
    print(json.dumps(result.metadata, indent=2))


if __name__ == "__main__":
    main()
