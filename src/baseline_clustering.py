"""Baseline clustering comparisons for the Paddy Dataset project."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


@dataclass(frozen=True)
class BaselineResult:
    """Container for baseline labels, metrics, and selected model metadata."""

    labels: pd.DataFrame
    metrics: pd.DataFrame
    metadata: dict


def project_root_from_file() -> Path:
    """Return the project root assuming this file lives in src/."""

    return Path(__file__).resolve().parents[1]


def load_pca_coordinates(path: str | Path, n_components: int = 6) -> pd.DataFrame:
    """Load leading PCA coordinates for clustering comparisons."""

    coordinates = pd.read_csv(path)
    component_columns = [f"PC{i + 1}" for i in range(n_components)]
    missing = [column for column in component_columns if column not in coordinates.columns]
    if missing:
        raise ValueError(f"Missing PCA coordinate columns: {missing}")
    return coordinates[component_columns]


def evaluate_labels(features: pd.DataFrame, labels: np.ndarray) -> dict:
    """Compute internal clustering validation metrics."""

    return {
        "silhouette_score": float(silhouette_score(features, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(features, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(features, labels)),
    }


def cluster_size_summary(labels: np.ndarray) -> tuple[int, int]:
    """Return smallest and largest cluster sizes."""

    counts = pd.Series(labels).value_counts()
    return int(counts.min()), int(counts.max())


def run_baseline_grid(
    coordinates: pd.DataFrame,
    k_values: range | list[int],
    random_state: int = 42,
) -> BaselineResult:
    """Run k-means and hierarchical clustering for candidate cluster counts."""

    label_columns = {"node": np.arange(len(coordinates))}
    metrics_rows = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=50, random_state=random_state)
        kmeans_labels = kmeans.fit_predict(coordinates)
        smallest, largest = cluster_size_summary(kmeans_labels)
        metrics_rows.append(
            {
                "method": "kmeans",
                "n_clusters": int(k),
                **evaluate_labels(coordinates, kmeans_labels),
                "smallest_cluster_size": smallest,
                "largest_cluster_size": largest,
            }
        )
        label_columns[f"kmeans_k{k}"] = kmeans_labels

        hierarchical = AgglomerativeClustering(n_clusters=k, linkage="ward")
        hierarchical_labels = hierarchical.fit_predict(coordinates)
        smallest, largest = cluster_size_summary(hierarchical_labels)
        metrics_rows.append(
            {
                "method": "hierarchical_ward",
                "n_clusters": int(k),
                **evaluate_labels(coordinates, hierarchical_labels),
                "smallest_cluster_size": smallest,
                "largest_cluster_size": largest,
            }
        )
        label_columns[f"hierarchical_ward_k{k}"] = hierarchical_labels

    metrics = pd.DataFrame(metrics_rows)
    best_rows = (
        metrics.sort_values(
            ["method", "silhouette_score", "calinski_harabasz_score"],
            ascending=[True, False, False],
        )
        .groupby("method", as_index=False)
        .head(1)
        .sort_values("silhouette_score", ascending=False)
    )
    best_overall = metrics.sort_values(
        ["silhouette_score", "calinski_harabasz_score"],
        ascending=[False, False],
    ).iloc[0]

    labels = pd.DataFrame(label_columns)
    selected_method = str(best_overall["method"])
    selected_k = int(best_overall["n_clusters"])
    selected_column = f"{selected_method}_k{selected_k}"
    labels["selected_baseline_cluster"] = labels[selected_column]

    metadata = {
        "candidate_k_values": [int(k) for k in k_values],
        "rows_clustered": int(len(coordinates)),
        "pca_components_used": int(coordinates.shape[1]),
        "baseline_selection_metric": "highest silhouette_score",
        "selected_baseline_method": selected_method,
        "selected_baseline_k": selected_k,
        "selected_baseline_label_column": selected_column,
        "best_by_method": best_rows.to_dict(orient="records"),
        "selected_baseline_metrics": best_overall.to_dict(),
        "selected_baseline_cluster_sizes": {
            str(int(cluster)): int(size)
            for cluster, size in labels["selected_baseline_cluster"].value_counts().sort_index().items()
        },
    }

    return BaselineResult(labels=labels, metrics=metrics, metadata=metadata)


def save_baseline_outputs(
    result: BaselineResult,
    spectral_metrics_path: str | Path,
    pca_2d_path: str | Path,
    output_dir: str | Path,
    table_dir: str | Path,
    figure_dir: str | Path,
) -> None:
    """Save baseline labels, metrics, comparison tables, and figures."""

    output_path = Path(output_dir)
    table_path = Path(table_dir)
    figure_path = Path(figure_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    table_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)

    result.labels.to_csv(output_path / "baseline_cluster_labels.csv", index=False)
    result.metrics.to_csv(table_path / "baseline_clustering_metrics.csv", index=False)
    (table_path / "baseline_clustering_metadata.json").write_text(
        json.dumps(result.metadata, indent=2),
        encoding="utf-8",
    )

    spectral_metrics = pd.read_csv(spectral_metrics_path)
    comparison = pd.concat([spectral_metrics, result.metrics], ignore_index=True)
    comparison.to_csv(table_path / "clustering_method_comparison.csv", index=False)

    best_comparison = (
        comparison.sort_values(
            ["method", "silhouette_score", "calinski_harabasz_score"],
            ascending=[True, False, False],
        )
        .groupby("method", as_index=False)
        .head(1)
        .sort_values("silhouette_score", ascending=False)
    )
    best_comparison.to_csv(table_path / "best_clustering_methods.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    for method, group in comparison.groupby("method"):
        ax.plot(
            group["n_clusters"],
            group["silhouette_score"],
            marker="o",
            linewidth=1.6,
            label=method,
        )
    ax.set_title("Clustering Method Comparison: Silhouette Score")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path / "clustering_method_silhouette_comparison.png", bbox_inches="tight")
    plt.close(fig)

    coords_2d = pd.read_csv(pca_2d_path)
    plot_df = coords_2d.join(result.labels["selected_baseline_cluster"])
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        plot_df["PC1"],
        plot_df["PC2"],
        c=plot_df["selected_baseline_cluster"],
        cmap="tab10",
        s=16,
        alpha=0.78,
    )
    ax.set_title(
        "Best Baseline Clusters on PCA Projection "
        f"({result.metadata['selected_baseline_method']}, "
        f"k={result.metadata['selected_baseline_k']})"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", frameon=True)
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(figure_path / "best_baseline_clusters_pca_2d.png", bbox_inches="tight")
    plt.close(fig)


def run_baseline_workflow(
    pca_coordinates_path: str | Path,
    pca_2d_path: str | Path,
    spectral_metrics_path: str | Path,
    output_dir: str | Path,
    table_dir: str | Path,
    figure_dir: str | Path,
    n_components: int = 6,
    min_k: int = 2,
    max_k: int = 10,
    random_state: int = 42,
) -> BaselineResult:
    """Run baseline clustering comparison and save outputs."""

    coordinates = load_pca_coordinates(pca_coordinates_path, n_components=n_components)
    result = run_baseline_grid(
        coordinates=coordinates,
        k_values=range(min_k, max_k + 1),
        random_state=random_state,
    )
    save_baseline_outputs(
        result=result,
        spectral_metrics_path=spectral_metrics_path,
        pca_2d_path=pca_2d_path,
        output_dir=output_dir,
        table_dir=table_dir,
        figure_dir=figure_dir,
    )
    return result


def parse_args() -> argparse.Namespace:
    root = project_root_from_file()
    parser = argparse.ArgumentParser(description="Run baseline clustering comparisons.")
    parser.add_argument(
        "--pca-coordinates-path",
        type=Path,
        default=root / "outputs" / "processed" / "pca_coordinates_all_components.csv",
    )
    parser.add_argument(
        "--pca-2d-path",
        type=Path,
        default=root / "outputs" / "processed" / "pca_coordinates_2d.csv",
    )
    parser.add_argument(
        "--spectral-metrics-path",
        type=Path,
        default=root / "outputs" / "tables" / "spectral_clustering_metrics.csv",
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
    parser.add_argument("--n-components", type=int, default=6)
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_baseline_workflow(
        pca_coordinates_path=args.pca_coordinates_path,
        pca_2d_path=args.pca_2d_path,
        spectral_metrics_path=args.spectral_metrics_path,
        output_dir=args.output_dir,
        table_dir=args.table_dir,
        figure_dir=args.figure_dir,
        n_components=args.n_components,
        min_k=args.min_k,
        max_k=args.max_k,
        random_state=args.random_state,
    )
    print("Best baseline:")
    print(json.dumps(result.metadata["selected_baseline_metrics"], indent=2))
    print("Best by method:")
    for row in result.metadata["best_by_method"]:
        print(row)


if __name__ == "__main__":
    main()
