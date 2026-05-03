"""Spectral clustering workflow for the Paddy Dataset project."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


@dataclass(frozen=True)
class ClusteringResult:
    """Container for spectral clustering labels and evaluation metrics."""

    labels: pd.DataFrame
    metrics: pd.DataFrame
    selected_k: int
    metadata: dict


def project_root_from_file() -> Path:
    """Return the project root assuming this file lives in src/."""

    return Path(__file__).resolve().parents[1]


def load_pca_coordinates(path: str | Path, n_components: int = 6) -> pd.DataFrame:
    """Load the leading PCA coordinates used for clustering."""

    coordinates = pd.read_csv(path)
    component_columns = [f"PC{i + 1}" for i in range(n_components)]
    missing = [column for column in component_columns if column not in coordinates.columns]
    if missing:
        raise ValueError(f"Missing PCA coordinate columns: {missing}")
    return coordinates[component_columns]


def load_similarity_adjacency(edge_path: str | Path, n_nodes: int) -> csr_matrix:
    """Reconstruct a symmetric sparse adjacency matrix from the saved edge list."""

    edges = pd.read_csv(edge_path)
    row = np.concatenate([edges["source"].to_numpy(), edges["target"].to_numpy()])
    col = np.concatenate([edges["target"].to_numpy(), edges["source"].to_numpy()])
    data = np.concatenate([edges["similarity"].to_numpy(), edges["similarity"].to_numpy()])
    adjacency = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    adjacency.setdiag(1.0)
    return adjacency


def evaluate_labels(features: pd.DataFrame, labels: np.ndarray) -> dict:
    """Compute internal cluster validation metrics."""

    return {
        "silhouette_score": float(silhouette_score(features, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(features, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(features, labels)),
    }


def run_spectral_clustering_grid(
    coordinates: pd.DataFrame,
    adjacency: csr_matrix,
    k_values: range | list[int],
    random_state: int = 42,
) -> ClusteringResult:
    """Run spectral clustering for candidate k values and choose by silhouette."""

    metrics_rows = []
    label_columns = {"node": np.arange(len(coordinates))}

    for k in k_values:
        model = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=random_state,
            n_init=20,
        )
        labels = model.fit_predict(adjacency)
        scores = evaluate_labels(coordinates, labels)
        counts = pd.Series(labels).value_counts().sort_index()
        label_columns[f"spectral_k{k}"] = labels
        metrics_rows.append(
            {
                "method": "spectral",
                "n_clusters": int(k),
                **scores,
                "smallest_cluster_size": int(counts.min()),
                "largest_cluster_size": int(counts.max()),
            }
        )

    metrics = pd.DataFrame(metrics_rows)
    selected_row = metrics.sort_values(
        ["silhouette_score", "calinski_harabasz_score"],
        ascending=[False, False],
    ).iloc[0]
    selected_k = int(selected_row["n_clusters"])
    labels_df = pd.DataFrame(label_columns)
    labels_df["selected_spectral_cluster"] = labels_df[f"spectral_k{selected_k}"]

    selected_counts = labels_df["selected_spectral_cluster"].value_counts().sort_index()
    metadata = {
        "selected_method": "spectral",
        "selected_k": selected_k,
        "selection_metric": "highest silhouette_score",
        "candidate_k_values": [int(k) for k in k_values],
        "rows_clustered": int(len(coordinates)),
        "pca_components_used": int(coordinates.shape[1]),
        "selected_silhouette_score": float(selected_row["silhouette_score"]),
        "selected_calinski_harabasz_score": float(selected_row["calinski_harabasz_score"]),
        "selected_davies_bouldin_score": float(selected_row["davies_bouldin_score"]),
        "selected_cluster_sizes": {
            str(int(cluster)): int(size) for cluster, size in selected_counts.items()
        },
    }

    return ClusteringResult(
        labels=labels_df,
        metrics=metrics,
        selected_k=selected_k,
        metadata=metadata,
    )


def save_clustering_outputs(
    result: ClusteringResult,
    coordinates_2d_path: str | Path,
    output_dir: str | Path,
    table_dir: str | Path,
    figure_dir: str | Path,
) -> None:
    """Save clustering labels, metrics, metadata, and figures."""

    output_path = Path(output_dir)
    table_path = Path(table_dir)
    figure_path = Path(figure_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    table_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)

    result.labels.to_csv(output_path / "spectral_cluster_labels.csv", index=False)
    result.metrics.to_csv(table_path / "spectral_clustering_metrics.csv", index=False)
    (table_path / "spectral_clustering_metadata.json").write_text(
        json.dumps(result.metadata, indent=2),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        result.metrics["n_clusters"],
        result.metrics["silhouette_score"],
        marker="o",
        color="#2E86AB",
        label="Silhouette",
    )
    ax.set_title("Spectral Clustering: Silhouette by Cluster Count")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")
    ax.axvline(result.selected_k, linestyle="--", color="#C8553D", linewidth=1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path / "spectral_silhouette_by_k.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    selected_counts = (
        result.labels["selected_spectral_cluster"].value_counts().sort_index()
    )
    ax.bar(selected_counts.index.astype(str), selected_counts.values, color="#5B8E7D")
    ax.set_title(f"Spectral Cluster Sizes, k={result.selected_k}")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Farms")
    fig.tight_layout()
    fig.savefig(figure_path / "spectral_cluster_sizes.png", bbox_inches="tight")
    plt.close(fig)

    coordinates_2d = pd.read_csv(coordinates_2d_path)
    plot_df = coordinates_2d.join(result.labels["selected_spectral_cluster"])
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        plot_df["PC1"],
        plot_df["PC2"],
        c=plot_df["selected_spectral_cluster"],
        cmap="tab10",
        s=16,
        alpha=0.78,
    )
    ax.set_title(f"Spectral Clusters on PCA Projection, k={result.selected_k}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    legend = ax.legend(
        *scatter.legend_elements(),
        title="Cluster",
        loc="best",
        frameon=True,
    )
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(figure_path / "spectral_clusters_pca_2d.png", bbox_inches="tight")
    plt.close(fig)


def run_spectral_workflow(
    pca_coordinates_path: str | Path,
    graph_edges_path: str | Path,
    pca_2d_path: str | Path,
    output_dir: str | Path,
    table_dir: str | Path,
    figure_dir: str | Path,
    n_components: int = 6,
    min_k: int = 2,
    max_k: int = 10,
    random_state: int = 42,
) -> ClusteringResult:
    """Load inputs, run spectral clustering, and save outputs."""

    coordinates = load_pca_coordinates(pca_coordinates_path, n_components=n_components)
    adjacency = load_similarity_adjacency(graph_edges_path, n_nodes=len(coordinates))
    result = run_spectral_clustering_grid(
        coordinates=coordinates,
        adjacency=adjacency,
        k_values=range(min_k, max_k + 1),
        random_state=random_state,
    )
    save_clustering_outputs(
        result=result,
        coordinates_2d_path=pca_2d_path,
        output_dir=output_dir,
        table_dir=table_dir,
        figure_dir=figure_dir,
    )
    return result


def parse_args() -> argparse.Namespace:
    root = project_root_from_file()
    parser = argparse.ArgumentParser(description="Run spectral clustering on Paddy farms.")
    parser.add_argument(
        "--pca-coordinates-path",
        type=Path,
        default=root / "outputs" / "processed" / "pca_coordinates_all_components.csv",
        help="Path to all PCA coordinates.",
    )
    parser.add_argument(
        "--graph-edges-path",
        type=Path,
        default=root / "outputs" / "processed" / "similarity_graph_edges.csv",
        help="Path to similarity graph edge list.",
    )
    parser.add_argument(
        "--pca-2d-path",
        type=Path,
        default=root / "outputs" / "processed" / "pca_coordinates_2d.csv",
        help="Path to PC1/PC2 coordinates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "processed",
        help="Directory for cluster label outputs.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=root / "outputs" / "tables",
        help="Directory for clustering summary tables.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=root / "outputs" / "figures",
        help="Directory for clustering figures.",
    )
    parser.add_argument("--n-components", type=int, default=6)
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_spectral_workflow(
        pca_coordinates_path=args.pca_coordinates_path,
        graph_edges_path=args.graph_edges_path,
        pca_2d_path=args.pca_2d_path,
        output_dir=args.output_dir,
        table_dir=args.table_dir,
        figure_dir=args.figure_dir,
        n_components=args.n_components,
        min_k=args.min_k,
        max_k=args.max_k,
        random_state=args.random_state,
    )
    print(f"Selected k: {result.selected_k}")
    print(f"Selection metric: {result.metadata['selection_metric']}")
    print("Metrics:")
    print(result.metrics.to_string(index=False))
    print("Selected cluster sizes:")
    for cluster, size in result.metadata["selected_cluster_sizes"].items():
        print(f"  Cluster {cluster}: {size}")


if __name__ == "__main__":
    main()
