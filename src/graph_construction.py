"""Similarity graph construction for the Paddy Dataset project."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class SimilarityGraphResult:
    """Container for a sparse similarity graph and its summaries."""

    adjacency: csr_matrix
    edge_list: pd.DataFrame
    degree_table: pd.DataFrame
    metadata: dict


def project_root_from_file() -> Path:
    """Return the project root assuming this file lives in src/."""

    return Path(__file__).resolve().parents[1]


def load_pca_coordinates(path: str | Path, n_components: int) -> pd.DataFrame:
    """Load PCA coordinates and keep the requested leading components."""

    coordinates = pd.read_csv(path)
    component_columns = [f"PC{i + 1}" for i in range(n_components)]
    missing_columns = [column for column in component_columns if column not in coordinates.columns]
    if missing_columns:
        raise ValueError(f"Missing PCA coordinate columns: {missing_columns}")
    return coordinates[component_columns]


def estimate_sigma(neighbor_distances: np.ndarray) -> float:
    """Estimate the Gaussian kernel bandwidth from nonzero neighbor distances."""

    nonzero = neighbor_distances[neighbor_distances > 0]
    if len(nonzero) == 0:
        return 1.0
    return float(np.median(nonzero))


def build_similarity_graph(
    coordinates: pd.DataFrame,
    n_neighbors: int = 400,
    sigma: float | None = None,
) -> SimilarityGraphResult:
    """Build a symmetric k-nearest-neighbor Gaussian similarity graph.

    Each farm is a node. Edges connect nearby farms in PCA space, and edge
    weights are Gaussian similarities based on Euclidean distance.
    """

    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")
    if n_neighbors >= len(coordinates):
        raise ValueError("n_neighbors must be smaller than the number of rows.")

    values = coordinates.to_numpy()
    nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nearest_neighbors.fit(values)
    distances, indices = nearest_neighbors.kneighbors(values)

    neighbor_distances = distances[:, 1:]
    neighbor_indices = indices[:, 1:]
    bandwidth = estimate_sigma(neighbor_distances) if sigma is None else float(sigma)
    if bandwidth <= 0:
        raise ValueError("sigma must be positive.")

    row_indices = np.repeat(np.arange(len(coordinates)), n_neighbors)
    col_indices = neighbor_indices.reshape(-1)
    flat_distances = neighbor_distances.reshape(-1)
    weights = np.exp(-(flat_distances**2) / (2 * bandwidth**2))

    directed = csr_matrix(
        (weights, (row_indices, col_indices)),
        shape=(len(coordinates), len(coordinates)),
    )
    adjacency = directed.maximum(directed.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    edge_sources, edge_targets = adjacency.nonzero()
    upper_mask = edge_sources < edge_targets
    sources = edge_sources[upper_mask]
    targets = edge_targets[upper_mask]
    edge_weights = adjacency[sources, targets].A1
    edge_distances = np.linalg.norm(values[sources] - values[targets], axis=1)
    edge_list = pd.DataFrame(
        {
            "source": sources,
            "target": targets,
            "distance": edge_distances,
            "similarity": edge_weights,
        }
    ).sort_values(["source", "target"], ignore_index=True)

    weighted_degree = np.asarray(adjacency.sum(axis=1)).ravel()
    unweighted_degree = np.diff(adjacency.indptr)
    degree_table = pd.DataFrame(
        {
            "node": np.arange(len(coordinates)),
            "degree": unweighted_degree,
            "weighted_degree": weighted_degree,
        }
    )

    n_components_graph, labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    component_sizes = pd.Series(labels).value_counts().sort_values(ascending=False)
    metadata = {
        "nodes": int(adjacency.shape[0]),
        "edges": int(edge_list.shape[0]),
        "requested_neighbors": int(n_neighbors),
        "pca_components_used": int(coordinates.shape[1]),
        "sigma": bandwidth,
        "connected_components": int(n_components_graph),
        "largest_component_size": int(component_sizes.iloc[0]),
        "largest_component_pct": float(component_sizes.iloc[0] / len(coordinates)),
        "min_degree": int(degree_table["degree"].min()),
        "median_degree": float(degree_table["degree"].median()),
        "mean_degree": float(degree_table["degree"].mean()),
        "max_degree": int(degree_table["degree"].max()),
        "min_similarity": float(edge_list["similarity"].min()),
        "median_similarity": float(edge_list["similarity"].median()),
        "mean_similarity": float(edge_list["similarity"].mean()),
        "max_similarity": float(edge_list["similarity"].max()),
    }

    return SimilarityGraphResult(
        adjacency=adjacency,
        edge_list=edge_list,
        degree_table=degree_table,
        metadata=metadata,
    )


def save_similarity_graph_outputs(
    result: SimilarityGraphResult,
    output_dir: str | Path,
    table_dir: str | Path,
    figure_dir: str | Path,
) -> None:
    """Save graph edge list, node degree table, metadata, and diagnostic plots."""

    output_path = Path(output_dir)
    table_path = Path(table_dir)
    figure_path = Path(figure_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    table_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)

    result.edge_list.to_csv(output_path / "similarity_graph_edges.csv", index=False)
    result.degree_table.to_csv(table_path / "similarity_graph_degrees.csv", index=False)
    (table_path / "similarity_graph_metadata.json").write_text(
        json.dumps(result.metadata, indent=2),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(result.degree_table["degree"], bins=30, color="#2E86AB", edgecolor="white")
    ax.set_title("Similarity Graph Degree Distribution")
    ax.set_xlabel("Node Degree")
    ax.set_ylabel("Number of Farms")
    fig.tight_layout()
    fig.savefig(figure_path / "similarity_graph_degree_distribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(result.edge_list["similarity"], bins=30, color="#5B8E7D", edgecolor="white")
    ax.set_title("Similarity Graph Edge Weight Distribution")
    ax.set_xlabel("Gaussian Similarity")
    ax.set_ylabel("Number of Edges")
    fig.tight_layout()
    fig.savefig(figure_path / "similarity_graph_weight_distribution.png", bbox_inches="tight")
    plt.close(fig)


def save_pca_graph_overlay(
    coordinates_2d_path: str | Path,
    edge_list: pd.DataFrame,
    figure_dir: str | Path,
    max_edges: int = 2500,
) -> None:
    """Save a PC1/PC2 plot with a sample of similarity graph edges."""

    coordinates_2d = pd.read_csv(coordinates_2d_path)
    figure_path = Path(figure_dir)
    figure_path.mkdir(parents=True, exist_ok=True)

    sampled_edges = edge_list.nlargest(max_edges, "similarity")
    fig, ax = plt.subplots(figsize=(9, 7))
    for edge in sampled_edges.itertuples(index=False):
        source = int(edge.source)
        target = int(edge.target)
        ax.plot(
            [coordinates_2d.loc[source, "PC1"], coordinates_2d.loc[target, "PC1"]],
            [coordinates_2d.loc[source, "PC2"], coordinates_2d.loc[target, "PC2"]],
            color="#9AA0A6",
            alpha=0.12,
            linewidth=0.6,
            zorder=1,
        )
    ax.scatter(
        coordinates_2d["PC1"],
        coordinates_2d["PC2"],
        s=12,
        alpha=0.75,
        color="#2E86AB",
        zorder=2,
    )
    ax.set_title("Similarity Graph Overlay on PCA Projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(figure_path / "similarity_graph_pca_overlay.png", bbox_inches="tight")
    plt.close(fig)


def run_graph_construction(
    pca_coordinates_path: str | Path,
    pca_2d_path: str | Path,
    output_dir: str | Path,
    table_dir: str | Path,
    figure_dir: str | Path,
    n_components: int = 6,
    n_neighbors: int = 400,
    sigma: float | None = None,
) -> SimilarityGraphResult:
    """Load PCA coordinates, build the graph, and save outputs."""

    coordinates = load_pca_coordinates(pca_coordinates_path, n_components=n_components)
    result = build_similarity_graph(
        coordinates,
        n_neighbors=n_neighbors,
        sigma=sigma,
    )
    save_similarity_graph_outputs(
        result,
        output_dir=output_dir,
        table_dir=table_dir,
        figure_dir=figure_dir,
    )
    save_pca_graph_overlay(
        coordinates_2d_path=pca_2d_path,
        edge_list=result.edge_list,
        figure_dir=figure_dir,
    )
    return result


def parse_args() -> argparse.Namespace:
    root = project_root_from_file()
    parser = argparse.ArgumentParser(description="Build a Paddy farm similarity graph.")
    parser.add_argument(
        "--pca-coordinates-path",
        type=Path,
        default=root / "outputs" / "processed" / "pca_coordinates_all_components.csv",
        help="Path to all PCA coordinates.",
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
        help="Directory for graph edge outputs.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=root / "outputs" / "tables",
        help="Directory for graph summary tables.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=root / "outputs" / "figures",
        help="Directory for graph figures.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=6,
        help="Number of leading PCA components to use.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=400,
        help="Number of nearest neighbors per farm before graph symmetrization.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Optional Gaussian bandwidth. Defaults to median nearest-neighbor distance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_graph_construction(
        pca_coordinates_path=args.pca_coordinates_path,
        pca_2d_path=args.pca_2d_path,
        output_dir=args.output_dir,
        table_dir=args.table_dir,
        figure_dir=args.figure_dir,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        sigma=args.sigma,
    )
    print(f"Nodes: {result.metadata['nodes']:,}")
    print(f"Edges: {result.metadata['edges']:,}")
    print(f"PCA components used: {result.metadata['pca_components_used']}")
    print(f"Requested neighbors: {result.metadata['requested_neighbors']}")
    print(f"Sigma: {result.metadata['sigma']:.4f}")
    print(f"Connected components: {result.metadata['connected_components']}")
    print(f"Largest component: {result.metadata['largest_component_size']:,} nodes")
    print(
        "Degree min/median/mean/max: "
        f"{result.metadata['min_degree']} / "
        f"{result.metadata['median_degree']:.1f} / "
        f"{result.metadata['mean_degree']:.2f} / "
        f"{result.metadata['max_degree']}"
    )


if __name__ == "__main__":
    main()
