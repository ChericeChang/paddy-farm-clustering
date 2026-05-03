# Paddy Farm Cluster Analysis

This project analyzes the UCI Paddy Dataset using preprocessing, PCA, graph-based similarity analysis, clustering, cluster interpretation, anomaly detection, and a static web portal.

## Project Structure

```text
data/                 Raw Paddy dataset
notebooks/            Step-by-step analysis notebooks
src/                  Reusable Python workflows
outputs/              Generated tables, figures, processed data, and model outputs
app/                  Static web portal
report/               Final report drafts
project proposal/     Proposal and project documentation
```

## Main Workflow

1. Exploratory data analysis
2. Preprocessing and one-hot encoding
3. PCA dimensionality reduction
4. Similarity graph construction
5. Spectral clustering
6. Baseline comparison with k-means and hierarchical clustering
7. Cluster interpretation
8. Anomaly detection
9. Static portal visualization

## Final Modeling Choice

The original project focused on graph-based spectral clustering. After comparing methods, k-means with `k=10` was selected as the final clustering model because it slightly outperformed spectral clustering on internal validation metrics.

| Method | k | Silhouette |
| --- | ---: | ---: |
| k-means | 10 | 0.5999 |
| spectral | 10 | 0.5886 |
| hierarchical Ward | 10 | 0.5650 |

## Web Portal

Open the portal directly in a browser:

```text
app/index.html
```

The portal includes PCA visualizations, clustering model comparison, cluster profiles, and anomaly results.

## Key Outputs

- `report/final_report_2page.md`
- `outputs/tables/cluster_summary.csv`
- `outputs/tables/best_clustering_methods.csv`
- `outputs/tables/top_anomalies.csv`
- `app/index.html`

## Re-running Core Steps

From the project root:

```bash
python3 src/preprocessing.py
python3 src/pca_analysis.py
python3 src/graph_construction.py
python3 src/clustering.py
python3 src/baseline_clustering.py
python3 src/cluster_interpretation.py
python3 src/anomaly_detection.py
```

