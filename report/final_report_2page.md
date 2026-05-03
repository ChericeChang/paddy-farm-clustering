# Graph-Based Analysis of Paddy Farming Practices

## Overview and Project Scope

This project analyzes the UCI Paddy Dataset to discover groups of farms with similar cultivation practices and environmental conditions. The dataset contains 2,789 paddy farm records and 45 variables, including farm size, agriblock, crop variety, soil type, nursery method, seed rate, fertilizer use, pesticide use, rainfall, irrigation, weather, humidity, trash bundles, and paddy yield. The project began with a graph-based spectral clustering approach and was expanded to include PCA dimensionality reduction, similarity graph construction, baseline clustering comparisons, cluster interpretation, anomaly detection, and a web portal for visual exploration.

The main goal was not only to cluster farms, but to understand whether the clusters correspond to meaningful farming profiles. Paddy yield was excluded from the clustering input features and used afterward for interpretation, so the model did not simply group farms by the outcome variable. This allowed yield to serve as an external descriptive measure of the discovered farm groups.

## Methodology and Formulation

The project is grounded in unsupervised learning, dimensionality reduction, and graph-based clustering. First, the raw dataset was cleaned by stripping column-name whitespace and separating numeric and categorical variables. The dataset had no missing values, but median imputation for numeric variables and most-frequent imputation for categorical variables were included to make the pipeline robust. Numeric features were standardized, and categorical features were one-hot encoded. After excluding `Paddy yield(in Kg)`, the original 44 input features became 71 processed features.

Principal Component Analysis (PCA) was used to reduce dimensionality before clustering. The first two components explained 57.60% of the variance, and the first six components explained at least 95% of the variance. Therefore, the first six principal components were used for graph construction, clustering, and anomaly detection.

A similarity graph was then constructed in PCA space. Each farm was represented as a node, and edges connected nearby farms using a k-nearest-neighbor graph. Edge weights were computed using a Gaussian similarity function based on Euclidean distance. Smaller values of `k` produced disconnected graphs; for example, `k=10` produced 147 connected components. The final graph used `k=400`, producing one connected graph with 2,789 nodes and 632,680 edges.

Spectral clustering was applied to the similarity graph and compared against k-means and hierarchical Ward clustering. Each method was tested for `k=2` through `k=10` and evaluated using silhouette score, Calinski-Harabasz score, and Davies-Bouldin score.

## Results and Implementation

The best result among the tested methods was k-means with `k=10`. Although spectral clustering was the original proposed graph-based method, k-means performed slightly better on the internal validation metrics, while spectral clustering remained very close. This comparison strengthened the project by showing that the final model choice was based on empirical evaluation rather than assuming the graph-based method would automatically perform best.

| Method | k | Silhouette | Calinski-Harabasz | Davies-Bouldin |
| --- | ---: | ---: | ---: | ---: |
| k-means | 10 | 0.5999 | 1907.69 | 0.6835 |
| spectral | 10 | 0.5886 | 1901.57 | 0.7365 |
| hierarchical Ward | 10 | 0.5650 | 1802.53 | 0.7555 |

The final k-means clusters showed clear differences in farm scale, input intensity, agriblock, and yield. High-yield clusters generally had larger average farm sizes and larger fertilizer/input quantities. For example, clusters 6, 9, 2, and 7 had average yields around 29,100 to 29,400 kg and average farm sizes around 4.76 hectares. Lower-yield clusters, such as clusters 1, 0, 3, and 8, had average yields around 11,300 to 13,000 kg and average farm sizes around 2 hectares.

| Cluster | Farms | Avg Yield kg | Avg Hectares | Dominant Agriblock |
| ---: | ---: | ---: | ---: | --- |
| 6 | 368 | 29,363.57 | 4.76 | Sankarapuram |
| 9 | 289 | 29,325.22 | 4.76 | Kurinjipadi |
| 2 | 240 | 29,231.16 | 4.76 | Kallakurichi |
| 7 | 265 | 29,145.81 | 4.76 | Cuddalore |
| 4 | 313 | 26,496.97 | 4.35 | Chinnasalem |
| 5 | 425 | 22,328.97 | 3.69 | Panruti |
| 1 | 237 | 13,015.99 | 2.27 | Sankarapuram |
| 0 | 197 | 12,764.93 | 2.22 | Kurinjipadi |
| 3 | 185 | 12,657.78 | 2.20 | Cuddalore |
| 8 | 270 | 11,324.11 | 1.97 | Kallakurichi |

Anomaly detection was also performed using a combined score based on distance from assigned cluster centroid, Isolation Forest score, and low weighted graph degree. The top 5% most anomalous farms were flagged, producing 139 anomalies. Cluster 8 had the most top anomalies, with 38 farms in the top 5%. The strongest anomalies were mostly one-hectare farms with low seed rate, low fertilizer inputs, and low yield. For example, node 2786 was a one-hectare farm in Chinnasalem with yield of 5,723 kg and a combined anomaly score of 0.9818.

## Conclusion and Limitations

The project demonstrates that unsupervised learning can reveal meaningful structure in paddy farming data. PCA reduced the 71-dimensional processed feature space to six principal components while preserving most of the variance. The graph-based spectral method performed well, but k-means slightly outperformed it and was selected as the final model for interpretation. The resulting clusters were interpretable and primarily reflected farm scale, input intensity, and agriblock-level farming patterns.

The main limitation is that the analysis is unsupervised, so there are no true labels for cluster validation. Internal metrics help compare methods, but they do not prove that the selected clusters are agronomically optimal. The dataset also contains duplicate records, which may reflect repeated farm profiles or data duplication. Finally, total yield is strongly associated with farm size and input quantities, so future work should consider yield per hectare or input efficiency to distinguish productivity from scale.

To support presentation and review, a static web portal was built in `app/index.html`. It visualizes the PCA projection, model comparisons, cluster profiles, and anomaly results.

