const DATA = window.PADDY_DATA;

const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const pct = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 });
const clusterColors = [
  "#2e6f95",
  "#b84a43",
  "#5b8e7d",
  "#d49a2a",
  "#6d5cae",
  "#287c7c",
  "#9b5d2e",
  "#4777b2",
  "#8a7a29",
  "#a54671",
];

function $(selector) {
  return document.querySelector(selector);
}

function metric(value, label) {
  return `<div class="stat"><span class="stat-value">${value}</span><span class="stat-label">${label}</span></div>`;
}

function num(value, digits = 2) {
  return Number.isFinite(Number(value)) ? fmt.format(Number(value).toFixed(digits)) : value;
}

function table(el, rows, columns) {
  if (!rows.length) {
    el.innerHTML = "";
    return;
  }
  el.innerHTML = `
    <thead><tr>${columns.map((c) => `<th>${c.label}</th>`).join("")}</tr></thead>
    <tbody>
      ${rows
        .map(
          (row) =>
            `<tr>${columns
              .map((c) => `<td>${c.format ? c.format(row[c.key], row) : row[c.key]}</td>`)
              .join("")}</tr>`,
        )
        .join("")}
    </tbody>
  `;
}

function extent(values) {
  return [Math.min(...values), Math.max(...values)];
}

function scale(value, domain, range) {
  if (domain[1] === domain[0]) return (range[0] + range[1]) / 2;
  return range[0] + ((value - domain[0]) / (domain[1] - domain[0])) * (range[1] - range[0]);
}

function colorRamp(value, domain) {
  const t = Math.max(0, Math.min(1, (value - domain[0]) / (domain[1] - domain[0] || 1)));
  const hue = 210 - t * 165;
  const light = 32 + t * 24;
  return `hsl(${hue}, 58%, ${light}%)`;
}

function svgFrame(width = 820, height = 420) {
  return { width, height, left: 58, right: 22, top: 22, bottom: 46 };
}

function drawScatter(mode = "cluster") {
  const el = $("#pca-scatter");
  const points = DATA.pcaPoints;
  const frame = svgFrame(860, 520);
  const xDomain = extent(points.map((d) => d.PC1));
  const yDomain = extent(points.map((d) => d.PC2));
  const yieldDomain = extent(points.map((d) => d.yield_kg));
  const anomalyDomain = extent(points.map((d) => d.combined_anomaly_score));
  const innerW = frame.width - frame.left - frame.right;
  const innerH = frame.height - frame.top - frame.bottom;

  const circles = points
    .map((d) => {
      const cx = scale(d.PC1, xDomain, [frame.left, frame.left + innerW]);
      const cy = scale(d.PC2, yDomain, [frame.top + innerH, frame.top]);
      const fill =
        mode === "cluster"
          ? clusterColors[d.cluster % clusterColors.length]
          : mode === "yield"
            ? colorRamp(d.yield_kg, yieldDomain)
            : colorRamp(d.combined_anomaly_score, anomalyDomain);
      return `<circle cx="${cx}" cy="${cy}" r="3.2" fill="${fill}" opacity="0.72"><title>Cluster ${d.cluster}, yield ${num(d.yield_kg, 0)}, anomaly ${num(d.combined_anomaly_score, 3)}</title></circle>`;
    })
    .join("");

  el.innerHTML = `
    <svg viewBox="0 0 ${frame.width} ${frame.height}" role="img" aria-label="PCA scatterplot">
      <line class="grid-line" x1="${frame.left}" y1="${frame.top + innerH}" x2="${frame.left + innerW}" y2="${frame.top + innerH}" />
      <line class="grid-line" x1="${frame.left}" y1="${frame.top}" x2="${frame.left}" y2="${frame.top + innerH}" />
      ${circles}
      <text class="chart-label" x="${frame.left + innerW / 2}" y="${frame.height - 8}" text-anchor="middle">PC1</text>
      <text class="chart-label" transform="translate(16 ${frame.top + innerH / 2}) rotate(-90)" text-anchor="middle">PC2</text>
    </svg>
  `;
}

function drawBars(el, rows, key, labelKey, options = {}) {
  const frame = svgFrame(760, options.height || 330);
  const sortedRows = options.sort ? [...rows].sort((a, b) => b[key] - a[key]) : rows;
  const maxValue = Math.max(...sortedRows.map((d) => Number(d[key])));
  const barGap = 8;
  const innerW = frame.width - frame.left - frame.right;
  const innerH = frame.height - frame.top - frame.bottom;
  const barW = innerW / sortedRows.length - barGap;
  const bars = sortedRows
    .map((d, i) => {
      const h = scale(Number(d[key]), [0, maxValue], [0, innerH]);
      const x = frame.left + i * (barW + barGap);
      const y = frame.top + innerH - h;
      return `
        <rect x="${x}" y="${y}" width="${barW}" height="${h}" rx="3" fill="${clusterColors[d.cluster % clusterColors.length] || "#2e6f95"}"></rect>
        <text class="chart-label" x="${x + barW / 2}" y="${frame.top + innerH + 20}" text-anchor="middle">${d[labelKey]}</text>
      `;
    })
    .join("");
  el.innerHTML = `
    <svg viewBox="0 0 ${frame.width} ${frame.height}" role="img">
      <line class="grid-line" x1="${frame.left}" y1="${frame.top + innerH}" x2="${frame.left + innerW}" y2="${frame.top + innerH}" />
      <line class="grid-line" x1="${frame.left}" y1="${frame.top}" x2="${frame.left}" y2="${frame.top + innerH}" />
      ${bars}
      <text class="chart-label" x="${frame.left + innerW / 2}" y="${frame.height - 8}" text-anchor="middle">${options.xLabel || "Cluster"}</text>
      <text class="chart-label" transform="translate(16 ${frame.top + innerH / 2}) rotate(-90)" text-anchor="middle">${options.yLabel || ""}</text>
    </svg>
  `;
}

function drawLines(el, rows, yKey, methodKey = "method", options = {}) {
  const frame = svgFrame(820, 340);
  const innerW = frame.width - frame.left - frame.right;
  const innerH = frame.height - frame.top - frame.bottom;
  const xDomain = extent(rows.map((d) => d.n_clusters));
  const yDomain = extent(rows.map((d) => d[yKey]));
  const methods = [...new Set(rows.map((d) => d[methodKey]))];
  const paths = methods
    .map((method, idx) => {
      const methodRows = rows.filter((d) => d[methodKey] === method).sort((a, b) => a.n_clusters - b.n_clusters);
      const points = methodRows
        .map((d) => `${scale(d.n_clusters, xDomain, [frame.left, frame.left + innerW])},${scale(d[yKey], yDomain, [frame.top + innerH, frame.top])}`)
        .join(" ");
      const color = clusterColors[idx];
      const dots = methodRows
        .map((d) => {
          const cx = scale(d.n_clusters, xDomain, [frame.left, frame.left + innerW]);
          const cy = scale(d[yKey], yDomain, [frame.top + innerH, frame.top]);
          return `<circle cx="${cx}" cy="${cy}" r="4" fill="${color}"><title>${method}, k=${d.n_clusters}: ${num(d[yKey], 4)}</title></circle>`;
        })
        .join("");
      return `<polyline fill="none" stroke="${color}" stroke-width="3" points="${points}" />${dots}`;
    })
    .join("");
  const legend = methods
    .map((method, idx) => `<span style="color:${clusterColors[idx]}">${method}</span>`)
    .join("   ");
  el.innerHTML = `
    <svg viewBox="0 0 ${frame.width} ${frame.height}" role="img">
      <line class="grid-line" x1="${frame.left}" y1="${frame.top + innerH}" x2="${frame.left + innerW}" y2="${frame.top + innerH}" />
      <line class="grid-line" x1="${frame.left}" y1="${frame.top}" x2="${frame.left}" y2="${frame.top + innerH}" />
      ${paths}
      <text class="chart-label" x="${frame.left + innerW / 2}" y="${frame.height - 8}" text-anchor="middle">${options.xLabel || "Number of clusters"}</text>
      <text class="chart-label" transform="translate(16 ${frame.top + innerH / 2}) rotate(-90)" text-anchor="middle">${options.yLabel || yKey}</text>
      <text class="chart-label" x="${frame.left}" y="16">${legend}</text>
    </svg>
  `;
}

function drawHeatmap() {
  const el = $("#profile-heatmap");
  const features = [
    ["avg_yield_kg", "Yield"],
    ["avg_hectares", "Hectares"],
    ["avg_seedrate_kg", "Seed"],
    ["avg_dap_20days", "DAP"],
    ["avg_urea_40days", "Urea"],
    ["avg_potash_50days", "Potash"],
    ["avg_pest_60day_ml", "Pest"],
  ];
  const rows = [...DATA.clusterSummary].sort((a, b) => a.cluster - b.cluster);
  const stats = Object.fromEntries(
    features.map(([key]) => {
      const values = rows.map((r) => Number(r[key]));
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const sd = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length) || 1;
      return [key, { mean, sd }];
    }),
  );
  const cells = rows
    .map(
      (row) => `
        <tr>
          <th>${row.cluster}</th>
          ${features
            .map(([key]) => {
              const z = (Number(row[key]) - stats[key].mean) / stats[key].sd;
              const color = z >= 0 ? `rgba(46,111,149,${Math.min(0.95, 0.25 + Math.abs(z) * 0.26)})` : `rgba(184,74,67,${Math.min(0.95, 0.25 + Math.abs(z) * 0.26)})`;
              return `<td class="heat-cell" style="background:${color}">${num(z, 1)}</td>`;
            })
            .join("")}
        </tr>`,
    )
    .join("");
  el.innerHTML = `
    <div class="table-wrap compact">
      <table>
        <thead><tr><th>Cluster</th>${features.map(([, label]) => `<th>${label}</th>`).join("")}</tr></thead>
        <tbody>${cells}</tbody>
      </table>
    </div>
  `;
}

function initTables() {
  const clusterCols = [
    { key: "cluster", label: "Cluster" },
    { key: "farm_count", label: "Farms", format: (v) => num(v, 0) },
    { key: "avg_yield_kg", label: "Avg Yield", format: (v) => num(v, 0) },
    { key: "avg_hectares", label: "Hectares", format: (v) => num(v, 2) },
    { key: "dominant_agriblock", label: "Agriblock" },
    { key: "dominant_variety", label: "Variety" },
    { key: "dominant_soil_types", label: "Soil" },
    { key: "dominant_nursery", label: "Nursery" },
  ];
  table($("#overview-cluster-table"), DATA.clusterSummary.slice(0, 6), clusterCols.slice(0, 5));
  table($("#cluster-summary-table"), DATA.clusterSummary, clusterCols);
  table($("#best-methods-table"), DATA.bestMethods, [
    { key: "method", label: "Method" },
    { key: "n_clusters", label: "k" },
    { key: "silhouette_score", label: "Silhouette", format: (v) => num(v, 4) },
    { key: "calinski_harabasz_score", label: "Calinski-Harabasz", format: (v) => num(v, 2) },
    { key: "davies_bouldin_score", label: "Davies-Bouldin", format: (v) => num(v, 4) },
  ]);
  table($("#top-anomalies-table"), DATA.topAnomalies.slice(0, 20), [
    { key: "anomaly_rank", label: "Rank" },
    { key: "node", label: "Node" },
    { key: "cluster", label: "Cluster" },
    { key: "combined_anomaly_score", label: "Score", format: (v) => num(v, 3) },
    { key: "Hectares", label: "Hectares" },
    { key: "Agriblock", label: "Agriblock" },
    { key: "Variety", label: "Variety" },
    { key: "Paddy yield(in Kg)", label: "Yield", format: (v) => num(v, 0) },
  ]);
}

function initMetrics() {
  $("#header-stats").innerHTML = [
    metric(num(DATA.pcaMetadata.input_rows, 0), "farms"),
    metric(num(DATA.pcaMetadata.input_features, 0), "processed features"),
    metric(num(DATA.clusterMetadata.selected_baseline_k, 0), "final clusters"),
    metric(num(DATA.anomalyMetadata.top_5pct_anomaly_count, 0), "top anomalies"),
  ].join("");
  $("#overview-metrics").innerHTML = [
    metric("k-means", "final model"),
    metric(num(DATA.clusterMetadata.selected_baseline_metrics.silhouette_score, 4), "silhouette"),
    metric(pct.format(DATA.pcaMetadata.pc1_pc2_cumulative_variance), "PC1 + PC2 variance"),
    metric(num(DATA.graphMetadata.edges, 0), "graph edges"),
  ].join("");
  $("#anomaly-metrics").innerHTML = [
    metric(num(DATA.anomalyMetadata.rows_scored, 0), "farms scored"),
    metric(num(DATA.anomalyMetadata.top_5pct_anomaly_count, 0), "top 5% anomalies"),
    metric(num(DATA.anomalyMetadata.highest_score_node, 0), "highest-score node"),
    metric(num(DATA.anomalyMetadata.highest_combined_anomaly_score, 3), "highest score"),
  ].join("");
}

function initClusterFilter() {
  const select = $("#cluster-filter");
  const clusters = [...DATA.clusterSummary].sort((a, b) => a.cluster - b.cluster);
  select.innerHTML = clusters.map((row) => `<option value="${row.cluster}">Cluster ${row.cluster}</option>`).join("");
  const render = () => {
    const row = DATA.clusterSummary.find((d) => String(d.cluster) === select.value);
    $("#cluster-metrics").innerHTML = [
      metric(num(row.farm_count, 0), "farms"),
      metric(num(row.avg_yield_kg, 0), "avg yield kg"),
      metric(num(row.avg_hectares, 2), "avg hectares"),
      metric(row.dominant_agriblock, "dominant agriblock"),
    ].join("");
  };
  select.addEventListener("change", render);
  render();
}

function initCharts() {
  drawScatter("cluster");
  drawBars($("#yield-bars"), DATA.clusterSummary, "avg_yield_kg", "cluster", {
    sort: true,
    xLabel: "Cluster",
    yLabel: "Average yield (kg)",
  });
  drawHeatmap();
  drawLines($("#model-score-lines"), DATA.methodComparison, "silhouette_score", "method", {
    xLabel: "Number of clusters",
    yLabel: "Silhouette score",
  });
  drawLines(
    $("#pca-variance-chart"),
    DATA.pcaVariance.slice(0, 12).map((d) => ({
      ...d,
      method: "cumulative",
      n_clusters: Number(String(d.component).replace("PC", "")),
      silhouette_score: d.cumulative_explained_variance,
    })),
    "silhouette_score",
    "method",
    {
      xLabel: "Principal component",
      yLabel: "Cumulative explained variance",
    },
  );
  drawLines(
    $("#graph-connectivity"),
    DATA.graphSensitivity.map((d) => ({
      ...d,
      method: "components",
      n_clusters: d.n_neighbors,
      silhouette_score: d.connected_components,
    })),
    "silhouette_score",
    "method",
    {
      xLabel: "Nearest neighbors (k)",
      yLabel: "Connected components",
    },
  );
  drawBars($("#anomaly-cluster-bars"), DATA.clusterAnomalySummary, "top_5pct_anomaly_count", "cluster", {
    xLabel: "Cluster",
    yLabel: "Number of anomalous farms",
  });
}

function initTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("is-active"));
      document.querySelectorAll(".view").forEach((view) => view.classList.remove("is-active"));
      button.classList.add("is-active");
      $(`#${button.dataset.view}`).classList.add("is-active");
    });
  });
}

function initRequestedView() {
  const params = new URLSearchParams(window.location.search);
  const requestedView = params.get("view");
  if (!requestedView || !$(`#${requestedView}`)) return;
  const requestedTab = document.querySelector(`.tab[data-view="${requestedView}"]`);
  if (requestedTab) requestedTab.click();
}

function init() {
  initMetrics();
  initTables();
  initClusterFilter();
  initCharts();
  initTabs();
  initRequestedView();
  $("#scatter-color").addEventListener("change", (event) => drawScatter(event.target.value));
}

init();
