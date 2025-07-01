import os
import json
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load PCA data from JSON ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PCA_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "initial", "pca", "pca_data_removed_errors.json")
with open(PCA_JSON_PATH, "r") as f:
    pca_data = json.load(f)

# --- Convert to DataFrame ---
rows = []
for k, v in pca_data.items():
    row = {
        "id": k,
        "server": v["server"],
        "gender": v["gender"],
        "PC1": v["PC1"],
        "PC2": v["PC2"],
        "PC3": v["PC3"],
        "Speed_MPH": v.get("Speed_MPH", np.nan)
    }
    rows.append(row)
df = pd.DataFrame(rows)

# --- Visualize all serves in PCA space by server ---
os.makedirs(os.path.join(PROJECT_ROOT, "data", "initial", "visualizations", "server_patterns"), exist_ok=True)
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="server", palette="tab20", alpha=0.7)
plt.title("All Serves in PCA Space (by Server)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "data", "initial", "visualizations", "server_patterns", "all_servers_pca2d.png"))
plt.close()

# --- Cluster analysis per server and pattern classification ---
summary = []
pattern_types = []
for server in df["server"].unique():
    sub = df[df["server"] == server].copy()
    X_server = sub[["PC1", "PC2", "PC3"]].values
    if len(sub) < 3:
        n_clusters = 0
        n_noise = 0
        compactness = np.nan
        sub["cluster"] = -1
        pattern = "too_few_serves"
    else:
        clustering = DBSCAN(eps=150, min_samples=3).fit(X_server)
        sub["cluster"] = clustering.labels_
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        n_noise = sum(clustering.labels_ == -1)
        # Compactness: only for clusters with >1 point
        cluster_compactness = []
        for c in set(clustering.labels_):
            if c == -1:
                continue
            mask = clustering.labels_ == c
            if np.sum(mask) > 1:
                cluster_points = X_server[mask]
                cluster_center = cluster_points.mean(axis=0)
                cluster_compactness.append(np.mean(np.linalg.norm(cluster_points - cluster_center, axis=1)))
        compactness = np.mean(cluster_compactness) if len(cluster_compactness) > 0 else np.nan
        if n_clusters == 0:
            pattern = "no_cluster"
        elif n_clusters == 1:
            if compactness < 120:
                pattern = "single_tight"
            else:
                pattern = "single_spread"
        elif n_clusters > 1:
            pattern = f"multi_cluster_{n_clusters}"
        else:
            pattern = "other"
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=sub["PC1"], y=sub["PC2"], hue=sub["cluster"], palette="tab10", style=sub["cluster"] >= 0, alpha=0.8)
    plt.title(f"{server} Serves in PCA Space (clusters)")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "data", "initial", "visualizations", "server_patterns", f"{server}_pca2d_clusters.png"))
    plt.close()
    summary.append({
        'server': server,
        'num_serves': len(sub),
        'num_clusters': n_clusters,
        'num_noise': n_noise,
        'compactness': compactness,
        'pattern': pattern
    })
    pattern_types.append(pattern)

# --- Save summary table ---
sum_df = pd.DataFrame(summary)
sum_df.to_csv(os.path.join(PROJECT_ROOT, "data", "initial", "visualizations", "server_patterns", "server_cluster_summary.csv"), index=False)

# --- Print pattern summary ---
pattern_counts = pd.Series(pattern_types).value_counts()
print("\n=== Server Cluster Pattern Summary ===")
print(pattern_counts)
print("\nSee server_cluster_summary.csv and PNGs in data/initial/visualizations/server_patterns/")
