import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def run_cluster_analysis():
    """Loads clustered OMNI data, prints physical profiles, and generates evaluation plots."""
    print("Running cluster analysis and generating figures...")
    
    # 1. Load Data
    df = pd.read_csv('outputs/omni_clustered_results.csv')
    metrics_df = pd.read_csv('outputs/cluster_evaluation_metrics.csv')

    # Dynamically find the cluster column (case-insensitive search for 'cluster')
    cluster_col = next((col for col in df.columns if 'cluster' in col.lower()), None)
    if cluster_col is None:
        raise ValueError("Could not find any column containing 'cluster'")

    print(f"    Using cluster column: '{cluster_col}'")

    print("\n--- CORE PHYSICAL PARAMETERS SUMMARY ---")
    core_cols = [c for c in ['FLOW_SPEED', 'ABS_B', 'PROTON_DENSITY', 'TEMP', 'PLASMA_BETA'] if c in df.columns]
    if core_cols:
        print(df.groupby(cluster_col)[core_cols].mean())

    # 2. Plot 1: Davies-Bouldin Index & Inertia Sweep
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Davies-Bouldin Index (Lower is Better)', color=color, fontsize=12)
    ax1.plot(metrics_df['k'], metrics_df['davies_bouldin'], marker='o', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='Optimal k=5')

    ax2 = ax1.twinx()  
    color = 'tab:gray'
    ax2.set_ylabel('Inertia (Elbow Method)', color=color, fontsize=12)
    ax2.plot(metrics_df['k'], metrics_df['inertia'], marker='s', color=color, linestyle=':', alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Solar Wind Clustering Optimization (k-Means Sweeps)', fontsize=14)
    fig.tight_layout()
    plt.savefig('outputs/k_means_sweep_evaluation.png', dpi=300)
    plt.close()

    # 3. Plot 2: 2D PCA Cluster Projection
    exclude_cols = [cluster_col, 'k', 'Timestamp', 'date', 'time']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].dropna()
    clusters = df.loc[X.index, cluster_col]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.4, s=5)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title('2D PCA Projection of Solar Wind Regimes (k=5)', fontsize=14)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('outputs/pca_cluster_projection.png', dpi=300)
    plt.close()

    print("    Figures successfully saved to outputs folder")


if __name__ == "__main__":
    run_cluster_analysis()