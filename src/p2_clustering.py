"""
Clustering and Regime Discovery Module
Implements k-Means parameter sweeps (k=2 to 10), optimization metrics 
(Davies-Bouldin index), and solar wind regime assignment.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

def run_clustering():
    print("Clustering: Loading processed features...")
    input_path = os.path.join("outputs", "processed_omni_features.csv")
    
    if not os.path.exists(input_path):
        print(f"    Processed features not found")
        return

    df = pd.read_csv(input_path)
    print(f"    Loaded {len(df)} rows for clustering analysis.")

    # Select features for clustering (excluding time index columns)
    feature_cols = [col for col in df.columns if col not in ['YEAR', 'DOY', 'HR']]
    X = df[feature_cols].fillna(0)

    # Standardize features for distance-based clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("    Executing k-Means parameter sweeps (k=2 to 10)...")
    results = []
    
    # Sweep k from 2 to 10
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Compute evaluation scores
        db_score = davies_bouldin_score(X_scaled, labels)
        inertia = kmeans.inertia_
        
        results.append({
            'k': k,
            'inertia': inertia,
            'davies_bouldin': db_score
        })
        print(f"    k={k}: Davies-Bouldin Index = {db_score:.4f}, Inertia = {inertia:.2f}")

    results_df = pd.DataFrame(results)
    
    # Save evaluation metrics
    os.makedirs("outputs", exist_ok=True)
    eval_output = os.path.join("outputs", "cluster_evaluation_metrics.csv")
    results_df.to_csv(eval_output, index=False)
    print(f"    Sweeps complete. Metrics saved to {eval_output}.")

    # Automatically select optimal k based on minimum Davies-Bouldin index (lower is better)
    best_k_row = results_df.loc[results_df['davies_bouldin'].idxmin()]
    best_k = int(best_k_row['k'])
    print(f"   Optimal k selected via Davies-Bouldin index: k={best_k}")

    # Run final k-Means with best_k and assign cluster labels
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['KMEANS_CLUSTER'] = final_kmeans.fit_predict(X_scaled)

    # Save clustered dataset for downstream analysis
    clustered_output = os.path.join("outputs", "omni_clustered_results.csv")
    df.to_csv(clustered_output, index=False)
    print(f"    Clustered dataset saved to {clustered_output}.")