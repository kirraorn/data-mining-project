import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def run_anomaly_detection():
    print("\n--- Running Isolation Forest Anomaly Detection ---")
    
    # 1. Load processed features
    df = pd.read_csv('outputs/processed_omni_features.csv')
    
    # Drop non-feature columns if present
    exclude_cols = ['YEAR', 'DOY', 'HR', 'Timestamp', 'date', 'time']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].dropna()
    
    # 2. Scale features for anomaly detection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Fit Isolation Forest (contamination set to 5% based on expected disturbance frequency)
    print("Fitting Isolation Forest model...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    preds = iso_forest.fit_predict(X_scaled)
    scores = iso_forest.decision_function(X_scaled)
    
    # 4. Save results
    df_results = df.loc[X.index].copy()
    df_results['ANOMALY_PRED'] = preds  # -1 for anomalies, 1 for normal
    df_results['ANOMALY_SCORE'] = scores
    
    output_path = 'outputs/omni_anomaly_results.csv'
    df_results.to_csv(output_path, index=False)
    print(f"Anomaly detection complete. Results saved to {output_path}")

if __name__ == "__main__":
    run_anomaly_detection()