"""Clean selected OMNI missing-value codes and build rolling-window features.

The generated feature table is consumed by the clustering and comparison stages.
Some OMNI fill codes (including large FLOW_SPEED, TEMP, and PLASMA_BETA values)
are not included in the replacement list; see the README limitation notes.
"""

import pandas as pd
import numpy as np
import os

def run_preprocessing():
    print("   Loading data...")

    data_path = os.path.join("data", "omni_data.csv")

    # Stop here so downstream stages do not run against a missing input file.
    if not os.path.exists(data_path):
        print(f"Error: Data file not found")
        return

    column_names = [
        'YEAR', 'DOY', 'HR', 
        'ABS_B', 'BX', 'BY', 'BZ', 'PARAM_5', 'PARAM_6', 
        'TEMP', 'PROTON_DENSITY', 'FLOW_SPEED', 'PLASMA_BETA'
    ]
    
    df = pd.read_csv(data_path, sep=r'\s+', header=None, names=column_names, engine='python')
    print(f"   Data loaded successfully")

    # Replace the recognized sentinel values, then fill gaps from neighboring hours.
    df.replace([999.9, 99.9, 9999.9, 99999.0, 999.0, 99.0], np.nan, inplace=True)
    df.interpolate(method='linear', limit_direction='both', inplace=True)

    # Keep time fields for alignment; model features are the solar-wind measurements.
    target_features = ['BZ', 'BY', 'BX', 'ABS_B', 'FLOW_SPEED', 'PROTON_DENSITY', 'TEMP', 'PLASMA_BETA']

    print("     Making 3, 6, and 24-hour rolling window statistics...")
    windows = [3, 6, 24]
    
    feature_df = pd.DataFrame()
    feature_df['YEAR'] = df['YEAR']
    feature_df['DOY'] = df['DOY']
    feature_df['HR'] = df['HR']
    
    for col in target_features:
        if col in df.columns:
            # Preserve each source measurement alongside its derived features.
            feature_df[col] = df[col]
            for w in windows:
                # Rolling mean
                feature_df[f'{col}_roll_mean_{w}h'] = df[col].rolling(window=w, min_periods=1).mean()
                # Rolling std dev
                feature_df[f'{col}_roll_std_{w}h'] = df[col].rolling(window=w, min_periods=1).std().fillna(0)
                # Rate of change (difference over window)
                feature_df[f'{col}_roc_{w}h'] = df[col].diff(periods=w).fillna(0)

    # Save processed features for the next phase
    os.makedirs("outputs", exist_ok=True)
    output_file = os.path.join("outputs", "processed_omni_features.csv")
    feature_df.to_csv(output_file, index=False)
    
    print(f"   Feature engineering complete. Saved processed data to {output_file}.")
