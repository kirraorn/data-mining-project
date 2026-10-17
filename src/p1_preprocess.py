"""
Preprocessing module for the Solar Wind Data Mining Pipeline
Handles the OMNI data preprocessing, including data cleaning, normalization, and feature engineering
"""

import pandas as pd
import numpy as np
import os

def run_preprocessing():
    print("   Loading data...")

    data_path = os.path.join("data", "omni_data.csv")

    # If path isn't there
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

    # Replace missing values with NaN and interpolate linearly
    df.replace([999.9, 99.9, 9999.9, 99999.0, 999.0, 99.0], np.nan, inplace=True)
    df.interpolate(method='linear', limit_direction='both', inplace=True)

    # Isolate the features and target variable
    target_features = ['BZ', 'BY', 'BX', 'ABS_B', 'FLOW_SPEED', 'PROTON_DENSITY', 'TEMP', 'PLASMA_BETA']

    print("     Making 3, 6, and 24-hour rolling window statistics...")
    windows = [3, 6, 24]
    
    feature_df = pd.DataFrame()
    feature_df['YEAR'] = df['YEAR']
    feature_df['DOY'] = df['DOY']
    feature_df['HR'] = df['HR']
    
    for col in target_features:
        if col in df.columns:
            # Base feature
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