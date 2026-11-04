import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

def run_rule_extraction():
    print("\n--- Running Interpretable Rule Extraction (Decision Tree) ---")
    
    # 1. Load clustered results
    df = pd.read_csv('outputs/omni_clustered_results.csv')
    
    # Define features and target (using KMEANS_CLUSTER as target labels)
    exclude_cols = ['YEAR', 'DOY', 'HR', 'KMEANS_CLUSTER', 'Timestamp', 'date', 'time']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].dropna()
    y = df.loc[X.index, 'KMEANS_CLUSTER']
    
    # 2. Train a shallow decision tree (depth 4 for readability)
    print("Training shallow Decision Tree (max_depth=4)...")
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X, y)
    
    # 3. Export human-readable text rules
    tree_rules = export_text(dt, feature_names=list(feature_cols))
    print("\nExtracted Decision Rules:")
    print(tree_rules)
    
    # Save rules to a text file for your report write-up
    with open('outputs/extracted_decision_rules.txt', 'w') as f:
        f.write(tree_rules)
    print("Rules successfully saved to outputs/extracted_decision_rules.txt")

if __name__ == "__main__":
    run_rule_extraction()