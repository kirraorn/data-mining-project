"""Run the end-to-end solar-wind mining and catalog-comparison pipeline.

Course: CIS 4930, Fall 2026
Team: Kirra Orndorff, Michael Murgia, Bianca Blevins

AI use: A generative AI assistant helped review and clarify project documentation and
comments. The team is responsible for reviewing the implementation, analysis, and
submitted results.
"""

import time

from src.p1_preprocess import run_preprocessing
from src.p2_clustering import run_clustering
from src.p2_b_analyze_clusters import run_cluster_analysis
from src.p3_anomaly import run_anomaly_detection
from src.p4_rules import run_rule_extraction
from src.p5_catalogs import run_catalog_labeling
from src.p6_compare_cluster_anomalies import run_cluster_anomaly_comparison
from src.p7_compare_rules import run_rule_comparison
from src.p8_compare_supervised import run_supervised_comparison


def main():
    """Run pipeline stages in dependency order and report elapsed time."""
    print("----------------------------------------")
    print("Starting Solar Wind Data Mining Pipeline")
    print("Estimated total run time: about 1 minute")
    print("----------------------------------------")

    start_time = time.time()

    # Create the shared feature table before the modeling stages consume it.
    print("Running preprocessing...")
    run_preprocessing()

    print("Running clustering...")
    run_clustering()

    print("Running cluster analysis and generating figures...")
    run_cluster_analysis()

    print("Running anomaly detection...")
    run_anomaly_detection()

    print("Running rule extraction...")
    run_rule_extraction()

    # Comparison stages align model outputs with catalog labels and score them.
    print("Building comparison data (ICME and CIR/HSS catalog labels)...")
    run_catalog_labeling()

    print("Comparing clusters and anomalies against the catalogs...")
    run_cluster_anomaly_comparison()

    print("Comparing decision-tree rules against Lepping-style thresholds...")
    run_rule_comparison()

    print("Comparing a supervised Random Forest against the unsupervised methods...")
    run_supervised_comparison()

    elapsed_time = time.time() - start_time
    print("--------------------------------------------------")
    print(f"Pipeline completed in {elapsed_time:.2f} seconds.")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
