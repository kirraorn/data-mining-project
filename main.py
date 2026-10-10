"""
Main entry point for the Solar Wind Data Mining Pipeline. This script orchestrates the entire pipeline, including preprocessing, clustering, anomaly detection, and rule extraction.
Team Members: Kirra Orndorff (KNO22A), Michael Murgia (), Bianca Blevins ()
Course: CIS 4930 Fall 2026
"""


import time
from src.p1_preprocess import run_preprocessing

def main():
    print("----------------------------------------")
    print("Starting Solar Wind Data Mining Pipeline")
    print("----------------------------------------")

    start_time = time.time()


    #run preprocessing
    print("Running preprocessing...")
    run_preprocessing()

    # put clusturing, anomoalyr, and rule extraction here
    # print("Running clustering...")
    # ... clustering code ...

    # print("Running anomaly detection...")
    # ... anomaly detection code ...

    # print("Running rule extraction...")
    # ... rule extraction code ...

    elapsed_time = time.time() - start_time
    print("--------------------------------------------------")
    print(f"Pipeline completed in {elapsed_time:.2f} seconds.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()