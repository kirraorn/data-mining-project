# NASA OMNI Solar Wind Data Mining Pipeline

An end-to-end unsupervised machine learning and feature engineering pipeline designed to cluster and analyze NASA OMNI solar wind telemetry data. This project uses k-Means clustering and parameter sweeps ($k \in [2, 10]$) evaluated via the Davies-Bouldin Index to discover physical space weather regimes and structural telemetry anomalies.

---

## 📌 Project Overview

Space weather events—such as Coronal Mass Ejections (CMEs) and Stream Interaction Regions (SIRs)—exert significant influence on Earth's magnetosphere and satellite infrastructure. This pipeline automates the extraction, rolling feature engineering, unsupervised clustering, and dimensionality reduction (PCA) of 26,000+ hourly observations from NASA's OMNI dataset.

---

## 💻 Tech Stack

- **Language:** Python 3.10+
- **Data Mining & Machine Learning:** `scikit-learn`, `numpy`, `pandas`
- **Data Visualization:** `matplotlib`, `seaborn`

---

## 📁 Repository Structure

```text
data-mining-project/
├── data/                             # Raw telemetry data (NASA OMNI)
│   └── omni_raw_data.csv
├── outputs/                          # Pipeline outputs, metrics, and figures
│   ├── processed_omni_features.csv
│   ├── cluster_evaluation_metrics.csv
│   ├── omni_clustered_results.csv
│   ├── k_means_sweep_evaluation.png
│   └── pca_cluster_projection.png
├── src/                              # Source code modules
│   ├── p1_preprocess.py              # Feature engineering & rolling window calculations
│   ├── p2_cluster.py                 # k-Means parameter sweeps & model fitting
│   └── p2_b_analyze_clusters.py      # Profiling, PCA projection, & figure generation
│   └── p3_anomaly.py                 # 
│   └── p4_rules.py                   # 
├── main.py                           # Single-entry pipeline orchestration script
├── requirements.txt                  # Python environment dependencies
└── README.md                         # Project documentation
```

---

## 🛠️ Setup & Installation

### Prerequisites
- **Python 3.10+**
- `pip` and `virtualenv`

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/data-mining-project.git
   cd data-mining-project
   ```

2. **Create and Activate a Virtual Environment:**
   - **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Program:**
```bash
python main.py
```

