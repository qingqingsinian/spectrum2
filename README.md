# Spectrum - Time Series Anomaly Detection Algorithms

A comprehensive collection of time series anomaly detection algorithms for identifying anomalous patterns in temporal data. This project provides implementations of state-of-the-art deep learning and machine learning approaches for unsupervised anomaly detection.

## Quick Start

### Prerequisites
- Python 3.8+
- conda

### Installation
```bash
# Clone the repository
git clone https://github.com/DeepShield-AI/spectrum.git
cd spectrum

# Create conda environment
conda env create -f environment.yml
conda activate spectrum
```

## Datasets

### Supported Datasets
We support the following public time series anomaly detection datasets:

| Dataset | Domain | Description | Dimensions | Anomaly Rate |
|---------|--------|-------------|------------|--------------|
| **MSL** | Space | Mars Science Laboratory telemetry | 55 | ~10.7% |
| **SMAP** | Space | Soil Moisture Active Passive satellite | 25 | ~13.1% |
| **SMD** | IT | Server Machine Dataset | 38 | ~4.2% |
| **PSM** | IT | Pooled Server Metrics | 25 | ~27.9% |
| **SWAT** | Industrial | Secure Water Treatment testbed | 51 | ~12.1% |
| **KPI** | Web | Key Performance Indicators | 1 | Variable |

### Data Download & Setup

1. **Download datasets** from: `https://cloud.tsinghua.edu.cn/d/75ceadaca416485e9f09/`
Download the `datasets.zip` file and move it to the `spectrum` directory

2. **Unzip datasets**:
```bash
unzip datasets.zip
```

2. **Extract and organize** the data as follows:
```
datasets/
└── kpi/
    ├── raw/
    │   ├── phase2_train.csv            # training set
    │   ├── phase2_train.csv.zip        # raw training set
    │   └── phase2_ground_truth.hdf     # test set
    │   └── phase2_ground_truth.hdf.zip # raw test set
    │   └── phase2_ground_truth.parquet # test set in Parquet format
    └── train/                          # training set divided according to KPI ID
    └── test/                           # test set divided according to KPI ID
```

3. **Run preprocessing scripts**:
```bash
# KPI-specific preprocessing (with missing value handling)
cd exp/preprocess/

# Manual cell execution instructions:
# 1. After opening the kpi.ipynb, run cells sequentially
# 2. Or use Cell -> Run All to execute all cells at once
# 3. For step-by-step execution, use Cell -> Run Cells to run selected cells
# 4. Monitor the output and adjust parameters as needed between cells

jupyter notebook kpi.ipynb  # Interactive preprocessing
```

### Data Processing Pipeline

Our preprocessing pipeline includes:

1. **Data Loading & Validation**
   - Format standardization (CSV/HDF to Parquet)
   - Schema validation and type conversion
   - Timestamp normalization

2. **Missing Value Analysis & Imputation**
   - Gap detection and characterization
   - Missing value statistics and visualization

3. **Data Splitting & Normalization**
   - Train/validation/test splits
   - Min-max or z-score normalization
   - Sliding window generation

**Processing Scripts Location:**
- `exp/preprocess/kpi.ipynb` - Interactive KPI preprocessing

## Algorithms

### Implemented Algorithms

| Algorithm | Type | Paper | Documentation |
|-----------|------|-------|---------------|
| **SRCNN** | CNN-based | Spectral Residual CNN | [docs/SRCNN.md](docs/SRCNN.md) |

### Algorithm Categories

- **Deep Autoencoders**: USAD, DAGMM, Donut
- **Spectral Methods**: SRCNN, SaVAE-SR
- **Temporal Convolution**: ModernTCN
- **Clustering-based**: DCdetector, DAGMM

## Results & Evaluation

### Evaluation Metrics
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **AUC-PR**: Area under the Precision-Recall curve