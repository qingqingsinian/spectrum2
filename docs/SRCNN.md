# SRCNN - Spectral Residual CNN for Time Series Anomaly Detection

## Overview

SRCNN (Spectral Residual Convolutional Neural Network) is a deep learning approach that combines spectral analysis with CNN for unsupervised time series anomaly detection. It leverages the spectral residual method to identify anomalous patterns in the frequency domain.

## Model Architecture

The SRCNN model consists of:
- **1D Convolutional Layer**: Input window → window features
- **1D Convolutional Layer**: window → 2×window features  
- **Fully Connected Layer**: 2×window → 4×window features
- **Fully Connected Layer**: 4×window → window output
- **Sigmoid Activation**: Final anomaly score output

## Quick Start

### Prerequisites
- Preprocessed dataset in the correct directory structure
- Python environment with required dependencies

## Step-by-Step Execution Guide

### 1. Data Preparation
First, ensure your dataset is properly organized:

```bash
# Create dataset directory structure
mkdir -p datasets/<dataset_name>/train
mkdir -p datasets/<dataset_name>/test

# Copy your CSV files to the train directory
cp your_data.csv datasets/<dataset_name>/train/
```

### 2. Generate Training Data
Convert raw CSV data to the required JSON format:

```bash
cd algorithms/srcnn
python generate_data.py --dataset <dataset_name> --window 32 --step 64
```

**Parameters:**
- `--dataset`: Dataset name (required)
- `--window`: Input window size (default: 32)
- `--step`: Sliding window step size (default: 64)

### 3. Train the Model
Execute the training process:

```bash
python train.py --dataset <dataset_name> --window 32 --epoch 20
```

**Training Parameters:**
- `--dataset`: Dataset name (required)
- `--window`: Input window size (default: 32)
- `--lr`: Learning rate (default: 1e-6)
- `--epoch`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 256)
- `--device`: Training device (auto-selects GPU/CPU)
- `--load`: Pre-trained model path (optional)
- `--save`: Model save directory (default: snapshot)
- `--num_workers`: Data loading workers (default: 8)

### 4. Monitor Training Progress
During training, you'll see:
- Loss values for each epoch
- Learning rate adjustments (50% decay every 10 epochs)
- Model checkpoints saved every 5 epochs

### 5. Evaluate Results
After training completion:

```bash
# Check saved models
ls snapshot/

# Model files are saved as: srcnn_retry<epoch>_<window_size>.bin
```

## Training Process Details

### Data Loading Pipeline
1. **JSON Data Loading**: Uses `gen_set` class to load preprocessed JSON data
2. **Spectral Residual Computation**: Calculates spectral residual features for each sample
3. **Batch Processing**: Processes data in configurable batch sizes

### Feature Processing
For each training sample:
- **Spectral Residual Transform**: Applies FFT-based spectral residual calculation
- **Average Filtering**: Uses sliding window averaging
- **Anomaly Labeling**: Labels anomalies based on spectral residual ratios

### Loss Function & Optimization
- **Loss**: Weighted binary cross-entropy loss
- **Optimizer**: Adam optimizer with configurable learning rate
- **Scheduler**: Learning rate decay (50% every 10 epochs)

## Expected Outputs

### Model Files
Trained models are saved in `snapshot/` directory:
```
snapshot/
├── srcnn_retry5_32.bin    # Checkpoint at epoch 5
├── srcnn_retry10_32.bin   # Checkpoint at epoch 10
├── srcnn_retry15_32.bin   # Checkpoint at epoch 15
└── srcnn_retry20_32.bin   # Final model
```

### Training Logs
Console output includes:
- Epoch-by-epoch loss values
- Learning rate adjustments
- Model save confirmations
- Training time statistics

## Best Practices

### Data Quality
- **Time Series Length**: Ensure CSV files contain sequences longer than window size
- **Data Consistency**: Maintain consistent sampling rates across datasets
- **Missing Values**: Handle missing values during preprocessing

### Training Tips
- **Window Size**: Start with window=32, adjust based on data characteristics
- **Learning Rate**: Use 1e-6 for stable training, reduce if loss oscillates
- **Batch Size**: Increase batch size for faster training on high-memory systems
- **Epochs**: Monitor validation loss to prevent overfitting

### Performance Optimization
- **Data Loading**: Adjust `num_workers` based on CPU cores
- **Memory Management**: Reduce batch size if encountering OOM errors

## Troubleshooting

### Common Issues
1. **Data Loading Errors**: Verify dataset directory structure
2. **Slow Training**: Check GPU utilization and data loading bottlenecks
3. **Poor Convergence**: Adjust learning rate or increase training epochs

### Debug Commands
```bash
# Check dataset structure
ls -la datasets/<dataset_name>/

# Verify generated data
python -c "import json; print(len(json.load(open('datasets/<dataset_name>/train.json'))))"
```
