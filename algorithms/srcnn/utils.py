import csv
import json
import pickle
from pathlib import Path
from typing import Tuple, List

import numpy as np
from torch.utils.data import Dataset

from net import *
from spectral_residual import SpectralResidual


def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def read_csv_kpi(path: str) -> Tuple[list, list, list]:
    timestamp = []
    value = []
    label = []
    with open(path) as f:
        content = csv.reader(f, delimiter=',')
        cnt = 0
        for row in content:
            if cnt == 0:
                cnt += 1
                continue
            timestamp.append(int(row[0]))
            value.append(float(row[1]))
            label.append(int(row[2]))
            cnt += 1
        f.close()
    return timestamp, value, label


def read_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    timestamp = []
    value = []
    with open(path, 'r+') as f:
        content = csv.reader(f, delimiter=',')
        cnt = 0
        for row in content:
            if cnt == 0:
                cnt += 1
                continue
            timestamp.append(cnt)
            value.append(float(row[1]))
        f.close()
    return np.array(timestamp), np.array(value)


def get_train_files(dataset_name: str) -> List[str]:
    """
    Get all CSV files in the dataset directory

    Args:
        dataset_name: Name of the dataset

    Returns:
        List of file paths
    """
    base_dir = Path(__file__).resolve().parent.parent / 'datasets' / dataset_name / 'train'

    if not base_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

    return sorted([str(file) for file in base_dir.glob('*.csv') if file.is_file()])


def average_filter(values: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Calculate the sliding window average for the given time series.
    Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
    
    Args:
        values: Input array of float numbers
        window_size: Window size for the moving average
        
    Returns:
        Array of values after the average_filter process
    """
    if not isinstance(values, np.ndarray):
        values = np.asarray(values, dtype=np.float64)

    n = min(window_size, len(values))

    # Handle edge cases
    if len(values) == 0:
        return np.array([])
    if len(values) <= n:
        return np.mean(values) * np.ones_like(values)

    # Compute cumulative sum
    cumsum = np.cumsum(values, dtype=np.float64)
    result: np.ndarray = np.empty_like(values, dtype=np.float64)  # type: ignore

    # calculate for the first n elements
    result[:n] = cumsum[:n] / np.arange(1, n + 1)

    # calculate for the rest of the elements
    if len(values) > n:
        result[n:] = (cumsum[n:] - cumsum[:-n]) / n

    return result


class gen_set(Dataset):
    def __init__(self, width: int, data_path: str):
        self.genlen = 0
        self._len = self.genlen
        self.width = width
        with open(data_path, 'r+') as fin:
            self.kpinegraw = json.load(fin)
        self.negrawlen = len(self.kpinegraw)
        print('length :', len(self.kpinegraw))
        self._len += self.negrawlen
        self.kpineglen = 0
        self.control = 0.

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        idx = index % self.negrawlen
        datas = self.kpinegraw[idx]
        datas: np.ndarray = np.array(datas)
        data: np.ndarray = datas[0, :].astype(np.float64)
        lbs: np.ndarray = datas[1, :].astype(np.float64)
        wave = SpectralResidual.spectral_residual(data)
        waveavg = average_filter(wave)
        for i in range(self.width):
            if wave[i] < 0.001 and waveavg[i] < 0.001:
                lbs[i] = 0
                continue
            ratio = wave[i] / waveavg[i]
            if ratio < 1.0 and lbs[i] == 1:
                lbs[i] = 0
            if ratio > 5.0:
                lbs[i] = 1
        srscore = abs(wave - waveavg) / (waveavg + 0.01)
        sortid = np.argsort(srscore)
        for idx in sortid[-2:]:
            if srscore[idx] > 5:
                lbs[idx] = 1
        resdata = torch.from_numpy(100 * wave)
        reslb = torch.from_numpy(lbs)
        return resdata, reslb
