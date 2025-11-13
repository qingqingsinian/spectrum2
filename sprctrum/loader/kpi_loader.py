from ..config import WINDOW_SIZE
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from ..models.sr_cnn.spectral_residual import spectral_residual, average_filter

class SRCNNKPILoader(Dataset):
    def __init__(
        self,
        values: pl.Series,
        labels: pl.Series,
        window_size: int = WINDOW_SIZE,
        step: int = WINDOW_SIZE // 2,
    ):
        self.control = 0
        self.window_size = window_size
        self.step = step
        v = []
        l = []
        length = len(values)
        for pt in range(window_size, length, self.step):
            head = max(0, pt - window_size)
            tail = min(length, pt)
            data = np.array(values[head:tail], dtype=np.float64)
            num = np.random.randint(1, 10)
            ids = np.random.choice(window_size, num, replace=False)
            lbs = np.zeros(self.window_size, dtype=np.int64)
            mean = np.mean(data)
            dataavg = average_filter(data)
            var = np.var(data)
            for id in ids:
                data[id] += (dataavg[id] + mean) * np.random.randn() * min((1 + var), 10)
                lbs[id] = 1
            v.append([data.tolist()])
            l.append([lbs.tolist()])
        self.values = np.array(v).squeeze()
        self.labels = np.array(l).squeeze()
        self.length = len(self.values)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx % self.length
        values = self.values[idx]
        labels = self.labels[idx]
        wave = spectral_residual(values)
        wave_avg = average_filter(wave)
        for i in range(self.window_size):
            if wave[i] < 0.001 and wave_avg[i] < 0.001:
                labels[i] = 0
                continue
            ratio = wave[i] / wave_avg[i]
            if ratio < 1.0 and labels[i] == 1:
                labels[i] = 0
            if ratio > 5.0:
                labels[i] = 1
        srscore = abs(wave - wave_avg) / (wave_avg + 0.01)
        sortid = np.argsort(srscore)
        for idx in sortid[-2:]:
            if srscore[idx] > 5:
                labels[idx] = 1
        resdata = torch.from_numpy(100 * wave)
        reslb = torch.from_numpy(labels)
        return resdata, reslb