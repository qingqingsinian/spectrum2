from ...config import WINDOW_SIZE
import torch
from torch import nn

class Anomaly(nn.Module):
    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
        super(Anomaly, self).__init__()
        self.layer1 = nn.Conv1d(window_size, window_size, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv1d(window_size, 2 * window_size, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(2 * window_size, 4 * window_size)
        self.fc2 = nn.Linear(4 * window_size, window_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), self.window_size, 1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)