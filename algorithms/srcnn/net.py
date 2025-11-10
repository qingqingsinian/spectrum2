import torch
from torch import nn
import os

class SRCNN(nn.Module):
    def __init__(self, window=1024):
        self.window = window
        super(SRCNN, self).__init__()
        self.layer1 = nn.Conv1d(window, window, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv1d(window, 2 * window, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(2 * window, 4 * window)
        self.fc2 = nn.Linear(4 * window, window)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), self.window, 1)

        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


def save_model(model: SRCNN, model_path: str):
    parent_dir = os.path.dirname(model_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    try:
        torch.save(model.state_dict(), model_path)
    except:
        torch.save(model, model_path)


def load_model(model: SRCNN, path: str):
    print("loading %s" % path)
    with open(path, 'rb') as f:
        pretrained = torch.load(f, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in pretrained.items() if k in model_dict}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)
    return model
