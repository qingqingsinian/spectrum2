import torch


def device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if torch.backends.mps.is_available() and torch.backends.mps.is_built()
            else "cpu"
        )
    )
