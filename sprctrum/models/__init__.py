from ..config import WINDOW_SIZE
import polars as pl

class BaseModel:
    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
    ):
        self.window_size = window_size

    def fit(self, values: pl.Series, labels: pl.Series):
        raise NotImplementedError
    
    def predict(self, values: pl.Series) -> pl.Series:
        raise NotImplementedError