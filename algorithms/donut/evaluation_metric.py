from sklearn.metrics import *
import numpy as np

def range_lift_with_delay(array: np.ndarray, label: np.ndarray, delay=None, inplace=False) -> np.ndarray:
    """
    @param delay: maximum acceptable delay,
        delay=None,indicates only any point is detected correctly, the whole anomaly segment is detected
        delay=0,indicates that don's allow the delay
    @param array: predication label 
    @param label: true label
    @param inplace:
    @return: new_array
    """
    assert np.shape(array) == np.shape(label)
    if delay is None:
        delay = len(array)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = (label[0] == 1)
    new_array = np.copy(array) if not inplace else array
    pos = 0
    for sp in splits:
        if is_anomaly:
            ptr = min(pos + delay + 1, sp)
            new_array[pos: ptr] = np.max(new_array[pos: ptr])
            new_array[ptr: sp] = np.maximum(new_array[ptr: sp], new_array[pos])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        ptr = min(pos + delay + 1, sp)
        new_array[pos: sp] = np.max(new_array[pos: ptr])
    return new_array