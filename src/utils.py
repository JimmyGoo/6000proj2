
import numpy as np
import pandas as pd
from typing import Union, Tuple

def time_series_generator(ret_series: pd.DataFrame, win_len: int) -> Tuple[np.array, np.array]:
    X = []
    y = []
    ret_mat = ret_series.values
    for i in range(ret_mat.shape[0] - win_len):
        X.append(ret_mat[i:i+win_len, :])
        y.append(ret_mat[i+win_len, :])
    X = np.array(X)
    y = np.array(y)

    return X, y