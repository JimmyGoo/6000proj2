import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

def plot_mse(pred: pd.DataFrame, gt: pd.DataFrame, index: [pd.Timestamp]) -> plt.Axes:
    if pred.shape != gt.shape:
        logger.error('Index not match for pred and gt')
        raise ValueError('Index not match for pred and gt')
    
    mse_by_time = ((pred - gt) ** 2).mean(axis=1)
    mse_df = pd.Series(mse_by_time, index=index)
    mse_df.index = pd.to_datetime(mse_df.index)
    plot = mse_df.plot.line()
    return plot, mse_df
