import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

def plot_mse(pred: pd.DataFrame, gt: pd.DataFrame) -> plt.Axes:
    if pred.index != gt.index:
        logger.error('Index not match for pred and gt')
        raise ValueError('Index not match for pred and gt')
    

    mse_by_time = ((pred - gt) ** 2).sum(axis=1)

    sns.lineplot(mse_by_time)
