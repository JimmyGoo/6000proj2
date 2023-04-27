import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from typing import List


def plot_mse(preds: List[pd.DataFrame], gt: pd.DataFrame, names: List[str], index: [pd.Timestamp]) -> plt.Axes:
    mse_df = pd.DataFrame(index=index)
    for i, pred in enumerate(preds):

        if pred.shape != gt.shape:
            logger.error('Index not match for pred and gt')
            raise ValueError('Index not match for pred and gt')
        
        mse_by_time = ((pred - gt) ** 2).mean(axis=1)
        mse_df[names[i]] = mse_by_time
        mse_df.index = pd.to_datetime(mse_df.index)
    sns.set(rc={'figure.figsize':(15,10)})
    plot = sns.lineplot(mse_df)
    plot.set(ylim=(0,0.03))
    return plot, mse_df


def generate_report(preds: List[pd.DataFrame], gt: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    
    if len(names) != len(preds):
        logger.error('Preds not match for names')
        raise ValueError('Preds not match for names')

    report = pd.DataFrame(columns=['mse', 'corr', 'deviation'])

    for idx, pred in enumerate(preds):

        mse = ((pred - gt) ** 2).mean(axis=1).mean()

        corr_tickers = []
        for i in range(pred.shape[1]):
            if np.cov([pred[:,i], gt[:,i]])[0,1] == 0:
                corr_tickers.append(0)
            else:
                corr_tickers.append(np.corrcoef([pred[:,i], gt[:,i]])[0,1])
        avg_corr = np.mean(corr_tickers)

        nd = np.mean(np.clip(np.abs(pred - gt) / (np.abs(gt) + 1e-6), 0, 2.0).flatten())

        report.loc[names[idx]] = [mse, avg_corr, nd]

    return report
