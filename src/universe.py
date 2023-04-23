import statsmodels.api as sm
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger

RESIDUAL_LOOKBACK = 60
RESIDUAL_START = '2018-06-30'
RESIDUAL_END = '2023-03-31'
RESIDUAL_TRAIN_FREQ = '3M'

class Universe:

    def __init__(self, mkt_data_folder: str = './data/universe') -> None:
        self._root = Path(mkt_data_folder)
        self._mkt_root = Path('./data/mkt')

    def get_liquid_ticker_return(self, inception_date: str, residual: bool = False) -> pd.DataFrame:

        universe = pd.read_csv(self._root / f'{inception_date}.csv', squeeze=True)
        ## get ticker returns
        rets = []
        for aid in tqdm(universe):
            ret = pd.read_csv(self._mkt_root / f'{aid}.csv', index_col=0)
            ret = ret['Close'].pct_change().dropna()
            ret = ret.rename(aid)
            rets.append(ret)

        ret_all = pd.concat(rets, axis=1)
        ret_all = ret_all.sort_index()
        ret_all = ret_all.fillna(0)

        ## using index model to get residual return(alpha) for each ticker
        if residual:
            ## rolling beta
            logger.info('generate residual')
            train_dates = pd.date_range(RESIDUAL_START, RESIDUAL_END, freq=RESIDUAL_TRAIN_FREQ)
            train_dates = [td.strftime('%Y-%m-%d') for td in train_dates]
            sp500_ret = pd.read_csv('./data/SP500.csv', index_col=0)['Close'].pct_change().dropna()
            resi = pd.DataFrame(columns=ret_all.columns, index=ret_all.index)

            for aid in tqdm(universe):
                for i in range(len(train_dates) - 1):

                    td = train_dates[i]
                    td_next = train_dates[i+1]
                    Y = ret_all.loc[:td, aid].iloc[-RESIDUAL_LOOKBACK:]
                    dates = Y.index
                    X = sp500_ret.loc[dates]
                    X = sm.add_constant(X)
                    model = sm.OLS(Y,X)
                    results = model.fit()
                    beta = results.params[1]
                    
                    pred_index = ret_all.loc[td:td_next, aid].iloc[1:].index
                    resi_pred = ret_all.loc[pred_index, aid] - beta * sp500_ret.loc[pred_index]
                    resi.loc[pred_index, aid] = resi_pred

            return resi

        return ret_all



        
