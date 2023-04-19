import pandas as pd
from pathlib import Path
from tqdm import tqdm

class Universe:

    def __init__(self, mkt_data_folder: str = './data/universe') -> None:
        self._root = Path(mkt_data_folder)
        self._mkt_root = Path('./data/mkt')

    def get_liquid_ticker_return(self, inception_date: str) -> pd.DataFrame:

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

        return ret_all


        
