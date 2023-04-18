import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

class Universe:

    def __init__(self, mkt_data_folder: str = './data/mkt'):
        print(mkt_data_folder)
        self._root = Path(mkt_data_folder)
    

    def get_liquid_ticker(self, inception_date: str, lookback: int = 60, top_k: int = 2500):
        csvs = list(self._root.glob('*.csv'))
        csvs = csvs[:10]
        with ProcessPoolExecutor(4) as executor:
            mkt_dfs = list(executor.map(lambda x : pd.read_csv(x), csvs), total=len(csvs))

        mkt_df = pd.concat(mkt_dfs)

        print(mkt_df.head())



        
