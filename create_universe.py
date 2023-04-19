import pandas as pd
import fire 
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Tuple, List

def select_by_turnover(inception_date: str, top_k: int = 2500):
    inception_date
    csvs = list(Path('./data/mkt').glob('*.csv'))
    turnover_dfs = []
    for csv in tqdm(csvs):
        df = pd.read_csv(csv, index_col=0)
        if df.index[0] > inception_date:
            continue
        df['turnover1d'] = df['Volume'] / (df['share'] * df['Close'])
        df['turnover_cumsum'] = df['turnover1d'].rolling(60).sum()
        turnover_df = df[['ticker', 'turnover_cumsum']]

        turnover_df = turnover_df.loc[:inception_date].iloc[-1:]
        turnover_df = turnover_df.reset_index().rename(columns={'index': 'date'})
        turnover_dfs.append(turnover_df)
    
    turnover = pd.concat(turnover_dfs)
    
    top_tickers = turnover.sort_values('turnover_cumsum', ascending=False).head(top_k)['ticker']
    csv_path = Path('./data/universe')
    csv_path.mkdir(parents=True, exist_ok=True)

    csv_path = csv_path / f"{inception_date}.csv"
    top_tickers.to_csv(csv_path, index=None)

def create_universe(start_date: str = '2018-12-31', end_date: str = '2023-03-31', freq: str = '3M', top_k: int = 2500):
    dates = pd.date_range(start_date, end_date, freq=freq)
    for d in dates:
        logger.info(f"select universe by date {d.strftime('%Y-%m-%d')}")
        select_by_turnover(d.strftime('%Y-%m-%d'), top_k=top_k)
        
if __name__ == '__main__':
    fire.Fire(create_universe)


