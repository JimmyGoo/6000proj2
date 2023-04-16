import fire
import yfinance as yf
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from concurrent.futures import ProcessPoolExecutor

def get_ticker_mkt_data(args):
    try:
        path, start_date, end_date, tick = args
        tick_obj = yf.Ticker(tick)
        hist = tick_obj.history(period="5Y")
        hist['ticker'] = tick
        hist.index = hist.index.strftime("%Y-%m-%d")
        
        shares = tick_obj.get_shares_full(start="2019-01-01", end=None)
        shares.index = shares.index.strftime('%Y-%m-%d')
        shares = shares.rename('share')
        shares = shares.to_frame()
        
        data_share = pd.concat((shares, hist), axis=0)
        data_share = data_share.sort_index()
        data_share['share'] = data_share['share'].bfill()
        data_share = data_share.dropna()
        data_share = data_share.loc[start_date:end_date]

        if len(data_share) > 0:
        
            parent = Path(path)
            parent.mkdir(parents=True, exist_ok = True)
            data_share.to_csv(parent / f"{tick}.csv")
        
    except Exception as e:
        # logger.exception(e)
        pass

def download(start_date='2018-06-01', end_date='2023-03-31', folder='./data/mkt'):
    symbols = pd.read_csv('./data/us_symbols.csv')
    args_list = [(folder, start_date, end_date, tick) for tick in symbols.ticker]
    with ProcessPoolExecutor(6) as executor:
        _ = list(tqdm(executor.map(get_ticker_mkt_data, args_list), total=len(args_list)))        
    
    
if __name__ == '__main__':
    fire.Fire(download)