
import yfinance as yf
import pandas as pd

def get_data(ticker="AAPL", start="2015-01-01"):
    # データを取得
    df = yf.download(ticker, start=start, progress=False)
    
    # 【エラー対策】データの表形式が複雑な場合、シンプルに直す
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

if __name__ == "__main__":
    get_data()
