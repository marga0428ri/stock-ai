import yfinance as yf

def fetch(stock_code="AAPL"):
    df = yf.download(
        stock_code,
        start="2015-01-01",
        interval="1d",
        auto_adjust=True
    )
    return df

if __name__ == "__main__":
    data = fetch("AAPL")
    data.to_csv("AAPL_daily.csv")
    print("Saved AAPL_daily.csv")
