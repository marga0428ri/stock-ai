import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- 🎯 銘柄リスト（2つのグループに分離） ---
# 🇺🇸 米国・世界株
STOCKS_US = [
    {"ticker": "NVDA", "name": "NVIDIA", "currency": "$"},
    {"ticker": "AAPL", "name": "Apple", "currency": "$"},
    {"ticker": "MSFT", "name": "Microsoft", "currency": "$"},
    {"ticker": "TSLA", "name": "Tesla", "currency": "$"},
    {"ticker": "AMZN", "name": "Amazon", "currency": "$"},
    {"ticker": "GOOGL", "name": "Google", "currency": "$"},
    {"ticker": "LLY", "name": "Eli Lilly", "currency": "$"}
]

# 🇯🇵 日本株 (Code + .T)
STOCKS_JP = [
    {"ticker": "7203.T", "name": "Toyota", "currency": "¥"},
    {"ticker": "6758.T", "name": "Sony Group", "currency": "¥"},
    {"ticker": "7974.T", "name": "Nintendo", "currency": "¥"},
    {"ticker": "9984.T", "name": "SoftBank G", "currency": "¥"},
    {"ticker": "8035.T", "name": "Tokyo Electron", "currency": "¥"},
    {"ticker": "6861.T", "name": "Keyence", "currency": "¥"},
    {"ticker": "9983.T", "name": "Fast Retailing", "currency": "¥"}
]

# --- 1. データ取得 ---
def get_data(ticker, start="2015-01-01"):
    try:
        df = yf.download(ticker, start=start, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

# --- 2. テクニカル分析 ---
def add_indicators(df):
    df = df.copy()
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    # MACD
    exp12 = df["Close"].ewm(span=12).mean()
    exp26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = exp12 - exp26
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    return df

# --- 3. 市場全体の分析 (S&P500) ---
def analyze_market():
    df = get_data("SPY")
    if df.empty: return "Unknown", "⚪"
    df = add_indicators(df)
    rsi = df["RSI"].iloc[-1]
    if rsi > 70: return "Overheated (Caution)", "🔥⚠️"
    if rsi < 30: return "Bargain Zone (Buy)", "💎✅"
    return "Neutral / Stable", "⚖️"

# --- 4. 予測ロジック ---
def predict_stock(stock_info):
    ticker = stock_info["ticker"]
    df = get_data(ticker)
    if df.empty or len(df) < 60: return None

    df = add_indicators(df)
    
    # 5日後予測
    future_return = (df["Close"].shift(-5) - df["Close"]) / df["Close"]
    df["Target"] = (future_return > 0.01).astype(int)
    df.dropna(inplace=True)

    # 学習
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(df[["RSI", "MACD"]].iloc[:-5], df["Target"].iloc[:-5])
    
    # 予測
    score = model.predict_proba(df[["RSI", "MACD"]].iloc[-1:])[0][1] * 100
    
    if score >= 60: grade = "S 🚀"
    elif score >= 50: grade = "A ↗️"
    elif score >= 40: grade = "B ➡️"
    else: grade = "C ↘️"
    
    return {
        "name": stock_info["name"],
        "price": df["Close"].iloc[-1],
        "currency": stock_info["currency"],
        "grade": grade,
        "score": score,
        "rsi": df["RSI"].iloc[-1]
    }

# --- 5. レポート作成（2つの表を作成） ---
def update_readme(market_status, res_us, res_jp):
    now = datetime.now().strftime("%Y-%m-%d %H:%M (UTC)")
    
    # スコア順に並べ替え
    res_us.sort(key=lambda x: x["score"], reverse=True)
    res_jp.sort(key=lambda x: x["score"], reverse=True)
    
    def make_table(results):
        rows = ""
        for r in results:
            rows += f"| {r['name']} | {r['currency']}{r['price']:,.0f} | **{r['grade']}** | {r['score']:.1f}% | {r['rsi']:.1f} |\n"
        return rows

    content = f"""# 🧠 AI Strategy Report (Dual Region)
    
## 🌍 Global Market Context
**Status:** {market_status[1]} **{market_status[0]}**

---

## 🇺🇸 US & Global Growth Stocks
| Stock | Price | Rating | Conf. | RSI |
| :--- | :--- | :--- | :--- | :--- |
{make_table(res_us)}

## 🇯🇵 Japan Leading Stocks
| Stock | Price | Rating | Conf. | RSI |
| :--- | :--- | :--- | :--- | :--- |
{make_table(res_jp)}

---
*Updated: {now}*
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)

def main():
    print("--- Market Check ---")
    status = analyze_market()
    
    res_us = []
    print("--- Predicting US Stocks ---")
    for s in STOCKS_US:
        r = predict_stock(s)
        if r: res_us.append(r)
        
    res_jp = []
    print("--- Predicting Japan Stocks ---")
    for s in STOCKS_JP:
        r = predict_stock(s)
        if r: res_jp.append(r)
            
    update_readme(status, res_us, res_jp)
    print("Done!")

if __name__ == "__main__":
    main()
