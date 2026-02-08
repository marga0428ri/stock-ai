import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import numpy as np

TICKERS = ["AAPL", "NVDA", "MSFT", "TSLA", "GOOGL", "AMZN"]

# --- 1. データ取得 ---
def get_data(ticker, start="2010-01-01"):
    try:
        df = yf.download(ticker, start=start, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

# --- 2. テクニカル指標の計算（ここが強化ポイント！） ---
def add_technical_indicators(df):
    df = df.copy()
    
    # 移動平均線
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    
    # RSI (買われすぎ・売られすぎセンサー)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD (トレンド転換センサー)
    exp12 = df["Close"].ewm(span=12, adjust=False).mean()
    exp26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp12 - exp26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # ボリンジャーバンド (異常値センサー)
    sma20 = df["Close"].rolling(window=20).mean()
    std20 = df["Close"].rolling(window=20).std()
    df["Upper_Band"] = sma20 + (std20 * 2)
    df["Lower_Band"] = sma20 - (std20 * 2)
    
    # 特徴量: 終値が各指標とどうなっているか
    df["RSI_Val"] = df["RSI"]
    df["MACD_Diff"] = df["MACD"] - df["Signal"] # プラスなら上昇トレンド
    df["Dist_Upper"] = (df["Upper_Band"] - df["Close"]) / df["Close"] # バンドまでの距離
    
    return df

# --- 3. 市場全体の分析 ---
def analyze_market_context():
    df = get_data("SPY")
    if df.empty: return "Unknown", "⚪"
    
    df = add_technical_indicators(df)
    latest = df.iloc[-1]
    
    # RSIで過熱感を判定
    if latest["RSI"] > 70:
        return "Overbought (Risk High)", "🔥⚠️"
    elif latest["RSI"] < 30:
        return "Oversold (Bounce Likely)", "💧🔄"
    
    # MACDでトレンド判定
    if latest["MACD_Diff"] > 0:
        return "Bull Trend (Positive)", "🐂✅"
    else:
        return "Bear Trend (Negative)", "🐻⚠️"

# --- 4. 個別株の予測 ---
def predict_stock(ticker, market_status):
    df = get_data(ticker)
    if df.empty or len(df) < 60: return None

    df = add_technical_indicators(df)
    
    # 5日後予測（1%以上上がるか？）
    prediction_days = 5
    future_return = (df["Close"].shift(-prediction_days) - df["Close"]) / df["Close"]
    df["Target"] = (future_return > 0.01).astype(int)
    
    df.dropna(inplace=True)

    # 学習に使う特徴量を増やす
    features = ["RSI_Val", "MACD_Diff", "Dist_Upper"]
    X = df[features]
    y = df["Target"]
    
    X_train = X.iloc[:-5]
    y_train = y.iloc[:-5]
    X_latest = X.iloc[-1:]

    # 市場状況に合わせてAIの判断基準を変える
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    prob = model.predict_proba(X_latest)[0]
    score = prob[1] * 100 # 上昇確率
    
    # 判定ロジックを微調整（40-60%は迷い中とする）
    if score >= 60:
        trend = "STRONG UP 🚀"
    elif score >= 50:
        trend = "WEAK UP ↗️"
    elif score >= 40:
        trend = "NEUTRAL ➡️"
    else:
        trend = "DOWN ↘️"
        
    return {
        "ticker": ticker,
        "price": df["Close"].iloc[-1],
        "trend": trend,
        "score": score,
        "rsi": df["RSI_Val"].iloc[-1] # RSIも表示してあげる
    }

# --- 5. レポート作成 ---
def update_readme(market_info, results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_text, status_icon = market_info
    
    rows = ""
    for res in results:
        # RSIも表示に追加
        rows += f"| {res['ticker']} | ${res['price']:.2f} | **{res['trend']}** | {res['score']:.1f}% | {res['rsi']:.1f} |\n"

    content = f"""# 🧠 AI Investment Strategy Report (Technical Ver.)
    
## 🌍 Market Context
**Status:** {status_icon} **{status_text}**
(Analyzed via RSI & MACD of S&P 500)

---

## 🎯 Individual Stock Predictions (5-Day Horizon)
*Updated: {now} (UTC)*

| Ticker | Price | Prediction | Probability (Up) | RSI (Heat) |
| :--- | :--- | :--- | :--- | :--- |
{rows}

- **RSI > 70:** Overbought (High risk of drop)
- **RSI < 30:** Oversold (Chance of bounce)
- **Probability:** >60% is a strong signal.

---
*Powered by GitHub Actions*
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)

def main():
    print("--- Analyzing Market ---")
    market_status = analyze_market_context()
    
    results = []
    print("--- Predicting Stocks ---")
    for ticker in TICKERS:
        try:
            res = predict_stock(ticker, market_status)
            if res: results.append(res)
        except Exception as e:
            print(f"Skip {ticker}: {e}")
            
    update_readme(market_status, results)
    print("Done!")

if __name__ == "__main__":
    main()
