import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- 🎯 設定：調べたい株のリスト ---
# ここを好きな銘柄に変えれば、何でも分析できます
TICKERS = ["AAPL", "NVDA", "MSFT", "TSLA", "GOOGL", "AMZN"]

# --- 1. データ取得（全体・個別） ---
def get_data(ticker, start="2010-01-01"):
    try:
        df = yf.download(ticker, start=start, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

# --- 2. 世界・文脈AI（上位レイヤー） ---
def analyze_market_context():
    """
    S&P500 (SPY) を分析して、今の市場が「安全」か「危険」かを判定する
    """
    df = get_data("SPY") # S&P500 ETF
    if df.empty: return "Unknown", "⚪"

    # 200日移動平均線（長期トレンドの王様）
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    latest = df.iloc[-1]
    
    # 短期的なショック判定（ボラティリティ）
    volatility = df["Close"].pct_change().rolling(window=20).std().iloc[-1]
    
    # 判定ロジック
    if latest["Close"] < latest["SMA_200"]:
        # 長期下落トレンド（リーマンショックやコロナ初期のような状態）
        return "Bear Market (Danger)", "🐻⚠️"
    elif volatility > 0.02:
        # トレンドは上だが、値動きが激しすぎる（不安定）
        return "Volatile (Caution)", "🌊⚠️"
    else:
        # 安定上昇
        return "Bull Market (Safe)", "🐂✅"

# --- 3. 個別株AI（下位レイヤー） ---
def predict_stock(ticker, market_status):
    df = get_data(ticker)
    if df.empty: return None

    # 特徴量エンジニアリング
    df["Return"] = df["Close"].pct_change()
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Volatility"] = df["Close"].pct_change().rolling(window=5).std()
    
    # 5日後予測（1%以上上がるか？）
    prediction_days = 5
    future_return = (df["Close"].shift(-prediction_days) - df["Close"]) / df["Close"]
    df["Target"] = (future_return > 0.01).astype(int)
    
    df.dropna(inplace=True)

    # 学習データ作成
    features = ["Return", "Volatility"]
    X = df[features]
    y = df["Target"]
    
    X_train = X.iloc[:-5]
    y_train = y.iloc[:-5]
    X_latest = X.iloc[-1:]

    # 市場が悪ければ、AIを慎重にする（木の深さを浅くする等）
    depth = 3 if "Danger" in market_status else 5
    
    model = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    
    prediction = model.predict(X_latest)[0]
    prob = model.predict_proba(X_latest)[0]
    
    # 結果整形
    score = prob[1] * 100 # 上昇確率
    trend = "UP 🚀" if prediction == 1 else "DOWN 📉"
    
    return {
        "ticker": ticker,
        "price": df["Close"].iloc[-1],
        "trend": trend,
        "score": score
    }

# --- 4. レポート作成機能 ---
def update_readme(market_info, results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_text, status_icon = market_info
    
    # テーブルの行を作成
    rows = ""
    for res in results:
        rows += f"| {res['ticker']} | ${res['price']:.2f} | **{res['trend']}** | {res['score']:.1f}% |\n"

    content = f"""# 🧠 AI Investment Strategy Report
    
## 🌍 Market Context (World AI)
**Status:** {status_icon} **{status_text}**
- The AI analyzes the S&P 500 trend to determine global risk.
- If "Danger", individual predictions become more conservative.

---

## 🎯 Individual Stock Predictions (5-Day Horizon)
*Updated: {now} (UTC)*

| Ticker | Price | Prediction | Confidence |
| :--- | :--- | :--- | :--- |
{rows}

---
*Powered by GitHub Actions & Python*
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)

# --- メイン処理 ---
def main():
    print("--- 1. Analyzing World Context ---")
    market_status, market_icon = analyze_market_context()
    print(f"Market Status: {market_status}")

    results = []
    print("--- 2. Predicting Individual Stocks ---")
    for ticker in TICKERS:
        print(f"Processing {ticker}...")
        res = predict_stock(ticker, market_status)
        if res:
            results.append(res)
    
    print("--- 3. Updating Report ---")
    update_readme((market_status, market_icon), results)
    print("Done!")

if __name__ == "__main__":
    main()
