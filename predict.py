import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import feedparser
from textblob import TextBlob
import numpy as np

# --- 🎯 銘柄リスト（ニュース検索ワード付き） ---
STOCKS_US = [
    {"ticker": "NVDA", "name": "NVIDIA", "currency": "$", "query": "NVIDIA stock"},
    {"ticker": "AAPL", "name": "Apple", "currency": "$", "query": "Apple stock"},
    {"ticker": "MSFT", "name": "Microsoft", "currency": "$", "query": "Microsoft stock"},
    {"ticker": "TSLA", "name": "Tesla", "currency": "$", "query": "Tesla stock"},
    {"ticker": "AMZN", "name": "Amazon", "currency": "$", "query": "Amazon stock"},
    {"ticker": "GOOGL", "name": "Google", "currency": "$", "query": "Google stock"},
    {"ticker": "LLY", "name": "Eli Lilly", "currency": "$", "query": "Eli Lilly stock"}
]

STOCKS_JP = [
    {"ticker": "7203.T", "name": "Toyota", "currency": "¥", "query": "Toyota stock"},
    {"ticker": "6758.T", "name": "Sony Group", "currency": "¥", "query": "Sony Group stock"},
    {"ticker": "7974.T", "name": "Nintendo", "currency": "¥", "query": "Nintendo stock"},
    {"ticker": "9984.T", "name": "SoftBank G", "currency": "¥", "query": "SoftBank Group stock"},
    {"ticker": "8035.T", "name": "Tokyo Electron", "currency": "¥", "query": "Tokyo Electron stock"},
    {"ticker": "6861.T", "name": "Keyence", "currency": "¥", "query": "Keyence stock"},
    {"ticker": "9983.T", "name": "Fast Retailing", "currency": "¥", "query": "Fast Retailing stock"}
]

# --- 1. ニュース分析機能 (ここを修正しました！) ---
def get_news_sentiment(query):
    """
    Googleニュースから感情分析を行う
    """
    # ★修正ポイント：URL内のスペースを「+」に変換してエラーを防ぐ
    safe_query = query.replace(" ", "+")
    
    rss_url = f"https://news.google.com/rss/search?q={safe_query}+when:1d&hl=en-US&gl=US&ceid=US:en"
    
    try:
        feed = feedparser.parse(rss_url)
        sentiments = []
        for entry in feed.entries[:5]:
            analysis = TextBlob(entry.title)
            sentiments.append(analysis.sentiment.polarity)
        
        if not sentiments:
            return 0.0
            
        return sum(sentiments) / len(sentiments)
    except Exception as e:
        print(f"News Error ({query}): {e}")
        return 0.0

# --- 2. データ取得 ---
def get_data(ticker, start="2015-01-01"):
    try:
        df = yf.download(ticker, start=start, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

# --- 3. テクニカル分析 ---
def add_indicators(df):
    df = df.copy()
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    exp12 = df["Close"].ewm(span=12).mean()
    exp26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = exp12 - exp26
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    return df

# --- 4. 市場全体の分析 ---
def analyze_market():
    df = get_data("SPY")
    if df.empty: return "Unknown", "⚪"
    df = add_indicators(df)
    rsi = df["RSI"].iloc[-1]
    
    # ニュース分析（ここもスペース対策済み）
    news_score = get_news_sentiment("Stock Market US")
    
    status = "Neutral"
    icon = "⚖️"
    
    if rsi > 70: 
        status = "Overheated"
        icon = "🔥"
    elif rsi < 30: 
        status = "Bargain"
        icon = "💎"
        
    if news_score < -0.2:
        status += " (News: Bad ☁️)"
        icon = "🐻⚠️"
    elif news_score > 0.2:
        status += " (News: Good ☀️)"
        icon = "🐂✅"
        
    return status, icon

# --- 5. 予測ロジック ---
def predict_stock(stock_info):
    ticker = stock_info["ticker"]
    df = get_data(ticker)
    if df.empty or len(df) < 60: return None

    df = add_indicators(df)
    
    future_return = (df["Close"].shift(-5) - df["Close"]) / df["Close"]
    df["Target"] = (future_return > 0.01).astype(int)
    df.dropna(inplace=True)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(df[["RSI", "MACD"]].iloc[:-5], df["Target"].iloc[:-5])
    
    tech_score = model.predict_proba(df[["RSI", "MACD"]].iloc[-1:])[0][1] * 100
    
    # ニュース分析実行
    news_score = get_news_sentiment(stock_info["query"])
    news_adjustment = news_score * 10 
    
    final_score = max(0, min(100, tech_score + news_adjustment))
    
    if final_score >= 60: grade = "S 🚀"
    elif final_score >= 50: grade = "A ↗️"
    elif final_score >= 40: grade = "B ➡️"
    else: grade = "C ↘️"
    
    news_icon = "⚪"
    if news_score > 0.1: news_icon = "☀️"
    if news_score < -0.1: news_icon = "☁️"
    
    return {
        "name": stock_info["name"],
        "price": df["Close"].iloc[-1],
        "currency": stock_info["currency"],
        "grade": grade,
        "score": final_score,
        "rsi": df["RSI"].iloc[-1],
        "news_icon": news_icon
    }

# --- 6. レポート作成 ---
def update_readme(market_status, res_us, res_jp):
    now = datetime.now().strftime("%Y-%m-%d %H:%M (UTC)")
    
    res_us.sort(key=lambda x: x["score"], reverse=True)
    res_jp.sort(key=lambda x: x["score"], reverse=True)
    
    def make_table(results):
        rows = ""
        for r in results:
            rows += f"| {r['name']} | {r['currency']}{r['price']:,.0f} | **{r['grade']}** | {r['score']:.1f}% | {r['rsi']:.1f} | {r['news_icon']} |\n"
        return rows

    content = f"""# 🧠 AI Strategy Report (News Integrated)
    
## 🌍 Global Market Context
**Status:** {market_status[1]} **{market_status[0]}**

---

## 🇺🇸 US & Global Growth Stocks
| Stock | Price | Rating | Conf. | RSI | News |
| :--- | :--- | :--- | :--- | :--- | :--- |
{make_table(res_us)}

## 🇯🇵 Japan Leading Stocks
| Stock | Price | Rating | Conf. | RSI | News |
| :--- | :--- | :--- | :--- | :--- | :--- |
{make_table(res_jp)}

### 💡 Legend
- **News:** ☀️=Good, ☁️=Bad, ⚪=Neutral
- **Conf:** Tech Score ± News Sentiment
- **Schedule:** Updates every 6 hours

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
        try:
            r = predict_stock(s)
            if r: res_us.append(r)
        except Exception as e:
            print(f"Error {s['name']}: {e}")
        
    res_jp = []
    print("--- Predicting Japan Stocks ---")
    for s in STOCKS_JP:
        try:
            r = predict_stock(s)
            if r: res_jp.append(r)
        except Exception as e:
            print(f"Error {s['name']}: {e}")
            
    update_readme(status, res_us, res_jp)
    print("Done!")

if __name__ == "__main__":
    main()
