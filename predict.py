import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import feedparser
from textblob import TextBlob
import numpy as np
import time

# --- 🎯 監視対象と「3つの視点」 ---
# 情報収集を倍増させるため、各銘柄に複数の検索クエリを設定
STOCKS = [
    # 🇺🇸 米国株
    {
        "ticker": "NVDA", "name": "NVIDIA", "currency": "$",
        "queries": ["NVIDIA stock news", "NVIDIA earnings analysis", "AI chip market demand"]
    },
    {
        "ticker": "AAPL", "name": "Apple", "currency": "$",
        "queries": ["Apple stock news", "iPhone sales report", "tech sector trends"]
    },
    {
        "ticker": "MSFT", "name": "Microsoft", "currency": "$",
        "queries": ["Microsoft stock news", "Azure cloud growth", "software industry news"]
    },
    {
        "ticker": "TSLA", "name": "Tesla", "currency": "$",
        "queries": ["Tesla stock news", "EV market outlook", "Elon Musk news"]
    },
    {
        "ticker": "AMZN", "name": "Amazon", "currency": "$",
        "queries": ["Amazon stock news", "AWS revenue", "e-commerce trends"]
    },
    
    # 🇯🇵 日本株
    {
        "ticker": "7203.T", "name": "Toyota", "currency": "¥",
        "queries": ["Toyota Motor stock", "Toyota financial results", "auto industry Japan"]
    },
    {
        "ticker": "6758.T", "name": "Sony Group", "currency": "¥",
        "queries": ["Sony Group stock", "PlayStation sales", "image sensor market"]
    },
    {
        "ticker": "7974.T", "name": "Nintendo", "currency": "¥",
        "queries": ["Nintendo stock", "Switch console sales", "video game market"]
    },
    {
        "ticker": "8035.T", "name": "Tokyo Electron", "currency": "¥",
        "queries": ["Tokyo Electron stock", "semiconductor equipment market", "chip industry news"]
    },
    {
        "ticker": "9983.T", "name": "Fast Retailing", "currency": "¥",
        "queries": ["Fast Retailing stock", "Uniqlo sales", "retail apparel trends"]
    }
]

# --- 1. Deep News Analysis (情報収集の倍増) ---
KEYWORDS_WEIGHT = {
    "record": 2.0, "surge": 1.5, "jump": 1.5, "beat": 1.5, "approval": 2.0,
    "buyback": 1.2, "dividend": 1.2, "acquisition": 1.5, "partnership": 1.2,
    "plunge": -1.5, "miss": -1.5, "drop": -1.2, "fail": -1.5, "lawsuit": -2.0,
    "scandal": -2.5, "cut": -1.2, "downgrade": -1.5, "inflation": -1.0
}

def analyze_deep_news(queries):
    """
    複数のクエリを使ってニュースを深掘りし、
    総合的な「センチメントスコア (-1.0 ~ 1.0)」を算出する
    """
    total_score = 0
    article_count = 0
    seen_titles = set() # 重複記事を排除

    for query in queries:
        safe_query = query.replace(" ", "+")
        rss_url = f"https://news.google.com/rss/search?q={safe_query}+when:1d&hl=en-US&gl=US&ceid=US:en"
        
        try:
            feed = feedparser.parse(rss_url)
            # 各クエリから上位5件を取得（合計最大15件）
            for entry in feed.entries[:5]:
                title = entry.title
                if title in seen_titles: continue
                seen_titles.add(title)

                # A. 感情分析
                blob = TextBlob(title)
                polarity = blob.sentiment.polarity
                
                # B. キーワード重み付け
                weight = 1.0
                title_lower = title.lower()
                for word, w_val in KEYWORDS_WEIGHT.items():
                    if word in title_lower:
                        weight = w_val # 強い言葉があれば重みを上書き
                        break # 最も強い言葉を優先
                
                # 重み付きスコアを加算
                total_score += polarity * abs(weight) * (1 if weight > 0 else -1)
                article_count += 1
                
        except Exception:
            continue
            
    if article_count == 0: return 0.0, 0
    
    # 平均スコアを算出 (-1.0 〜 1.0 に正規化)
    avg_score = total_score / article_count
    # 少し値を強調する（ニュースの影響を反映しやすくするため）
    final_sentiment = max(-1.0, min(1.0, avg_score * 1.5))
    
    return final_sentiment, article_count

# --- 2. 過去データとボラティリティの取得 ---
def get_market_data(ticker):
    try:
        # 過去1年分のデータ
        df = yf.download(ticker, period="1y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

# --- 3. 利益予想ロジック (Expected Return Calculation) ---
def calculate_expected_profit(df, sentiment_score):
    """
    【ここが核心】
    その株が持つ「変動エネルギー(Volatility)」に「ニュースの勢い」を掛け合わせ、
    明日の具体的な予想利益(%)を計算する。
    """
    # 1. その株の「1日の平均変動幅」を計算（ボラティリティ）
    # 最近の動きを重視するため、直近20日の標準偏差を使う
    daily_volatility = df["Close"].pct_change().rolling(20).std().iloc[-1]
    
    # データが取れない場合の安全策
    if np.isnan(daily_volatility): daily_volatility = 0.015 # 1.5%と仮定

    # 2. ベースのトレンド（最近上がってるか下がってるか）
    # 5日移動平均と20日移動平均の乖離率
    sma5 = df["Close"].rolling(5).mean().iloc[-1]
    sma20 = df["Close"].rolling(20).mean().iloc[-1]
    trend_strength = (sma5 - sma20) / sma20
    
    # 3. 予想変動率の計算式
    # 予想% = (トレンド由来の変動) + (ニュース由来の衝撃)
    # ニュースがない(0)なら、トレンドに従う。ニュースが強ければ、それを大きく反映。
    
    # ニュースの影響力をボラティリティの何倍にするか（感応度係数）
    impact_factor = 2.0 
    
    expected_change_pct = (trend_strength * 0.3) + (sentiment_score * daily_volatility * impact_factor)
    
    # 現実的な範囲に収める（1日で±15%以上動く予想は異常値としてカット）
    expected_change_pct = max(-0.15, min(0.15, expected_change_pct))
    
    return expected_change_pct * 100 # %表記にする

# --- 4. 総合分析実行 ---
def analyze_stock(stock_info):
    ticker = stock_info["ticker"]
    
    # A. データをじっくり取得
    df = get_market_data(ticker)
    if df.empty or len(df) < 20: return None
    
    # B. ニュースを深く読む（3倍の情報量）
    sentiment, art_count = analyze_deep_news(stock_info["queries"])
    
    # C. 利益予想を計算
    exp_profit = calculate_expected_profit(df, sentiment)
    
    # D. アクション判定
    # 予想利益がプラスならBUY、マイナスならSELL、微小ならWAIT
    action = "WAIT ⚪"
    if exp_profit > 1.0: action = "BUY 🔵" # 1%以上の利益が見込めるならGO
    if exp_profit > 3.0: action = "STRONG BUY 🚀" # 3%以上なら激熱
    if exp_profit < -1.0: action = "SELL 🔴"
    if exp_profit < -3.0: action = "STRONG SELL ⚡"
    
    return {
        "name": stock_info["name"],
        "price": df["Close"].iloc[-1],
        "currency": stock_info["currency"],
        "action": action,
        "exp_profit": exp_profit, # 予想利益 (%)
        "sentiment": sentiment,
        "articles": art_count,
        "volatility": df["Close"].pct_change().rolling(20).std().iloc[-1] * 100
    }

# --- 5. レポート作成（予想利益欄を追加） ---
def update_readme(results_us, results_jp):
    now = datetime.now().strftime("%Y-%m-%d %H:%M (UTC)")
    
    # 予想利益が高い順に並べ替え（一番儲かりそうな株を上に）
    results_us.sort(key=lambda x: x["exp_profit"], reverse=True)
    results_jp.sort(key=lambda x: x["exp_profit"], reverse=True)
    
    def make_table(results):
        rows = ""
        for r in results:
            # 利益予想の表示色づけ
            prof_str = f"{r['exp_profit']:+.2f}%"
            if r['exp_profit'] > 0: prof_str = f"**{prof_str}** 📈"
            elif r['exp_profit'] < 0: prof_str = f"{prof_str} 📉"
            
            # センチメントアイコン
            sent_icon = "⚪"
            if r['sentiment'] > 0.3: sent_icon = "☀️"
            if r['sentiment'] < -0.3: sent_icon = "☁️"
            
            rows += f"| {r['action']} | {r['name']} | {r['currency']}{r['price']:,.0f} | {prof_str} | {sent_icon} ({r['articles']} news) |\n"
        return rows

    content = f"""# 🔭 Deep Impact Stock Forecast
    
## 📊 Project Goal
To calculate the **Exact Expected Profit (%)** for tomorrow by analyzing:
1.  **Multi-Angle News:** Analyzing company, financial, and sector news.
2.  **Volatility Energy:** Calculating how much the stock *can* move.

* **Updates:** 3 times daily (Every 8 hours).
* **Focus:** Quality of Information > Frequency of Updates.

---

## 🇺🇸 US Stocks: Expected Profit
| Action | Stock | Price | **Exp. Profit (Target)** | News Power |
| :--- | :--- | :--- | :--- | :--- |
{make_table(results_us)}

## 🇯🇵 Japan Stocks: Expected Profit
| Action | Stock | Price | **Exp. Profit (Target)** | News Power |
| :--- | :--- | :--- | :--- | :--- |
{make_table(results_jp)}

### 💡 How to read "Exp. Profit"
* **+2.5% 📈**: AI predicts the price will rise by 2.5% tomorrow based on news impact.
* **-1.2% 📉**: Negative news pressure suggests a drop.
* **Logic**: `Volatility` × `News Sentiment Score` = `Expected Move`

---
*Updated: {now}*
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)

def main():
    # 日本株と米国株を分けてリスト化
    stocks_us = [s for s in STOCKS if s['currency'] == "$"]
    stocks_jp = [s for s in STOCKS if s['currency'] == "¥"]
    
    print("--- Analyzing US Stocks ---")
    res_us = [r for s in stocks_us if (r := analyze_stock(s))]
    
    print("--- Analyzing Japan Stocks ---")
    res_jp = [r for s in stocks_jp if (r := analyze_stock(s))]
            
    update_readme(res_us, res_jp)
    print("Done!")

if __name__ == "__main__":
    main()

