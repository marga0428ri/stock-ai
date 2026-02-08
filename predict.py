import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import feedparser
from textblob import TextBlob
import numpy as np

# --- 🎯 銘柄リスト ---
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

# --- 1. ニュース（現在の事件）の仮採点 ---
KEYWORDS_SCORE = {
    # 影響度が大きい単語のみに絞り、点数を控えめにする（補正前）
    "record": 5, "beat": 3, "surge": 3, "partnership": 3, "acquisition": 3,
    "lawsuit": -5, "miss": -3, "plunge": -3, "scandal": -5, "regulatory": -3
}

def get_base_news_score(query):
    safe_query = query.replace(" ", "+")
    rss_url = f"https://news.google.com/rss/search?q={safe_query}+when:1d&hl=en-US&gl=US&ceid=US:en"
    total = 0
    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:5]:
            title = entry.title.lower()
            # 単語マッチング
            for word, score in KEYWORDS_SCORE.items():
                if word in title: total += score
            # 感情分析（補助）
            blob = TextBlob(title)
            total += blob.sentiment.polarity * 2
    except: pass
    return total # これが「補正前」の点数

# --- 2. データ取得 ---
def get_data(ticker, start="2012-01-01"):
    try:
        df = yf.download(ticker, start=start, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

# --- 3. ★歴史的感応度（Impact Factor）の計算 ---
def calculate_historical_sensitivity(df):
    """
    【因果関係の学習】
    過去に「大きな動き（事件）」があった時、その後の株価はどう反応したか？
    """
    df = df.copy()
    
    # 1. 過去の「事件日」を定義（3%以上動いた日）
    df["Daily_Return"] = df["Close"].pct_change()
    df["Is_Shock"] = df["Daily_Return"].abs() > 0.03 
    
    # 2. その事件の「5日後の結果」を見る
    df["Next_5d_Return"] = df["Close"].shift(-5).pct_change(periods=5)
    
    # 3. 衝撃（原因）と結果（価格）の相関を調べる
    # Shock_Sensitivity: 事件があった方向にさらに伸びるか、逆に行くか
    # 正の値 = 順張り（ニュースに従いやすい）
    # 負の値 = 逆張り（ニュースを無視して戻しやすい）
    shock_data = df[df["Is_Shock"] == True]
    
    if len(shock_data) < 10:
        return 1.0 # データ不足なら標準（1.0倍）
        
    sensitivity = shock_data["Daily_Return"].corr(shock_data["Next_5d_Return"])
    
    # 係数がNaNになる場合（動きがない場合）の対策
    if np.isnan(sensitivity): sensitivity = 0.5

    # 係数を使いやすい形に正規化 (0.5倍 〜 2.0倍の範囲に収める)
    # これにより「過剰な点数」を防ぐ
    impact_factor = 1.0 + sensitivity
    impact_factor = max(0.5, min(2.0, impact_factor))
    
    return impact_factor

# --- 4. 予測ロジック（因果関係の統合） ---
def predict_stock(stock_info):
    ticker = stock_info["ticker"]
    df = get_data(ticker)
    if df.empty or len(df) < 300: return None

    # A. 過去の傾向（感応度）を学習
    impact_factor = calculate_historical_sensitivity(df)
    
    # B. 現在のニュース点数を取得
    base_news_score = get_base_news_score(stock_info["query"])
    
    # C. 点数の補正（ここが重要！）
    # 「ただの+5点」ではなく、「この株は事件に敏感だから +5 * 1.5 = +7.5点」とする
    adjusted_event_points = base_news_score * impact_factor * 5 # 5はスケール調整
    
    # 上限下限クリップ（暴走防止）
    adjusted_event_points = max(-30, min(30, adjusted_event_points))

    # D. テクニカル分析（トレンド確認）
    df["RSI"] = 100 - (100 / (1 + df["Close"].diff().where(df["Close"].diff() > 0, 0).rolling(14).mean() / (-df["Close"].diff().where(df["Close"].diff() < 0, 0).rolling(14).mean())))
    last_rsi = df["RSI"].iloc[-1]
    
    # E. 最終スコア算出
    # 基準点50 + 事件点(補正済み) + テクニカル微調整
    tech_bias = 0
    if last_rsi > 70: tech_bias = -5 # 買われすぎなら少し引く
    elif last_rsi < 30: tech_bias = 5 # 売られすぎなら少し足す
    
    final_score = 50 + adjusted_event_points + tech_bias
    final_score = max(0, min(100, final_score))
    
    # 評価
    if final_score >= 60: grade = "S 🚀"
    elif final_score >= 53: grade = "A ↗️"
    elif final_score >= 47: grade = "B ➡️"
    elif final_score >= 40: grade = "C ↘️"
    else: grade = "D 💀"

    # ニュースアイコン
    news_icon = "⚪"
    if adjusted_event_points > 5: news_icon = "☀️"
    if adjusted_event_points < -5: news_icon = "☁️"
    if adjusted_event_points > 15: news_icon = "🔥"
    if adjusted_event_points < -15: news_icon = "⚡"

    return {
        "name": stock_info["name"],
        "price": df["Close"].iloc[-1],
        "currency": stock_info["currency"],
        "grade": grade,
        "score": final_score,
        "event_pts": adjusted_event_points,
        "sensitivity": impact_factor, # これを表示して傾向を確認
        "news_icon": news_icon
    }

# --- 5. レポート作成 ---
def update_readme(res_us, res_jp):
    now = datetime.now().strftime("%Y-%m-%d %H:%M (UTC)")
    res_us.sort(key=lambda x: x["score"], reverse=True)
    res_jp.sort(key=lambda x: x["score"], reverse=True)
    
    def make_table(results):
        rows = ""
        for r in results:
            # Impact Factor (感応度) も表示
            rows += f"| {r['name']} | {r['currency']}{r['price']:,.0f} | **{r['grade']}** | {r['score']:.0f} | {r['event_pts']:.1f} {r['news_icon']} | x{r['sensitivity']:.2f} |\n"
        return rows

    content = f"""# 🧠 AI Strategy Report (History-Adjusted)
    
## ⚖️ How "Event Points" work now?
The AI doesn't just read news. It checks **History**.
It calculates a **Sensitivity Factor (x1.0)** for each stock.

* **Equation:** `News Keywords` × `Sensitivity Factor` = **True Impact**
* **Sensitivity > 1.0:** This stock tends to **overreact** to news (High Risk).
* **Sensitivity < 1.0:** This stock is **resilient** (Low Risk).
* **Event Pts:** The final calculated impact of today's news.

---

## 🇺🇸 US & Global Stocks
| Stock | Price | Rating | Total | Event Pts | Sensitivity |
| :--- | :--- | :--- | :--- | :--- | :--- |
{make_table(res_us)}

## 🇯🇵 Japan Stocks
| Stock | Price | Rating | Total | Event Pts | Sensitivity |
| :--- | :--- | :--- | :--- | :--- | :--- |
{make_table(res_jp)}

---
*Updated: {now}*
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)

def main():
    print("--- Predicting US Stocks ---")
    res_us = [r for s in STOCKS_US if (r := predict_stock(s))]
    print("--- Predicting Japan Stocks ---")
    res_jp = [r for s in STOCKS_JP if (r := predict_stock(s))]
            
    update_readme(res_us, res_jp)
    print("Done!")

if __name__ == "__main__":
    main()

