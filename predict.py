import pandas as pd
import yfinance as yf
from datetime import datetime
import feedparser
from textblob import TextBlob
import numpy as np
import time

# ==========================================
# 👇 スプレッドシートのURL (output=csv)
# ==========================================
SHEET_URL = "https://docs.google.com/spreadsheets/d/10MtVu1vgAq0qJ0-O0lxHMy29_EZ7uG3-cSQlcXd0FUY/edit?usp=drivesdk"

# --- 🎯 市場全体の監視リスト (Market Watch) ---
STOCKS_MARKET = [
    {"ticker": "NVDA", "name": "NVIDIA", "currency": "$", "queries": ["NVIDIA stock", "AI chip demand", "Semiconductor trends"]},
    {"ticker": "MSFT", "name": "Microsoft", "currency": "$", "queries": ["Microsoft stock", "Azure cloud", "AI copilot"]},
    {"ticker": "TSLA", "name": "Tesla", "currency": "$", "queries": ["Tesla stock", "EV market", "Elon Musk news"]},
    {"ticker": "8035.T", "name": "Tokyo Electron", "currency": "¥", "queries": ["Tokyo Electron", "chip equipment", "semiconductor market"]},
    {"ticker": "9983.T", "name": "Fast Retailing", "currency": "¥", "queries": ["Fast Retailing", "Uniqlo sales", "apparel trends"]}
]

# --- 1. シート読み込み (マルチクエリ対応) ---
def load_portfolio_from_sheet():
    try:
        print("Loading portfolio from Google Sheets...")
        df = pd.read_csv(SHEET_URL)
        portfolio = []
        for index, row in df.iterrows():
            if pd.isna(row["Ticker"]): continue
            
            # Queryセルをカンマで区切ってリスト化する
            raw_query = str(row["Query"])
            queries = [q.strip() for q in raw_query.split(",")]
            
            item = {
                "ticker": str(row["Ticker"]).strip(),
                "name": str(row["Name"]),
                "buy_price": float(row["BuyPrice"]),
                "amount": int(row["Amount"]),
                "currency": str(row["Currency"]).strip(),
                "queries": queries
            }
            portfolio.append(item)
        print(f"Loaded {len(portfolio)} stocks.")
        return portfolio
    except Exception as e:
        print(f"Sheet Error: {e}")
        return []

# --- 2. ニュース分析 (Deep Analysis) ---
KEYWORDS_WEIGHT = {
    "record": 2.0, "surge": 1.5, "jump": 1.5, "beat": 1.5, "approval": 2.0,
    "buyback": 1.2, "dividend": 1.2, "acquisition": 1.5, "partnership": 1.2,
    "launch": 1.2, "breakthrough": 1.5,
    "plunge": -1.5, "miss": -1.5, "drop": -1.2, "fail": -1.5, "lawsuit": -2.0,
    "scandal": -2.5, "cut": -1.2, "downgrade": -1.5, "warn": -1.2, "investigation": -2.0
}

def analyze_deep_news(queries):
    total_score = 0
    article_count = 0
    seen_titles = set()

    for query in queries:
        # 時間がかかってもいいので、API制限を避けるため少し待つ
        time.sleep(0.5) 
        
        safe_query = query.replace(" ", "+")
        rss_url = f"https://news.google.com/rss/search?q={safe_query}+when:1d&hl=en-US&gl=US&ceid=US:en"
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:5]:
                title = entry.title
                if title in seen_titles: continue
                seen_titles.add(title)
                
                blob = TextBlob(title)
                polarity = blob.sentiment.polarity
                
                weight = 1.0
                title_lower = title.lower()
                for word, w_val in KEYWORDS_WEIGHT.items():
                    if word in title_lower:
                        weight = w_val
                        break
                
                total_score += polarity * abs(weight) * (1 if weight > 0 else -1)
                article_count += 1
        except: continue
            
    if article_count == 0: return 0.0, 0
    # スコアを強調 (-1.0 ~ 1.0)
    return max(-1.0, min(1.0, total_score / article_count * 2.5)), article_count

# --- 3. データ取得 (歴史学習のため長期間) ---
def get_market_data(ticker):
    try:
        # 過去2年分 (730日)
        df = yf.download(ticker, period="2y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

# --- 4. ★歴史的感応度 (Historical Sensitivity) ---
# 過去の「事件(Shock)」と「その後の動き」の因果関係を計算
def calculate_sensitivity(df):
    df = df.copy()
    # 3%以上の変動を「事件」と定義
    df["Daily_Return"] = df["Close"].pct_change()
    df["Is_Shock"] = df["Daily_Return"].abs() > 0.03
    
    # 事件の5日後の結果
    df["Next_Move"] = df["Close"].shift(-5).pct_change(periods=5)
    
    shock_data = df[df["Is_Shock"] == True]
    
    if len(shock_data) < 5: return 1.0 # データ不足は標準値
    
    # 相関計数 (-1.0 ~ 1.0)
    correlation = shock_data["Daily_Return"].corr(shock_data["Next_Move"])
    
    if np.isnan(correlation): return 1.0
    
    # 順張り体質（相関正）なら感応度を高く、逆張り体質（相関負）なら低く
    sensitivity = 1.0 + (correlation * 0.8)
    return max(0.5, min(2.5, sensitivity))

# --- 5. ★事件ベクトル解析 (Volume & Panic) ---
# 価格以外の「不気味な予兆」を数値化
def analyze_vectors(df):
    # A. 出来高ショック (Volume Shock)
    # 直近の出来高が、過去20日平均の何倍か？
    vol_mean = df["Volume"].rolling(20).mean()
    current_vol = df["Volume"].iloc[-1]
    vol_shock = current_vol / vol_mean.iloc[-1] if vol_mean.iloc[-1] > 0 else 1.0
    
    # B. パニックレベル (Volatility)
    # 市場の恐怖度 (直近20日の変動率の標準偏差)
    panic_level = df["Close"].pct_change().rolling(20).std().iloc[-1]
    if np.isnan(panic_level): panic_level = 0.015
    
    return vol_shock, panic_level

# --- 6. 総合分析 (Deep Impact Logic) ---
def analyze_stock(stock_info, is_portfolio=False):
    ticker = stock_info["ticker"]
    df = get_market_data(ticker)
    if df.empty or len(df) < 60: return None
    
    # --- Step 1: ニュース分析 (Sentiment) ---
    sentiment, art_count = analyze_deep_news(stock_info["queries"])
    
    # --- Step 2: 歴史的感応度 (Sensitivity) ---
    sensitivity = calculate_sensitivity(df)
    
    # --- Step 3: 事件ベクトル (Vector) ---
    vol_shock, panic_level = analyze_vectors(df)
    
    # --- Step 4: 精密利益予想 (Expected Profit) ---
    # 計算式: トレンド + (ニュース × (ボラティリティ × 感応度) × 出来高ブースト)
    
    # トレンド要素 (Trend)
    sma5 = df["Close"].rolling(5).mean().iloc[-1]
    sma20 = df["Close"].rolling(20).mean().iloc[-1]
    trend = (sma5 - sma20) / sma20
    
    # 出来高ブースト: 出来高が急増している時は、ニュースの信頼度を上げる
    volume_boost = 1.0
    if vol_shock > 1.5: volume_boost = 1.5 # 事件発生中！
    
    # ★ 最終計算式 ★
    impact_power = sentiment * panic_level * sensitivity * volume_boost * 4.0
    exp_profit_pct = (trend * 0.2) + impact_power
    
    # %表記にする (異常値クリップ)
    exp_profit_pct = max(-0.15, min(0.15, exp_profit_pct)) * 100
    
    # --- Step 5: アクション判定 ---
    ai_action = "WAIT ⚪"
    if exp_profit_pct > 1.0: ai_action = "BUY 🔵"
    if exp_profit_pct > 3.0: ai_action = "STRONG BUY 🚀"
    if exp_profit_pct < -1.0: ai_action = "SELL 🔴"
    if exp_profit_pct < -3.0: ai_action = "STRONG SELL ⚡"

    current_price = df["Close"].iloc[-1]
    
    # --- Step 6: ポートフォリオ診断 ---
    portfolio_advice = ""
    profit_loss = 0
    profit_loss_pct = 0
    
    if is_portfolio:
        buy_price = stock_info["buy_price"]
        amount = stock_info["amount"]
        profit_loss = (current_price - buy_price) * amount
        profit_loss_pct = (current_price - buy_price) / buy_price * 100
        
        if "BUY" in ai_action:
            if profit_loss > 0: portfolio_advice = "📈 Extend Gains"
            else: portfolio_advice = "🛡️ Buy Dip / Hold"
        elif "SELL" in ai_action:
            if profit_loss > 0: portfolio_advice = "💰 Take Profit"
            else: portfolio_advice = "🚑 Stop Loss"
        else:
            portfolio_advice = "👀 Watch"

    # 出来高警告アイコン
    vol_icon = ""
    if vol_shock > 2.0: vol_icon = "❗❗"
    elif vol_shock > 1.5: vol_icon = "❗"

    return {
        "name": stock_info["name"],
        "price": current_price,
        "currency": stock_info["currency"],
        "action": ai_action,
        "exp_profit": exp_profit_pct,
        "sentiment": sentiment,
        "articles": art_count,
        "sensitivity": sensitivity,
        "vol_shock": vol_shock,
        "vol_icon": vol_icon,
        "pl_amount": profit_loss,
        "pl_pct": profit_loss_pct,
        "advice": portfolio_advice
    }

# --- 7. レポート作成 ---
def update_readme(my_results, market_results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M (UTC)")
    
    total_pl_usd = sum([r["pl_amount"] for r in my_results if r["currency"] == "$"])
    total_pl_jpy = sum([r["pl_amount"] for r in my_results if r["currency"] == "¥"])
    
    def make_row(r, is_my_stock=False):
        prof_str = f"{r['exp_profit']:+.2f}%"
        if r['exp_profit'] > 0: prof_str = f"**{prof_str}**"
        
        sent = "⚪"
        if r['sentiment'] > 0.3: sent = "☀️"
        if r['sentiment'] < -0.3: sent = "☁️"
        
        # 感応度と出来高ショックを表示
        metrics = f"Sens:x{r['sensitivity']:.2f} / Vol:x{r['vol_shock']:.1f}{r['vol_icon']}"
        
        if is_my_stock:
            pl_str = f"{r['currency']}{r['pl_amount']:+,.0f} ({r['pl_pct']:+.1f}%)"
            if r['pl_amount'] > 0: pl_str = f"**{pl_str}** 🟢"
            else: pl_str = f"{pl_str} 🔴"
            return f"| {r['name']} | {pl_str} | {r['advice']} | {r['action']} | {prof_str} | {metrics} |\n"
        else:
            return f"| {r['action']} | {r['name']} | {r['currency']}{r['price']:,.0f} | {prof_str} | {metrics} | {sent} ({r['articles']}) |\n"

    my_rows = ""
    for r in my_results: my_rows += make_row(r, is_my_stock=True)
    
    market_rows = ""
    for r in market_results: market_rows += make_row(r, is_my_stock=False)

    content = f"""# 🎒 Deep Impact Portfolio (Heavy Analysis)
    
## 💰 Asset Summary
* **USD P/L:** ${total_pl_usd:+,.2f}
* **JPY P/L:** ¥{total_pl_jpy:+,.0f}

---

## 📢 My Portfolio Strategy
*Advice based on Position P/L + AI Prediction + Historical Sensitivity + Volume Shock.*

| Stock | Your P/L | Advice | Signal | Exp. Move | Metrics (Sens/Vol) |
| :--- | :--- | :--- | :--- | :--- | :--- |
{my_rows}

- **Sens:** Historical overreaction factor.
- **Vol:** Current volume shock (❗=Incident detected).

---

## 🌍 Market Watch
| Action | Stock | Price | Exp. Move | Metrics (Sens/Vol) | News |
| :--- | :--- | :--- | :--- | :--- | :--- |
{market_rows}

---
*Updated: {now}*
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)

def main():
    print("--- Loading Portfolio from Sheets ---")
    my_portfolio = load_portfolio_from_sheet()
    
    print("--- Analyzing My Portfolio ---")
    my_results = []
    for s in my_portfolio:
        res = analyze_stock(s, is_portfolio=True)
        if res: my_results.append(res)
    
    print("--- Analyzing Market ---")
    market_results = []
    for s in STOCKS_MARKET:
        res = analyze_stock(s, is_portfolio=False)
        if res: market_results.append(res)
            
    update_readme(my_results, market_results)
    print("Done!")

if __name__ == "__main__":
    main()

