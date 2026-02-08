import pandas as pd
import yfinance as yf
from datetime import datetime
import feedparser
from textblob import TextBlob
import numpy as np
import time

# ==========================================
# 👇 スプレッドシートのURL
# シートを使わない場合は、このまま（空文字のまま）でOKです。
# ==========================================
SHEET_URL = "https://docs.google.com/spreadsheets/d/10MtVu1vgAq0qJ0-O0lxHMy29_EZ7uG3-cSQlcXd0FUY/edit?usp=drivesdk" 

# --- 🧪 テスト用データ (シートがない場合に自動で使用) ---
TEST_PORTFOLIO = [
    {"ticker": "NVDA", "name": "NVIDIA", "buy_price": 50.0, "amount": 20, "currency": "$", "queries": ["NVIDIA stock", "AI chip demand"]},
    {"ticker": "AAPL", "name": "Apple", "buy_price": 180.0, "amount": 10, "currency": "$", "queries": ["Apple stock", "iPhone sales"]},
    {"ticker": "TSLA", "name": "Tesla", "buy_price": 400.0, "amount": 15, "currency": "$", "queries": ["Tesla stock", "EV market"]},
    {"ticker": "7974.T", "name": "任天堂", "buy_price": 6000, "amount": 100, "currency": "¥", "queries": ["任天堂 株価", "Switch 後継機"]},
    {"ticker": "7203.T", "name": "トヨタ", "buy_price": 2000, "amount": 200, "currency": "¥", "queries": ["トヨタ自動車", "円安 影響"]},
    {"ticker": "9984.T", "name": "SBG", "buy_price": 9000, "amount": 100, "currency": "¥", "queries": ["ソフトバンクグループ", "AI投資"]},
    {"ticker": "6758.T", "name": "ソニー", "buy_price": 15000, "amount": 100, "currency": "¥", "queries": ["ソニーグループ", "PS5 販売"]}
]

# --- 🌎 世界市場リスト (Market Watch: World) ---
MARKET_WORLD = [
    {"ticker": "MSFT", "name": "Microsoft", "currency": "$", "queries": ["Microsoft stock", "Azure cloud", "AI copilot"]},
    {"ticker": "GOOGL", "name": "Google", "currency": "$", "queries": ["Google stock", "Gemini AI", "ad revenue"]},
    {"ticker": "AMZN", "name": "Amazon", "currency": "$", "queries": ["Amazon stock", "AWS cloud", "e-commerce"]},
    {"ticker": "META", "name": "Meta", "currency": "$", "queries": ["Meta stock", "AI investment", "ad sales"]},
    {"ticker": "LLY", "name": "Eli Lilly", "currency": "$", "queries": ["Eli Lilly stock", "obesity drug"]}
]

# --- 🇯🇵 日本市場リスト (Market Watch: Japan) ---
MARKET_JAPAN = [
    {"ticker": "8035.T", "name": "東エレク", "currency": "¥", "queries": ["東京エレクトロン", "半導体製造装置"]},
    {"ticker": "9983.T", "name": "ファストリ", "currency": "¥", "queries": ["ファーストリテイリング", "ユニクロ 売上"]},
    {"ticker": "6861.T", "name": "キーエンス", "currency": "¥", "queries": ["キーエンス", "FAセンサー"]},
    {"ticker": "6098.T", "name": "リクルート", "currency": "¥", "queries": ["リクルート", "Indeed"]},
    {"ticker": "8306.T", "name": "三菱UFJ", "currency": "¥", "queries": ["三菱UFJ", "金利政策"]}
]

# --- 1. ポートフォリオ読み込み ---
def load_portfolio():
    print("\n📦 ポートフォリオデータの読み込みを開始します...")
    portfolio = []
    
    if SHEET_URL:
        try:
            print(f"   🌐 Googleスプレッドシートに接続中: {SHEET_URL[:30]}...")
            df = pd.read_csv(SHEET_URL)
            for index, row in df.iterrows():
                if pd.isna(row["Ticker"]): continue
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
            print(f"   ✅ {len(portfolio)} 銘柄の読み込みに成功しました。")
            return portfolio
        except Exception as e:
            print(f"   ❌ エラー: シートの読み込みに失敗しました ({e})")
            print("   ⚠️ テストデータモードに切り替えます。")

    print("   🧪 テストデータを使用します。")
    return TEST_PORTFOLIO

# --- 2. ★ニュース検索機能 (ここが検索学習の入り口) ---
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

    print(f"   🔍 ニュース検索中 (キーワード: {queries})")
    
    for query in queries:
        time.sleep(1.0) # 丁寧に待つ
        safe_query = query.replace(" ", "+")
        rss_url = f"https://news.google.com/rss/search?q={safe_query}+when:1d&hl=en-US&gl=US&ceid=US:en"
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:3]: # 各クエリ上位3件
                title = entry.title
                if title in seen_titles: continue
                seen_titles.add(title)
                
                # 感情分析
                blob = TextBlob(title)
                polarity = blob.sentiment.polarity
                
                # キーワード重み付け
                weight = 1.0
                title_lower = title.lower()
                detected_keyword = ""
                for word, w_val in KEYWORDS_WEIGHT.items():
                    if word in title_lower:
                        weight = w_val
                        detected_keyword = word
                        break
                
                score = polarity * abs(weight) * (1 if weight > 0 else -1)
                total_score += score
                article_count += 1
                
                # ログ出力（何を見つけたか報告）
                if detected_keyword:
                    print(f"      📰 重要: '{title[:30]}...' (単語:{detected_keyword}, スコア:{score:.2f})")
                
        except: continue
            
    if article_count == 0:
        print("      ⚪ ニュースなし")
        return 0.0, 0
    
    final_score = max(-1.0, min(1.0, total_score / article_count * 2.5))
    print(f"      📝 ニュース分析完了: 合計スコア {final_score:.2f} ({article_count}件)")
    return final_score, article_count

# --- 3. データ取得 ---
def get_market_data(ticker):
    try:
        df = yf.download(ticker, period="2y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

# --- 4. ★歴史的学習機能 (ここが学習の脳みそ) ---
def calculate_sensitivity(df):
    df = df.copy()
    # 過去のデータをスキャンして「事件」を探す
    df["Daily_Return"] = df["Close"].pct_change()
    df["Is_Shock"] = df["Daily_Return"].abs() > 0.03 # 3%以上の変動
    
    # 事件後の動きを追跡
    df["Next_Move"] = df["Close"].shift(-5).pct_change(periods=5)
    
    shock_data = df[df["Is_Shock"] == True]
    shock_count = len(shock_data)
    
    if shock_count < 5:
        print(f"      📚 歴史データ不足: ショック回数 {shock_count}回 (標準設定を使用)")
        return 1.0
    
    # 相関関係を学習 (相関係数)
    correlation = shock_data["Daily_Return"].corr(shock_data["Next_Move"])
    if np.isnan(correlation): correlation = 0
    
    # 感応度を算出
    sensitivity = 1.0 + (correlation * 0.8)
    sensitivity = max(0.5, min(2.5, sensitivity))
    
    print(f"      📚 歴史学習完了: 過去のショック {shock_count}回 → 感応度 x{sensitivity:.2f}")
    return sensitivity

# --- 5. 事件ベクトル (Volume & Panic) ---
def analyze_vectors(df):
    vol_mean = df["Volume"].rolling(20).mean()
    current_vol = df["Volume"].iloc[-1]
    vol_shock = current_vol / vol_mean.iloc[-1] if vol_mean.iloc[-1] > 0 else 1.0
    
    panic_level = df["Close"].pct_change().rolling(20).std().iloc[-1]
    if np.isnan(panic_level): panic_level = 0.015
    
    if vol_shock > 1.5:
        print(f"      ❗ 警告: 出来高急増中 (通常の{vol_shock:.1f}倍) - 事件の予兆あり")
    
    return vol_shock, panic_level

# --- 6. 総合分析 (Deep Impact Logic) ---
def analyze_stock(stock_info, is_portfolio=False):
    ticker = stock_info["ticker"]
    print(f"\n🤖 分析開始: {stock_info['name']} ({ticker})")
    
    df = get_market_data(ticker)
    if df.empty or len(df) < 60:
        print("   ❌ データ取得失敗")
        return None
    
    # ここで各機能を呼び出す
    sentiment, art_count = analyze_deep_news(stock_info["queries"]) # ニュース検索・学習
    sensitivity = calculate_sensitivity(df) # 歴史学習
    vol_shock, panic_level = analyze_vectors(df) # 異常検知
    
    # トレンド計算
    sma5 = df["Close"].rolling(5).mean().iloc[-1]
    sma20 = df["Close"].rolling(20).mean().iloc[-1]
    trend = (sma5 - sma20) / sma20
    
    # ★ 利益予想の計算 ★
    volume_boost = 1.5 if vol_shock > 1.5 else 1.0
    impact_power = sentiment * panic_level * sensitivity * volume_boost * 4.0
    exp_profit_pct = (trend * 0.2) + impact_power
    exp_profit_pct = max(-0.15, min(0.15, exp_profit_pct)) * 100
    
    print(f"      💰 予想利益率: {exp_profit_pct:+.2f}%")

    # アクション判定
    ai_action = "WAIT ⚪"
    if exp_profit_pct > 1.0: ai_action = "BUY 🔵"
    if exp_profit_pct > 3.0: ai_action = "STRONG BUY 🚀"
    if exp_profit_pct < -1.0: ai_action = "SELL 🔴"
    if exp_profit_pct < -3.0: ai_action = "STRONG SELL ⚡"

    current_price = df["Close"].iloc[-1]
    
    # ポートフォリオアドバイス
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

    vol_icon = "❗" if vol_shock > 1.5 else ""
    news_icon = "☀️" if sentiment > 0.3 else ("☁️" if sentiment < -0.3 else "⚪")

    return {
        "name": stock_info["name"],
        "price": current_price,
        "currency": stock_info["currency"],
        "action": ai_action,
        "exp_profit": exp_profit_pct,
        "sentiment": sentiment,
        "news_icon": news_icon,
        "articles": art_count,
        "sensitivity": sensitivity,
        "vol_shock": vol_shock,
        "vol_icon": vol_icon,
        "pl_amount": profit_loss,
        "pl_pct": profit_loss_pct,
        "advice": portfolio_advice
    }

# --- 7. レポート作成 ---
def update_readme(my_results, world_results, japan_results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M (UTC)")
    
    total_pl_usd = sum([r["pl_amount"] for r in my_results if r["currency"] == "$"])
    total_pl_jpy = sum([r["pl_amount"] for r in my_results if r["currency"] == "¥"])
    
    def make_table(results, table_type):
        if not results: return "No data available."
        
        if table_type == "MY_PORTFOLIO":
            header = "| Action | Stock | Your P/L | Advice | Exp. Move | Metrics (Sens/Vol) |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        else:
            header = "| Action | Stock | Price | Exp. Move | Metrics (Sens/Vol) | News |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
            
        rows = ""
        results.sort(key=lambda x: x["exp_profit"], reverse=True)
        
        for r in results:
            prof_str = f"{r['exp_profit']:+.2f}%"
            if r['exp_profit'] > 0: prof_str = f"**{prof_str}**"
            
            metrics = f"x{r['sensitivity']:.2f} / x{r['vol_shock']:.1f}{r['vol_icon']}"
            
            if table_type == "MY_PORTFOLIO":
                pl_str = f"{r['currency']}{r['pl_amount']:+,.0f} ({r['pl_pct']:+.1f}%)"
                if r['pl_amount'] > 0: pl_str = f"**{pl_str}** 🟢"
                else: pl_str = f"{pl_str} 🔴"
                rows += f"| {r['action']} | {r['name']} | {pl_str} | **{r['advice']}** | {prof_str} | {metrics} |\n"
            else:
                rows += f"| {r['action']} | {r['name']} | {r['currency']}{r['price']:,.0f} | {prof_str} | {metrics} | {r['news_icon']} ({r['articles']}) |\n"
        
        return header + rows

    content = f"""# 🏛️ Deep Impact Portfolio (Visualized Ver.)
    
## 💰 My Assets (Portfolio)
**Total P/L:** USD **${total_pl_usd:+,.2f}** / JPY **¥{total_pl_jpy:+,.0f}**

{make_table(my_results, "MY_PORTFOLIO")}

---

## 🌎 World Market (Watchlist)
{make_table(world_results, "MARKET")}

---

## 🇯🇵 Japan Market (Watchlist)
{make_table(japan_results, "MARKET")}

---
### 💡 Guide
* **Exp. Move:** AI predicted price change (Logic: `Trend` + `News` × `Sens` × `Vol`).
* **Sens (Sensitivity):** Learned from 2 years of history. `>1.0` means overreaction habit.
* **Vol (Volume Shock):** `>1.5` means incident detected ❗.

*Updated: {now}*
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)

def main():
    print("--- 🚀 AI Stock Analyst Starting ---")
    
    my_portfolio = load_portfolio()
    my_results = []
    print("\n--- 💰 Analyzing My Portfolio ---")
    for s in my_portfolio:
        res = analyze_stock(s, is_portfolio=True)
        if res: my_results.append(res)
    
    print("\n--- 🌎 Analyzing World Market ---")
    world_results = []
    for s in MARKET_WORLD:
        res = analyze_stock(s, is_portfolio=False)
        if res: world_results.append(res)
        
    print("\n--- 🇯🇵 Analyzing Japan Market ---")
    japan_results = []
    for s in MARKET_JAPAN:
        res = analyze_stock(s, is_portfolio=False)
        if res: japan_results.append(res)
            
    update_readme(my_results, world_results, japan_results)
    print("\n--- ✅ All Analysis Completed ---")

if __name__ == "__main__":
    main()

