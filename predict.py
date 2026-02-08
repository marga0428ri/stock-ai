import pandas as pd
import yfinance as yf
from datetime import datetime
import feedparser
from textblob import TextBlob
import numpy as np
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# ==========================================
# 👇 スプレッドシートのURL (使う場合)
# ==========================================
SHEET_URL = "https://docs.google.com/spreadsheets/d/10MtVu1vgAq0qJ0-O0lxHMy29_EZ7uG3-cSQlcXd0FUY/edit?usp=drivesdk" 

# --- 🧪 テスト用データ ---
TEST_PORTFOLIO = [
    {"ticker": "NVDA", "name": "NVIDIA", "buy_price": 50.0, "amount": 20, "currency": "$", "queries": ["NVIDIA stock", "AI chip demand"]},
    {"ticker": "AAPL", "name": "Apple", "buy_price": 180.0, "amount": 10, "currency": "$", "queries": ["Apple stock", "iPhone sales"]},
    {"ticker": "TSLA", "name": "Tesla", "buy_price": 400.0, "amount": 15, "currency": "$", "queries": ["Tesla stock", "EV market"]},
    {"ticker": "7974.T", "name": "任天堂", "buy_price": 6000, "amount": 100, "currency": "¥", "queries": ["任天堂 株価", "Switch 後継機"]},
    {"ticker": "7203.T", "name": "トヨタ", "buy_price": 2000, "amount": 200, "currency": "¥", "queries": ["トヨタ自動車", "円安 影響"]},
    {"ticker": "9984.T", "name": "SBG", "buy_price": 9000, "amount": 100, "currency": "¥", "queries": ["ソフトバンクグループ", "AI投資"]},
    {"ticker": "6758.T", "name": "ソニー", "buy_price": 15000, "amount": 100, "currency": "¥", "queries": ["ソニーグループ", "PS5 販売"]}
]

MARKET_WORLD = [
    {"ticker": "MSFT", "name": "Microsoft", "currency": "$", "queries": ["Microsoft stock", "Azure cloud"]},
    {"ticker": "GOOGL", "name": "Google", "currency": "$", "queries": ["Google stock", "Gemini AI"]},
    {"ticker": "AMZN", "name": "Amazon", "currency": "$", "queries": ["Amazon stock", "AWS cloud"]},
    {"ticker": "META", "name": "Meta", "currency": "$", "queries": ["Meta stock", "AI investment"]},
    {"ticker": "LLY", "name": "Eli Lilly", "currency": "$", "queries": ["Eli Lilly stock", "obesity drug"]}
]

MARKET_JAPAN = [
    {"ticker": "8035.T", "name": "東エレク", "currency": "¥", "queries": ["東京エレクトロン", "半導体製造装置"]},
    {"ticker": "9983.T", "name": "ファストリ", "currency": "¥", "queries": ["ファーストリテイリング", "ユニクロ 売上"]},
    {"ticker": "6861.T", "name": "キーエンス", "currency": "¥", "queries": ["キーエンス", "FAセンサー"]},
    {"ticker": "6098.T", "name": "リクルート", "currency": "¥", "queries": ["リクルート", "Indeed"]},
    {"ticker": "8306.T", "name": "三菱UFJ", "currency": "¥", "queries": ["三菱UFJ", "金利政策"]}
]

# --- 📧 メール通知機能 ---
def send_email_notify(subject, body):
    email_from = os.environ.get("EMAIL_FROM")
    email_pass = os.environ.get("EMAIL_PASS")
    email_to = os.environ.get("EMAIL_TO")

    if not email_from or not email_pass or not email_to:
        print("⚠️ メール設定なし。通知スキップ。")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = email_from
        msg['To'] = email_to
        msg['Subject'] = f"AI Stock Alert: {subject}"
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_from, email_pass)
        server.sendmail(email_from, email_to, msg.as_string())
        server.quit()
        print("📧 メール送信完了")
    except Exception as e:
        print(f"❌ メールエラー: {e}")

# --- 1. データ読み込み ---
def load_portfolio():
    print("\n📦 ポートフォリオ読み込み...")
    portfolio = []
    if SHEET_URL:
        try:
            df = pd.read_csv(SHEET_URL)
            for index, row in df.iterrows():
                if pd.isna(row["Ticker"]): continue
                queries = [q.strip() for q in str(row["Query"]).split(",")]
                item = {
                    "ticker": str(row["Ticker"]).strip(),
                    "name": str(row["Name"]),
                    "buy_price": float(row["BuyPrice"]),
                    "amount": int(row["Amount"]),
                    "currency": str(row["Currency"]).strip(),
                    "queries": queries
                }
                portfolio.append(item)
            return portfolio
        except: pass
    return TEST_PORTFOLIO

# --- 2. ニュース分析 ---
KEYWORDS_WEIGHT = {
    "record": 2.0, "surge": 1.5, "jump": 1.5, "beat": 1.5, "approval": 2.0,
    "buyback": 1.2, "dividend": 1.2, "partnership": 1.2, "launch": 1.2,
    "plunge": -1.5, "miss": -1.5, "drop": -1.2, "fail": -1.5, "lawsuit": -2.0,
    "scandal": -2.5, "cut": -1.2, "investigation": -2.0
}

def analyze_deep_news(queries):
    total_score = 0
    article_count = 0
    seen_titles = set()
    print(f"   🔍 ニュース検索: {queries}")
    for query in queries:
        time.sleep(1.0)
        safe_query = query.replace(" ", "+")
        rss_url = f"https://news.google.com/rss/search?q={safe_query}+when:1d&hl=en-US&gl=US&ceid=US:en"
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:3]:
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
    return max(-1.0, min(1.0, total_score / article_count * 2.5)), article_count

# --- 3. データ取得 ---
def get_market_data(ticker):
    try:
        # 過去2年分 (730日)
        df = yf.download(ticker, period="2y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

# --- 4. ★テクニカル分析 (最高値・RSI) ---
def analyze_technical(df):
    # RSI (14日) の計算
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    # 最高値からの下落率 (Drawdown) - 過去1年
    max_price = df["Close"].rolling(252).max().iloc[-1]
    current_price = df["Close"].iloc[-1]
    drawdown = (current_price - max_price) / max_price # マイナスの値になる (-0.1 = -10%)
    
    return current_rsi, drawdown

# --- 5. 歴史的学習 ---
def calculate_sensitivity(df):
    df = df.copy()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Is_Shock"] = df["Daily_Return"].abs() > 0.03
    df["Next_Move"] = df["Close"].shift(-5).pct_change(periods=5)
    shock_data = df[df["Is_Shock"] == True]
    if len(shock_data) < 5: return 1.0
    correlation = shock_data["Daily_Return"].corr(shock_data["Next_Move"])
    if np.isnan(correlation): correlation = 0
    return max(0.5, min(2.5, 1.0 + (correlation * 0.8)))

# --- 6. 事件ベクトル ---
def analyze_vectors(df):
    vol_mean = df["Volume"].rolling(20).mean()
    current_vol = df["Volume"].iloc[-1]
    vol_shock = current_vol / vol_mean.iloc[-1] if vol_mean.iloc[-1] > 0 else 1.0
    panic_level = df["Close"].pct_change().rolling(20).std().iloc[-1]
    if np.isnan(panic_level): panic_level = 0.015
    return vol_shock, panic_level

# --- 7. 総合分析 ---
def analyze_stock(stock_info, is_portfolio=False):
    ticker = stock_info["ticker"]
    print(f"\n🤖 分析開始: {stock_info['name']} ({ticker})")
    df = get_market_data(ticker)
    if df.empty or len(df) < 252: return None
    
    # 各種分析
    sentiment, art_count = analyze_deep_news(stock_info["queries"])
    sensitivity = calculate_sensitivity(df)
    vol_shock, panic_level = analyze_vectors(df)
    rsi, drawdown = analyze_technical(df) # ★追加: チャート分析
    
    # トレンド
    sma5 = df["Close"].rolling(5).mean().iloc[-1]
    sma20 = df["Close"].rolling(20).mean().iloc[-1]
    trend = (sma5 - sma20) / sma20
    
    # ★利益予想の計算 (RSIと最高値も考慮)★
    volume_boost = 1.5 if vol_shock > 1.5 else 1.0
    
    # チャート要因の補正
    # RSI > 75 (買われすぎ) なら下落圧力、RSI < 30 (売られすぎ) なら上昇圧力
    rsi_pressure = 0
    if rsi > 75: rsi_pressure = -0.5
    elif rsi < 30: rsi_pressure = 0.5
    
    # 最高値に近い(Drawdown > -0.05)なら、ブレイクアウト期待でプラス
    # 逆に大きく落ちている(-0.3)なら、反発期待でプラス
    # 中途半端(-0.15)な位置が一番動きにくい
    drawdown_factor = 0
    if drawdown > -0.05: drawdown_factor = 0.2 # 高値更新期待
    elif drawdown < -0.30: drawdown_factor = 0.3 # バーゲンセール
    
    # 最終スコア
    impact_power = sentiment * panic_level * sensitivity * volume_boost * 4.0
    exp_profit_pct = (trend * 0.2) + impact_power + (rsi_pressure * 0.01) + (drawdown_factor * 0.01)
    exp_profit_pct = max(-0.15, min(0.15, exp_profit_pct)) * 100
    
    print(f"      💰 予想: {exp_profit_pct:+.2f}% (RSI:{rsi:.0f}, Drop:{drawdown*100:.1f}%)")

    # アクション判定
    ai_action = "WAIT"
    if exp_profit_pct > 1.0: ai_action = "BUY"
    if exp_profit_pct > 3.0: ai_action = "STRONG BUY"
    if exp_profit_pct < -1.0: ai_action = "SELL"
    if exp_profit_pct < -3.0: ai_action = "STRONG SELL"

    current_price = df["Close"].iloc[-1]
    portfolio_advice = ""
    profit_loss = 0
    
    if is_portfolio:
        buy_price = stock_info["buy_price"]
        amount = stock_info["amount"]
        profit_loss = (current_price - buy_price) * amount
        profit_loss_pct = (current_price - buy_price) / buy_price * 100
        
        # アドバイスロジック (RSIも考慮)
        if "BUY" in ai_action:
            if profit_loss > 0: portfolio_advice = "Extend Gains"
            else: portfolio_advice = "Buy Dip"
        elif "SELL" in ai_action:
            if profit_loss > 0: portfolio_advice = "Take Profit"
            else: portfolio_advice = "Stop Loss"
        else:
            # RSIが高すぎる場合の警告
            if rsi > 80: portfolio_advice = "Overheated (Caution)"
            else: portfolio_advice = "Watch"

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
        "rsi": rsi,
        "drawdown": drawdown * 100, # %表示
        "pl_amount": profit_loss,
        "pl_pct": profit_loss_pct if is_portfolio else 0,
        "advice": portfolio_advice
    }

# --- 8. レポート作成 & 通知 ---
def update_readme_and_notify(my_results, world_results, japan_results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M (UTC)")
    total_pl_usd = sum([r["pl_amount"] for r in my_results if r["currency"] == "$"])
    total_pl_jpy = sum([r["pl_amount"] for r in my_results if r["currency"] == "¥"])

    # メール作成
    email_body = f"AI Stock Report - {now}\n\n"
    notify_needed = False

    email_body += "--- 💰 Portfolio Check ---\n"
    for r in my_results:
        if "STRONG" in r["action"] or r["vol_shock"] > 1.5 or r["rsi"] > 80 or r["rsi"] < 20:
            notify_needed = True
            
        pl_mark = "🟢" if r["pl_amount"] > 0 else "🔴"
        email_body += f"■ {r['name']}: {r['action']} ({r['advice']})\n"
        email_body += f"   P/L: {r['currency']}{r['pl_amount']:+,.0f} {pl_mark}\n"
        email_body += f"   RSI: {r['rsi']:.0f} / Drop: {r['drawdown']:.1f}%\n\n"

    email_body += "--- 🌍 Market Watch ---\n"
    for r in world_results + japan_results:
        if "STRONG" in r["action"]:
            notify_needed = True
            email_body += f"★ {r['name']}: {r['action']} (Exp: {r['exp_profit']:+.2f}%)\n"

    if notify_needed:
        print("🔔 通知条件クリア: メール送信")
        send_email_notify("Important Market Updates", email_body)
    else:
        print("⚪ 通知なし")

    # README更新
    def make_table(results, table_type):
        if not results: return "No data."
        if table_type == "MY_PORTFOLIO":
            header = "| Action | Stock | Your P/L | Advice | Exp. Move | Chart (RSI/Drop) |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        else:
            header = "| Action | Stock | Price | Exp. Move | Chart (RSI/Drop) | News |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        rows = ""
        results.sort(key=lambda x: x["exp_profit"], reverse=True)
        for r in results:
            prof_str = f"{r['exp_profit']:+.2f}%"
            if r['exp_profit'] > 0: prof_str = f"**{prof_str}**"
            
            # Chart Metrics (RSIと下落率)
            chart_metrics = f"RSI:{r['rsi']:.0f} / Drop:{r['drawdown']:.1f}%"
            if r['rsi'] > 75: chart_metrics += " 🔥" # 加熱
            if r['rsi'] < 30: chart_metrics += " 💎" # バーゲン
            
            if table_type == "MY_PORTFOLIO":
                pl_str = f"{r['currency']}{r['pl_amount']:+,.0f} ({r['pl_pct']:+.1f}%)"
                if r['pl_amount'] > 0: pl_str = f"**{pl_str}** 🟢"
                else: pl_str = f"{pl_str} 🔴"
                rows += f"| {r['action']} | {r['name']} | {pl_str} | **{r['advice']}** | {prof_str} | {chart_metrics} |\n"
            else:
                rows += f"| {r['action']} | {r['name']} | {r['currency']}{r['price']:,.0f} | {prof_str} | {chart_metrics} | {r['news_icon']} ({r['articles']}) |\n"
        return header + rows

    content = f"""# 🏛️ Deep Impact Portfolio (Chart Master)
*Updated: {now}*

## 💰 My Assets
**Total P/L:** USD **${total_pl_usd:+,.2f}** / JPY **¥{total_pl_jpy:+,.0f}**
{make_table(my_results, "MY_PORTFOLIO")}

---
## 🌎 World Market
{make_table(world_results, "MARKET")}

---
## 🇯🇵 Japan Market
{make_table(japan_results, "MARKET")}

---
### 💡 Guide
* **RSI (0-100):**
    * `>70`: Overheated 🔥 (Risk of drop).
    * `<30`: Oversold 💎 (Chance to buy).
* **Drop (Drawdown):** How much % down from 1-year High.
* **Advice:** AI combines News, Trends, and Chart Levels.
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
            
    update_readme_and_notify(my_results, world_results, japan_results)
    print("\n--- ✅ All Analysis Completed ---")

if __name__ == "__main__":
    main()
