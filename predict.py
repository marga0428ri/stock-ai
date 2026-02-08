import pandas as pd
import yfinance as yf
from datetime import datetime
import feedparser
from textblob import TextBlob
import numpy as np
import time
import smtplib # メール送信機能
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# ==========================================
# 👇 スプレッドシートのURL
# ==========================================
SHEET_URL = "" 

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
    # GitHub Secretsから情報を取得
    email_from = os.environ.get("EMAIL_FROM")
    email_pass = os.environ.get("EMAIL_PASS")
    email_to = os.environ.get("EMAIL_TO")

    if not email_from or not email_pass or not email_to:
        print("⚠️ メール設定が足りません。GitHub Secretsを確認してください。")
        return

    try:
        # メールの作成
        msg = MIMEMultipart()
        msg['From'] = email_from
        msg['To'] = email_to
        msg['Subject'] = f"AI Stock Alert: {subject}"
        
        # 本文（プレーンテキスト）
        msg.attach(MIMEText(body, 'plain'))
        
        # Gmailのサーバーに接続して送信
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_from, email_pass)
        text = msg.as_string()
        server.sendmail(email_from, email_to, text)
        server.quit()
        
        print("📧 レポートメールを送信しました！")
    except Exception as e:
        print(f"❌ メール送信エラー: {e}")

# --- 1. ポートフォリオ読み込み ---
def load_portfolio():
    print("\n📦 ポートフォリオデータの読み込み...")
    portfolio = []
    if SHEET_URL:
        try:
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
            return portfolio
        except: pass
    return TEST_PORTFOLIO

# --- 2. ニュース分析 ---
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
        df = yf.download(ticker, period="2y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

# --- 4. 歴史的学習 ---
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

# --- 5. 事件ベクトル ---
def analyze_vectors(df):
    vol_mean = df["Volume"].rolling(20).mean()
    current_vol = df["Volume"].iloc[-1]
    vol_shock = current_vol / vol_mean.iloc[-1] if vol_mean.iloc[-1] > 0 else 1.0
    panic_level = df["Close"].pct_change().rolling(20).std().iloc[-1]
    if np.isnan(panic_level): panic_level = 0.015
    return vol_shock, panic_level

# --- 6. 総合分析 ---
def analyze_stock(stock_info, is_portfolio=False):
    ticker = stock_info["ticker"]
    print(f"\n🤖 分析開始: {stock_info['name']} ({ticker})")
    df = get_market_data(ticker)
    if df.empty or len(df) < 60: return None
    
    sentiment, art_count = analyze_deep_news(stock_info["queries"])
    sensitivity = calculate_sensitivity(df)
    vol_shock, panic_level = analyze_vectors(df)
    
    sma5 = df["Close"].rolling(5).mean().iloc[-1]
    sma20 = df["Close"].rolling(20).mean().iloc[-1]
    trend = (sma5 - sma20) / sma20
    
    volume_boost = 1.5 if vol_shock > 1.5 else 1.0
    impact_power = sentiment * panic_level * sensitivity * volume_boost * 4.0
    exp_profit_pct = (trend * 0.2) + impact_power
    exp_profit_pct = max(-0.15, min(0.15, exp_profit_pct)) * 100
    print(f"      💰 予想利益率: {exp_profit_pct:+.2f}%")

    ai_action = "WAIT"
    if exp_profit_pct > 1.0: ai_action = "BUY"
    if exp_profit_pct > 3.0: ai_action = "STRONG BUY"
    if exp_profit_pct < -1.0: ai_action = "SELL"
    if exp_profit_pct < -3.0: ai_action = "STRONG SELL"

    current_price = df["Close"].iloc[-1]
    portfolio_advice = ""
    profit_loss = 0
    profit_loss_pct = 0
    
    if is_portfolio:
        buy_price = stock_info["buy_price"]
        amount = stock_info["amount"]
        profit_loss = (current_price - buy_price) * amount
        profit_loss_pct = (current_price - buy_price) / buy_price * 100
        
        if "BUY" in ai_action:
            if profit_loss > 0: portfolio_advice = "Extend Gains"
            else: portfolio_advice = "Buy Dip"
        elif "SELL" in ai_action:
            if profit_loss > 0: portfolio_advice = "Take Profit"
            else: portfolio_advice = "Stop Loss"
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
        "pl_amount": profit_loss,
        "pl_pct": profit_loss_pct,
        "advice": portfolio_advice
    }

# --- 7. レポート作成 & メール通知 ---
def update_readme_and_notify(my_results, world_results, japan_results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M (UTC)")
    total_pl_usd = sum([r["pl_amount"] for r in my_results if r["currency"] == "$"])
    total_pl_jpy = sum([r["pl_amount"] for r in my_results if r["currency"] == "¥"])

    # メール本文の作成
    email_body = f"AI Stock Report - {now}\n\n"
    notify_needed = False

    # 自分のポートフォリオ
    email_body += "--- 💰 Your Portfolio ---\n"
    for r in my_results:
        # 警告条件: 強いシグナル or 出来高異常 or ポートフォリオへの重要アドバイス
        if "STRONG" in r["action"] or r["vol_shock"] > 1.5 or "Stop Loss" in r["advice"] or "Take Profit" in r["advice"]:
            notify_needed = True
            
        pl_mark = "🟢" if r["pl_amount"] > 0 else "🔴"
        email_body += f"■ {r['name']}: {r['action']} ({r['advice']})\n"
        email_body += f"   P/L: {r['currency']}{r['pl_amount']:+,.0f} {pl_mark}\n"
        email_body += f"   Exp: {r['exp_profit']:+.2f}% / Vol: x{r['vol_shock']:.1f}\n\n"

    # 市場のチャンス
    email_body += "--- 🌍 Market Opportunities ---\n"
    for r in world_results + japan_results:
        if "STRONG" in r["action"]:
            notify_needed = True
            email_body += f"★ {r['name']}: {r['action']} (Exp: {r['exp_profit']:+.2f}%)\n"

    # 重要事項があればメール送信
    if notify_needed:
        print("🔔 重要なシグナルあり。メール送信します。")
        send_email_notify("Important Market Updates", email_body)
    else:
        print("⚪ 平穏な市場です。メール通知はスキップします。")

    # README更新（表示用）
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

    content = f"""# 🏛️ Deep Impact Portfolio
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
