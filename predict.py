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
import io
import requests

# ==========================================
# 👇 2つのスプレッドシート設定
# ==========================================
# 1. 買った株 (My Portfolio)
PORTFOLIO_ID = "10MtVu1vgAq0qJ0-O0lxHMy29_EZ7uG3-cSQlcXd0FUY"
PORTFOLIO_URL = f"https://docs.google.com/spreadsheets/d/{PORTFOLIO_ID}/pub?output=csv"

# 2. 気になる株 (Watch List)
WATCHLIST_ID = "1xLSJ_neFSs_1_huTZ_zW4pYDYLzcx-iFE77mOpZUT2U"
WATCHLIST_URL = f"https://docs.google.com/spreadsheets/d/{WATCHLIST_ID}/pub?output=csv"

# ==========================================
# 👇 3. 世界・日本の有名企業リスト (標準装備)
# ==========================================
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
        print("   ⚠️ メール設定なし。通知スキップ。")
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
        print("   📧 メール送信完了")
    except Exception as e:
        print(f"   ❌ メールエラー: {e}")

# --- 1. シート読み込み ---
def load_sheet_data(url, is_watchlist=False):
    sheet_name = "Watch List" if is_watchlist else "My Portfolio"
    print(f"\n📦 シート読み込み中: {sheet_name}...")
    data_list = []
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        
        for index, row in df.iterrows():
            if pd.isna(row.get("Ticker")): continue
            
            queries = []
            if "Query" in row and not pd.isna(row["Query"]):
                queries = [q.strip() for q in str(row["Query"]).split(",")]
            else:
                queries = [f"{row['Ticker']} stock news"]
            
            item = {
                "ticker": str(row["Ticker"]).strip(),
                "name": str(row.get("Name", row["Ticker"])),
                "currency": str(row.get("Currency", "$")).strip(),
                "queries": queries
            }

            if not is_watchlist:
                item["buy_price"] = float(row.get("BuyPrice", 0))
                item["amount"] = int(row.get("Amount", 0))
            
            data_list.append(item)
        print(f"   ✅ {len(data_list)} 件読み込み成功")
        return data_list
    except Exception as e:
        print(f"   ❌ 読み込み失敗: {e}")
        return []

# --- 2. ニュース分析 ---
KEYWORDS_WEIGHT = {
    "record": 2.0, "surge": 1.5, "jump": 1.5, "beat": 1.5, "approval": 2.0,
    "buyback": 1.2, "dividend": 1.2, "partnership": 1.2, "launch": 1.2,
    "plunge": -1.5, "miss": -1.5, "drop": -1.2, "fail": -1.5, "lawsuit": -2.0,
    "scandal": -2.5, "cut": -1.2, "investigation": -2.0, "warn": -1.2
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
        df = yf.download(ticker, period="2y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

# --- 4. 歴史的感応度 ---
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

# --- 5. 事件ベクトル & テクニカル ---
def analyze_vectors_and_chart(df):
    vol_mean = df["Volume"].rolling(20).mean()
    current_vol = df["Volume"].iloc[-1]
    vol_shock = current_vol / vol_mean.iloc[-1] if vol_mean.iloc[-1] > 0 else 1.0
    
    panic_level = df["Close"].pct_change().rolling(20).std().iloc[-1]
    if np.isnan(panic_level): panic_level = 0.015
    
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    max_price = df["Close"].rolling(252).max().iloc[-1]
    current_price = df["Close"].iloc[-1]
    drawdown = (current_price - max_price) / max_price
    
    return vol_shock, panic_level, current_rsi, drawdown

# --- 6. 総合分析 ---
def analyze_stock(stock_info, is_portfolio=False):
    ticker = stock_info["ticker"]
    print(f"\n🤖 分析開始: {stock_info['name']} ({ticker})")
    df = get_market_data(ticker)
    if df.empty or len(df) < 252: return None
    
    sentiment, art_count = analyze_deep_news(stock_info["queries"])
    sensitivity = calculate_sensitivity(df)
    vol_shock, panic_level, rsi, drawdown = analyze_vectors_and_chart(df)
    
    sma5 = df["Close"].rolling(5).mean().iloc[-1]
    sma20 = df["Close"].rolling(20).mean().iloc[-1]
    trend = (sma5 - sma20) / sma20
    
    volume_boost = 1.5 if vol_shock > 1.5 else 1.0
    rsi_pressure = 0
    if rsi > 75: rsi_pressure = -0.5
    elif rsi < 30: rsi_pressure = 0.5
    
    drawdown_factor = 0
    if drawdown > -0.05: drawdown_factor = 0.1
    elif drawdown < -0.30: drawdown_factor = 0.3
    
    impact_power = sentiment * panic_level * sensitivity * volume_boost * 4.0
    exp_profit_pct = (trend * 0.2) + impact_power + (rsi_pressure * 0.01) + (drawdown_factor * 0.01)
    exp_profit_pct = max(-0.15, min(0.15, exp_profit_pct)) * 100
    
    print(f"      💰 予想: {exp_profit_pct:+.2f}%")

    ai_action = "WAIT"
    action_emoji = "⚪"
    if exp_profit_pct > 1.0: 
        ai_action = "BUY"
        action_emoji = "🔵"
    if exp_profit_pct > 3.0: 
        ai_action = "STRONG BUY"
        action_emoji = "🚀"
    if exp_profit_pct < -1.0: 
        ai_action = "SELL"
        action_emoji = "🔴"
    if exp_profit_pct < -3.0: 
        ai_action = "STRONG SELL"
        action_emoji = "⚡"

    current_price = df["Close"].iloc[-1]
    portfolio_advice = "-"
    profit_loss = 0
    profit_loss_pct = 0
    
    if is_portfolio:
        buy_price = stock_info.get("buy_price", 0)
        amount = stock_info.get("amount", 0)
        if buy_price > 0:
            profit_loss = (current_price - buy_price) * amount
            profit_loss_pct = (current_price - buy_price) / buy_price * 100
            
            if "BUY" in ai_action:
                if profit_loss > 0: portfolio_advice = "利益拡大 (Extend)"
                else: portfolio_advice = "押し目/耐える (Hold)"
            elif "SELL" in ai_action:
                if profit_loss > 0: portfolio_advice = "利確推奨 (Take Profit)"
                else: portfolio_advice = "損切り検討 (Stop Loss)"
            else:
                if rsi > 80: portfolio_advice = "加熱注意 (Caution)"
                elif rsi < 20: portfolio_advice = "売られすぎ (Oversold)"
                else: portfolio_advice = "様子見 (Watch)"

    vol_icon = "❗" if vol_shock > 1.5 else ""
    news_icon = "☀️" if sentiment > 0.3 else ("☁️" if sentiment < -0.3 else "⚪")

    return {
        "name": stock_info["name"],
        "price": current_price,
        "currency": stock_info["currency"],
        "action": ai_action,
        "emoji": action_emoji,
        "exp_profit": exp_profit_pct,
        "sentiment": sentiment,
        "news_icon": news_icon,
        "articles": art_count,
        "sensitivity": sensitivity,
        "vol_shock": vol_shock,
        "vol_icon": vol_icon,
        "rsi": rsi,
        "drawdown": drawdown * 100,
        "pl_amount": profit_loss,
        "pl_pct": profit_loss_pct,
        "advice": portfolio_advice
    }

# --- 7. レポート作成 & 通知 ---
def update_readme_and_notify(my_results, watch_results, world_results, japan_results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M (UTC)")
    
    total_pl_usd = sum([r["pl_amount"] for r in my_results if r["currency"] == "$"])
    total_pl_jpy = sum([r["pl_amount"] for r in my_results if r["currency"] == "¥"])

    # メール本文
    email_body = f"AI Stock Report - {now}\n\n"
    notify_needed = False

    email_body += "--- 💰 My Portfolio ---\n"
    for r in my_results:
        if "STRONG" in r["action"] or r["vol_shock"] > 1.5: notify_needed = True
        email_body += f"■ {r['name']}: {r['action']} ({r['advice']}) P/L:{r['pl_amount']}\n"

    # 通知判定
    if notify_needed:
        print("🔔 通知条件クリア: メール送信")
        send_email_notify("Market Update", email_body)
    else:
        print("⚪ 通知なし")

    # --- テーブル作成 (デザイン重視) ---
    def make_table(results, table_type="MARKET"):
        if not results: return "No Data."
        
        # ヘッダー切り替え
        if table_type == "MY_PORTFOLIO":
            header = "| Signal | Stock | P/L (損益) | Advice | Data (Exp/RSI) |\n| :---: | :--- | :--- | :--- | :--- |\n"
        else:
            header = "| Signal | Stock | Price | Exp. Move | Analysis |\n| :---: | :--- | :--- | :--- | :--- |\n"
            
        rows = ""
        results.sort(key=lambda x: x["exp_profit"], reverse=True)
        
        for r in results:
            # 共通項目
            details = f"Exp: **{r['exp_profit']:+.1f}%** <br> RSI: {r['rsi']:.0f}"
            analysis = f"{r['news_icon']} News <br> RSI: {r['rsi']:.0f}"
            if r['vol_shock'] > 1.5: analysis += f" <br> ❗ Vol: x{r['vol_shock']:.1f}"

            if table_type == "MY_PORTFOLIO":
                pl_str = f"{r['currency']}{r['pl_amount']:+,.0f} <br> ({r['pl_pct']:+.1f}%)"
                pl_icon = "🟢" if r['pl_amount'] >= 0 else "🔴"
                rows += f"| {r['emoji']} **{r['action']}** | **{r['name']}** | {pl_icon} {pl_str} | {r['advice']} | {details} |\n"
            else:
                price_str = f"{r['currency']}{r['price']:,.0f}"
                rows += f"| {r['emoji']} **{r['action']}** | **{r['name']}** | {price_str} | **{r['exp_profit']:+.2f}%** | {analysis} |\n"
        return header + rows

    content = f"""# 📊 AI Investment Dashboard
*Updated: {now}*

## 💰 My Portfolio (保有資産)
**Total P/L:** USD **${total_pl_usd:+,.2f}** / JPY **¥{total_pl_jpy:+,.0f}**

{make_table(my_results, "MY_PORTFOLIO")}

---

## 👀 Watch List (気になる株)
{make_table(watch_results, "WATCH")}

---

## 🌎 World Giants (有名企業)
{make_table(world_results, "MARKET")}

---

## 🇯🇵 Japan Giants (有名企業)
{make_table(japan_results, "MARKET")}

---
### 💡 Guide
* **Signal:** 🚀Strong Buy / 🔵Buy / ⚪Wait / 🔴Sell / ⚡Strong Sell
* **Analysis:**
    * **Exp:** Expected Move for tomorrow.
    * **RSI:** >70(High) / <30(Low).
    * **Vol:** ❗Unusual Volume Detected.
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)

def main():
    print("--- 🚀 AI Stock Analyst Starting ---")
    
    # 1. 読み込み
    my_portfolio = load_sheet_data(PORTFOLIO_URL, is_watchlist=False)
    watch_list = load_sheet_data(WATCHLIST_URL, is_watchlist=True)
    
    # 2. 分析ループ
    all_lists = [
        ("💰 My Portfolio", my_portfolio),
        ("👀 Watch List", watch_list),
        ("🌎 World Market", MARKET_WORLD),
        ("🇯🇵 Japan Market", MARKET_JAPAN)
    ]
    
    results = {}
    
    for name, data in all_lists:
        print(f"\n--- {name} ---")
        res_list = []
        for s in data:
            res = analyze_stock(s, is_portfolio=(name == "💰 My Portfolio"))
            if res: res_list.append(res)
        results[name] = res_list

    # 3. レポート更新
    update_readme_and_notify(
        results["💰 My Portfolio"],
        results["👀 Watch List"],
        results["🌎 World Market"],
        results["🇯🇵 Japan Market"]
    )
    print("\n--- ✅ All Analysis Completed ---")

if __name__ == "__main__":
    main()
