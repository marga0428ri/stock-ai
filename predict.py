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
# 👇 スプレッドシート設定
# ==========================================
PORTFOLIO_ID = "10MtVu1vgAq0qJ0-O0lxHMy29_EZ7uG3-cSQlcXd0FUY"
PORTFOLIO_URL = f"https://docs.google.com/spreadsheets/d/{PORTFOLIO_ID}/pub?output=csv"

WATCHLIST_ID = "1xLSJ_neFSs_1_huTZ_zW4pYDYLzcx-iFE77mOpZUT2U"
WATCHLIST_URL = f"https://docs.google.com/spreadsheets/d/{WATCHLIST_ID}/pub?output=csv"

# ==========================================
# 👇 有名企業リスト (固定監視)
# ==========================================
MARKET_WORLD = [
    {"ticker": "MSFT", "name": "Microsoft", "currency": "$", "queries": ["Microsoft stock", "Azure cloud"]},
    {"ticker": "GOOGL", "name": "Google", "currency": "$", "queries": ["Google stock", "Gemini AI"]},
    {"ticker": "AMZN", "name": "Amazon", "currency": "$", "queries": ["Amazon stock", "AWS cloud"]},
    {"ticker": "META", "name": "Meta", "currency": "$", "queries": ["Meta stock", "AI ad sales"]},
    {"ticker": "LLY", "name": "Eli Lilly", "currency": "$", "queries": ["Eli Lilly stock", "Lilly earnings"]}
]

MARKET_JAPAN = [
    {"ticker": "8035.T", "name": "東エレク", "currency": "¥", "queries": ["東京エレクトロン", "半導体製造装置"]},
    {"ticker": "9983.T", "name": "ファストリ", "currency": "¥", "queries": ["ファーストリテイリング", "ユニクロ売上"]},
    {"ticker": "6861.T", "name": "キーエンス", "currency": "¥", "queries": ["キーエンス", "FAセンサー"]},
    {"ticker": "6098.T", "name": "リクルート", "currency": "¥", "queries": ["リクルート", "Indeed"]},
    {"ticker": "8306.T", "name": "三菱UFJ", "currency": "¥", "queries": ["三菱UFJ", "金利政策"]}
]

# --- 📧 メール送信機能 ---
def send_email_notify(subject, body):
    email_from = os.environ.get("EMAIL_FROM")
    email_pass = os.environ.get("EMAIL_PASS")
    email_to = os.environ.get("EMAIL_TO")
    if not all([email_from, email_pass, email_to]): return
    try:
        msg = MIMEMultipart()
        msg['From'], msg['To'], msg['Subject'] = email_from, email_to, f"AI Alert: {subject}"
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_from, email_pass)
        server.sendmail(email_from, email_to, msg.as_string())
        server.quit()
        print("   📧 メールを送信しました")
    except Exception as e: print(f"   ❌ メールエラー: {e}")

# --- 1. シート読み込み ---
def load_sheet_data(url, is_watchlist=False):
    try:
        response = requests.get(url)
        df = pd.read_csv(io.StringIO(response.text))
        res = []
        for _, row in df.iterrows():
            if pd.isna(row.get("Ticker")): continue
            q = [q.strip() for q in str(row.get("Query", "")).split(",")] if row.get("Query") else [f"{row['Ticker']} news"]
            item = {"ticker": str(row["Ticker"]).strip(), "name": str(row.get("Name", row["Ticker"])), "currency": str(row.get("Currency", "$")).strip(), "queries": q}
            if not is_watchlist:
                item.update({"buy_price": float(row.get("BuyPrice", 0)), "amount": int(row.get("Amount", 0))})
            res.append(item)
        return res
    except: return []

# --- 2. ★Deep News 解析 (単語ポイント換算) ---
KEYWORDS_SCORE = {
    "record": 2.0, "surge": 1.5, "beat": 1.5, "approval": 2.0, "buyback": 1.2, "partnership": 1.2,
    "plunge": -1.5, "miss": -1.5, "lawsuit": -2.0, "scandal": -2.5, "cut": -1.2, "investigation": -2.0
}

def analyze_deep_news(queries):
    total_score, article_count, seen = 0, 0, set()
    for query in queries:
        time.sleep(1.0)
        try:
            feed = feedparser.parse(f"https://news.google.com/rss/search?q={query.replace(' ','+')}+when:1d&hl=en-US&gl=US&ceid=US:en")
            for entry in feed.entries[:3]:
                if entry.title in seen: continue
                seen.add(entry.title)
                score = TextBlob(entry.title).sentiment.polarity
                title_lower = entry.title.lower()
                for word, weight in KEYWORDS_SCORE.items():
                    if word in title_lower:
                        score *= abs(weight) * (1 if weight > 0 else -1)
                        print(f"      📰 抽出: '{entry.title[:30]}...' (点数補正: {weight})")
                total_score += score
                article_count += 1
        except: continue
    if article_count == 0: return 0.0, 0
    return max(-1.0, min(1.0, total_score / article_count * 2.5)), article_count

# --- 3. ★歴史的学習 (Sensitivity) ---
def calculate_sensitivity(df):
    df = df.copy()
    df["Ret"] = df["Close"].pct_change()
    df["Shock"] = df["Ret"].abs() > 0.03
    df["Next"] = df["Close"].shift(-5).pct_change(periods=5)
    sd = df[df["Shock"]]
    if len(sd) < 5: return 1.0
    corr = sd["Ret"].corr(sd["Next"])
    sens = max(0.5, min(2.5, 1.0 + (corr * 0.8 if not np.isnan(corr) else 0)))
    print(f"      📚 学習完了: 過去の反応癖 x{sens:.2f}")
    return sens

# --- 4. ★事件・チャート分析 ---
def analyze_market_condition(df):
    vol_shock = df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1]
    panic = df["Close"].pct_change().rolling(20).std().iloc[-1] or 0.015
    delta = df["Close"].diff()
    gain, loss = delta.where(delta > 0, 0).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / loss).iloc[-1]))
    dd = (df["Close"].iloc[-1] - df["Close"].rolling(252).max().iloc[-1]) / df["Close"].rolling(252).max().iloc[-1]
    return vol_shock, panic, rsi, dd

# --- 5. 統合分析エンジン ---
def analyze_stock(info, is_portfolio=False):
    print(f"\n🤖 分析中: {info['name']} ({info['ticker']})")
    df = yf.download(info['ticker'], period="2y", progress=False)
    if df.empty or len(df) < 252: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

    sentiment, art_cnt = analyze_deep_news(info['queries'])
    sens = calculate_sensitivity(df)
    vol, pnc, rsi, dd = analyze_market_condition(df)
    trend = (df["Close"].rolling(5).mean().iloc[-1] - df["Close"].rolling(20).mean().iloc[-1]) / df["Close"].rolling(20).mean().iloc[-1]
    
    # ★計算: トレンド + (ニュース感情 × 恐怖度 × 感応度 × 出来高ブースト)
    impact = sentiment * pnc * sens * (1.5 if vol > 1.5 else 1.0) * 4.0
    exp = max(-0.15, min(0.15, (trend * 0.2) + impact + (0.005 if rsi < 30 else -0.005 if rsi > 75 else 0))) * 100
    
    act, emo = ("WAIT", "⚪")
    if exp > 3: act, emo = ("STRONG BUY", "🚀")
    elif exp > 1: act, emo = ("BUY", "🔵")
    elif exp < -3: act, emo = ("STRONG SELL", "⚡")
    elif exp < -1: act, emo = ("SELL", "🔴")

    pl, pl_pct, adv = 0, 0, "-"
    if is_portfolio and info.get("buy_price", 0) > 0:
        pl = (df["Close"].iloc[-1] - info["buy_price"]) * info["amount"]
        pl_pct = (df["Close"].iloc[-1] - info["buy_price"]) / info["buy_price"] * 100
        if "BUY" in act: adv = "利益拡大 (Extend)" if pl > 0 else "押し目買い (Hold/Buy)"
        elif "SELL" in act: adv = "利益確定 (Profit)" if pl > 0 else "損切り検討 (Stop Loss)"
        else: adv = "加熱注意" if rsi > 80 else "様子見"

    return {**info, "price": df["Close"].iloc[-1], "action": act, "emoji": emo, "exp": exp, "rsi": rsi, "vol": vol, "pl": pl, "pl_pct": pl_pct, "advice": adv, "news_icon": "☀️" if sentiment > 0.3 else "☁️" if sentiment < -0.3 else "⚪", "art_cnt": art_cnt}

# --- 6. レポート作成 ---
def update_report(my, watch, world, jp):
    now = datetime.now().strftime("%Y-%m-%d %H:%M (UTC)")
    total_usd = sum(r["pl"] for r in my if r["currency"] == "$")
    total_jpy = sum(r["pl"] for r in my if r["currency"] == "¥")

    def make_table(res, mode):
        if not res: return "データなし"
        h = "| Signal | Stock | P/L | Advice | Data (Exp/RSI) |\n| :---: | :--- | :--- | :--- | :--- |\n" if mode == "MY" else "| Signal | Stock | Price | Exp. | Analysis |\n| :---: | :--- | :--- | :--- | :--- |\n"
        rows = ""
        for r in sorted(res, key=lambda x: x["exp"], reverse=True):
            if mode == "MY": rows += f"| {r['emoji']} {r['action']} | **{r['name']}** | {'🟢' if r['pl']>=0 else '🔴'} {r['currency']}{r['pl']:+,.0f}<br>({r['pl_pct']:+.1f}%) | {r['advice']} | Exp: **{r['exp']:+.1f}%**<br>RSI: {r['rsi']:.0f} |\n"
            else: rows += f"| {r['emoji']} {r['action']} | **{r['name']}** | {r['currency']}{r['price']:,.0f} | **{r['exp']:+.2f}%** | {r['news_icon']} ({r['art_cnt']})<br>RSI: {r['rsi']:.0f} |\n"
        return h + rows

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(f"# 📊 AI Strategy Dashboard\n*Update: {now}*\n\n## 💰 My Portfolio\n**Total:** ${total_usd:+,.0f} / ¥{total_jpy:+,.0f}\n\n{make_table(my, 'MY')}\n\n## 👀 Watch List (Interest)\n{make_table(watch, 'WATCH')}\n\n## 🌎 World Giants\n{make_table(world, 'WORLD')}\n\n## 🇯🇵 Japan Leading\n{make_table(jp, 'JP')}")

def main():
    my = load_sheet_data(PORTFOLIO_URL)
    watch = load_sheet_data(WATCHLIST_URL, True)
    results = {"MY": [analyze_stock(s, True) for s in my], "WATCH": [analyze_stock(s) for s in watch], "WORLD": [analyze_stock(s) for s in MARKET_WORLD], "JP": [analyze_stock(s) for s in MARKET_JAPAN]}
    update_report(results["MY"], results["WATCH"], results["WORLD"], results["JP"])

if __name__ == "__main__": main()
