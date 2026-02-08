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
# 👇 スプレッドシート設定 (自動連携)
# ==========================================
# 1. 買った株 (My Portfolio)
PORTFOLIO_ID = "10MtVu1vgAq0qJ0-O0lxHMy29_EZ7uG3-cSQlcXd0FUY"
PORTFOLIO_URL = f"https://docs.google.com/spreadsheets/d/{PORTFOLIO_ID}/pub?output=csv"

# 2. 気になる株 (Watch List)
WATCHLIST_ID = "1xLSJ_neFSs_1_huTZ_zW4pYDYLzcx-iFE77mOpZUT2U"
WATCHLIST_URL = f"https://docs.google.com/spreadsheets/d/{WATCHLIST_ID}/pub?output=csv"

# --- 🧪 テスト用データ (万が一シートが読めない時用) ---
TEST_PORTFOLIO = [
    {"ticker": "NVDA", "name": "NVIDIA", "buy_price": 50.0, "amount": 20, "currency": "$", "queries": ["NVIDIA stock", "AI chip demand"]},
    {"ticker": "7203.T", "name": "トヨタ", "buy_price": 2000, "amount": 200, "currency": "¥", "queries": ["トヨタ自動車", "円安 影響"]}
]

# --- 📧 メール通知機能 ---
def send_email_notify(subject, body):
    email_from = os.environ.get("EMAIL_FROM")
    email_pass = os.environ.get("EMAIL_PASS")
    email_to = os.environ.get("EMAIL_TO")

    if not email_from or not email_pass or not email_to:
        print("   ⚠️ メール設定が見つかりません。通知はスキップします。")
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
        print("   📧 レポートメールを送信しました！")
    except Exception as e:
        print(f"   ❌ メール送信エラー: {e}")

# --- 1. シート読み込み (共通関数) ---
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
            
            # クエリ処理 (カンマ区切り対応)
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

            # ポートフォリオ専用データ
            if not is_watchlist:
                item["buy_price"] = float(row.get("BuyPrice", 0))
                item["amount"] = int(row.get("Amount", 0))
            
            data_list.append(item)
        print(f"   ✅ {len(data_list)} 件のデータを取得しました。")
        return data_list
    except Exception as e:
        print(f"   ❌ 読み込み失敗: {e}")
        return []

# --- 2. ニュース分析 (Deep News Logic) ---
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
    
    print(f"   🔍 ニュース検索開始 (キーワード: {queries})")
    
    for query in queries:
        time.sleep(1.0) # 丁寧に読むための待機
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
                detected_word = ""
                
                for word, w_val in KEYWORDS_WEIGHT.items():
                    if word in title_lower:
                        weight = w_val
                        detected_word = word
                        break
                
                score = polarity * abs(weight) * (1 if weight > 0 else -1)
                total_score += score
                article_count += 1
                
                if detected_word:
                    print(f"      📰 重要記事発見: '{title[:30]}...' (単語:{detected_word}, スコア:{score:.2f})")
                    
        except: continue
            
    if article_count == 0:
        print("      ⚪ 関連ニュースなし")
        return 0.0, 0
    
    final_score = max(-1.0, min(1.0, total_score / article_count * 2.5))
    print(f"      📝 ニュース分析完了: 感情スコア {final_score:.2f} ({article_count}件)")
    return final_score, article_count

# --- 3. データ取得 (過去2年分) ---
def get_market_data(ticker):
    try:
        df = yf.download(ticker, period="2y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

# --- 4. ★歴史的感応度 (Historical Sensitivity) ---
# 過去のショックに対する反応癖を学習する最重要ロジック
def calculate_sensitivity(df):
    df = df.copy()
    # 3%以上の変動を「事件」と定義
    df["Daily_Return"] = df["Close"].pct_change()
    df["Is_Shock"] = df["Daily_Return"].abs() > 0.03
    
    # その後の動き (5日間)
    df["Next_Move"] = df["Close"].shift(-5).pct_change(periods=5)
    
    shock_data = df[df["Is_Shock"] == True]
    shock_count = len(shock_data)
    
    if shock_count < 5:
        print(f"      📚 歴史データ不足: ショック回数 {shock_count}回 (標準設定を使用)")
        return 1.0
    
    # 相関関係を計算 (順張り癖か、逆張り癖か)
    correlation = shock_data["Daily_Return"].corr(shock_data["Next_Move"])
    if np.isnan(correlation): correlation = 0
    
    sensitivity = 1.0 + (correlation * 0.8)
    sensitivity = max(0.5, min(2.5, sensitivity))
    
    print(f"      📚 歴史学習完了: 過去のショック {shock_count}回 → 感応度 x{sensitivity:.2f}")
    return sensitivity

# --- 5. 事件ベクトル & テクニカル分析 ---
def analyze_vectors_and_chart(df):
    # A. 出来高ショック (Volume Shock)
    vol_mean = df["Volume"].rolling(20).mean()
    current_vol = df["Volume"].iloc[-1]
    vol_shock = current_vol / vol_mean.iloc[-1] if vol_mean.iloc[-1] > 0 else 1.0
    
    if vol_shock > 1.5:
        print(f"      ❗ 警告: 出来高急増中 ({vol_shock:.1f}倍) - 事件の予兆あり")

    # B. パニックレベル (Volatility)
    panic_level = df["Close"].pct_change().rolling(20).std().iloc[-1]
    if np.isnan(panic_level): panic_level = 0.015

    # C. RSI (買われすぎ判定)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    # D. 最高値からの下落率 (Drawdown)
    max_price = df["Close"].rolling(252).max().iloc[-1]
    current_price = df["Close"].iloc[-1]
    drawdown = (current_price - max_price) / max_price
    
    return vol_shock, panic_level, current_rsi, drawdown

# --- 6. 総合分析 (Deep Impact Logic: Complete) ---
def analyze_stock(stock_info, is_portfolio=False):
    ticker = stock_info["ticker"]
    print(f"\n🤖 分析開始: {stock_info['name']} ({ticker})")
    
    df = get_market_data(ticker)
    if df.empty or len(df) < 252:
        print("   ❌ データ不足のためスキップ")
        return None
    
    # 各種詳細分析を実行
    sentiment, art_count = analyze_deep_news(stock_info["queries"])
    sensitivity = calculate_sensitivity(df)
    vol_shock, panic_level, rsi, drawdown = analyze_vectors_and_chart(df)
    
    # トレンド計算
    sma5 = df["Close"].rolling(5).mean().iloc[-1]
    sma20 = df["Close"].rolling(20).mean().iloc[-1]
    trend = (sma5 - sma20) / sma20
    
    # ★ 利益予想の計算式 (全要素統合版) ★
    # 1. 出来高ブースト: 事件発生時はニュースの影響を倍増
    volume_boost = 1.5 if vol_shock > 1.5 else 1.0
    
    # 2. ニュース・イベントの影響力計算
    # (ニュース感情 × 市場の恐怖度 × その株の感応度 × 出来高)
    impact_power = sentiment * panic_level * sensitivity * volume_boost * 4.0
    
    # 3. チャート要因の補正
    # RSI > 75 (加熱) -> 下落圧力 / RSI < 30 (売られすぎ) -> 上昇圧力
    rsi_pressure = 0
    if rsi > 75: rsi_pressure = -0.5
    elif rsi < 30: rsi_pressure = 0.5
    
    # 最高値からの距離 (Drawdown)
    # ほどよい下落(-5%)は押し目買いチャンス、暴落(-30%)はリバウンド期待
    drawdown_factor = 0
    if drawdown > -0.05: drawdown_factor = 0.1
    elif drawdown < -0.30: drawdown_factor = 0.3
    
    # 最終予想計算
    exp_profit_pct = (trend * 0.2) + impact_power + (rsi_pressure * 0.01) + (drawdown_factor * 0.01)
    
    # 現実的な範囲にクリップ (-15% ~ +15%)
    exp_profit_pct = max(-0.15, min(0.15, exp_profit_pct)) * 100
    
    print(f"      💰 最終予想: {exp_profit_pct:+.2f}% (RSI:{rsi:.0f})")

    # アクション判定
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
            
            # アドバイス生成ロジック
            if "BUY" in ai_action:
                if profit_loss > 0: portfolio_advice = "利益拡大チャンス (Extend)"
                else: portfolio_advice = "押し目買い/耐える (Hold/Buy)"
            elif "SELL" in ai_action:
                if profit_loss > 0: portfolio_advice = "利益確定推奨 (Take Profit)"
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
def update_readme_and_notify(my_results, watch_results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M (UTC)")
    
    # 資産計算
    total_pl_usd = sum([r["pl_amount"] for r in my_results if r["currency"] == "$"])
    total_pl_jpy = sum([r["pl_amount"] for r in my_results if r["currency"] == "¥"])

    # メール本文作成
    email_body = f"AI Stock Report - {now}\n\n"
    notify_needed = False

    email_body += "--- 💰 My Portfolio Analysis ---\n"
    for r in my_results:
        # 通知条件: 強いシグナル or 損切り推奨 or 出来高異常
        if "STRONG" in r["action"] or "Stop Loss" in r["advice"] or r["vol_shock"] > 1.5:
            notify_needed = True
        
        pl_mark = "🟢" if r["pl_amount"] > 0 else "🔴"
        email_body += f"■ {r['name']}: {r['emoji']} {r['action']} ({r['advice']})\n"
        email_body += f"   P/L: {r['currency']}{r['pl_amount']:+,.0f} {pl_mark}\n"
        email_body += f"   Exp: {r['exp_profit']:+.2f}% / RSI: {r['rsi']:.0f}\n\n"

    email_body += "--- 👀 Watch List Opportunities ---\n"
    for r in watch_results:
        if "STRONG" in r["action"]:
            notify_needed = True
            email_body += f"★ {r['name']}: {r['emoji']} {r['action']} (Exp: {r['exp_profit']:+.2f}%)\n"

    if notify_needed:
        print("🔔 通知条件クリア: メール送信を実行します")
        send_email_notify("Important Market Update", email_body)
    else:
        print("⚪ 通知なし: 平穏な市場です")

    # --- 見やすさ重視のテーブル作成 ---
    def make_portfolio_table(results):
        if not results: return "データがありません。"
        header = "| Signal | Stock | P/L (損益) | AI Advice | Data (Exp/RSI) |\n| :---: | :--- | :--- | :--- | :--- |\n"
        rows = ""
        results.sort(key=lambda x: x["exp_profit"], reverse=True)
        
        for r in results:
            pl_str = f"{r['currency']}{r['pl_amount']:+,.0f} <br> ({r['pl_pct']:+.1f}%)"
            pl_icon = "🟢" if r['pl_amount'] >= 0 else "🔴"
            details = f"Exp: **{r['exp_profit']:+.1f}%** <br> RSI: {r['rsi']:.0f}"
            rows += f"| {r['emoji']} **{r['action']}** | **{r['name']}** | {pl_icon} {pl_str} | {r['advice']} | {details} |\n"
        return header + rows

    def make_watchlist_table(results):
        if not results: return "データがありません。"
        header = "| Signal | Stock | Price | Exp. Move | Analysis |\n| :---: | :--- | :--- | :--- | :--- |\n"
        rows = ""
        results.sort(key=lambda x: x["exp_profit"], reverse=True)
        
        for r in results:
            price_str = f"{r['currency']}{r['price']:,.0f}"
            exp_str = f"**{r['exp_profit']:+.2f}%**"
            
            # 分析アイコンまとめ (News, RSI, Vol)
            analysis = f"{r['news_icon']} News <br> RSI: {r['rsi']:.0f}"
            if r['vol_shock'] > 1.5: analysis += f" <br> ❗ Vol: x{r['vol_shock']:.1f}"
            
            rows += f"| {r['emoji']} **{r['action']}** | **{r['name']}** | {price_str} | {exp_str} | {analysis} |\n"
        return header + rows

    content = f"""# 📊 AI Investment Dashboard
*Updated: {now}*

## 💰 My Portfolio (保有資産)
**Total P/L:** USD **${total_pl_usd:+,.2f}** / JPY **¥{total_pl_jpy:+,.0f}**

{make_portfolio_table(my_results)}

---

## 👀 Watch List (気になる株)
**Market Opportunities**

{make_watchlist_table(watch_results)}

---
### 💡 Dashboard Guide
* **Signal:**
    * 🚀 **STRONG BUY**: 強い買いシグナル (+3%以上予想)
    * 🔵 **BUY**: 上昇トレンド (+1%以上予想)
    * ⚪ **WAIT**: 様子見
    * 🔴 **SELL**: 下落警戒 (-1%以下予想)
    * ⚡ **STRONG SELL**: 暴落警戒 (-3%以下予想)
* **Analysis:**
    * **News:** ☀️ Good / ☁️ Bad
    * **RSI:** >70 (Overheated), <30 (Oversold)
    * **Vol:** ❗ Unusual Volume Detected (Incident)
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)

def main():
    print("--- 🚀 AI Stock Analyst Starting ---")
    
    # 1. 持ち株の読み込み
    my_portfolio = load_sheet_data(PORTFOLIO_URL, is_watchlist=False)
    if not my_portfolio: my_portfolio = TEST_PORTFOLIO
    
    # 2. 気になる株の読み込み
    watch_list = load_sheet_data(WATCHLIST_URL, is_watchlist=True)
    
    # 3. 分析実行
    my_results = []
    print("\n--- 💰 Analyzing My Portfolio ---")
    for s in my_portfolio:
        res = analyze_stock(s, is_portfolio=True)
        if res: my_results.append(res)
    
    watch_results = []
    print("\n--- 👀 Analyzing Watch List ---")
    for s in watch_list:
        res = analyze_stock(s, is_portfolio=False)
        if res: watch_results.append(res)
            
    # 4. レポート更新
    update_readme_and_notify(my_results, watch_results)
    print("\n--- ✅ All Analysis Completed ---")

if __name__ == "__main__":
    main()
