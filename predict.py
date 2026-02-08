%%writefile predict.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data.fetch_data import get_data

def add_features(df):
    """
    【修正版】5日後の予測に向けたデータ加工
    """
    df = df.copy()
    
    # 変化率（リターン）
    df["Return"] = df["Close"].pct_change()
    
    # 移動平均線（5日、25日、60日を追加）
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_25"] = df["Close"].rolling(window=25).mean()
    df["SMA_60"] = df["Close"].rolling(window=60).mean()
    
    # 特徴量: 移動平均からの乖離率（トレンドの強さ）
    df["Deviation_5"] = (df["Close"] - df["SMA_5"]) / df["SMA_5"]
    df["Deviation_25"] = (df["Close"] - df["SMA_25"]) / df["SMA_25"]
    
    # ★ここが重要変更点★
    # 「明日(1)」ではなく「5日後(5)」を予測対象にする
    # 5日後の終値が、今日よりも高ければ 1、そうでなければ 0
    prediction_days = 5
    df["Target"] = (df["Close"].shift(-prediction_days) > df["Close"]).astype(int)
    
    # データのない行を削除
    df.dropna(inplace=True)
    return df

def main():
    print("--- 1. データ取得開始 ---")
    df = get_data("AAPL")
    
    if df.empty:
        print("エラー: データが取得できませんでした。")
        return

    print("--- 2. AI学習用データ加工（5日後予測） ---")
    df_processed = add_features(df)
    
    # 学習に使うデータ
    features = ["Return", "Deviation_5", "Deviation_25"]
    target = "Target"
    
    X = df_processed[features]
    y = df_processed[target]
    
    # 最新のデータ以外で学習
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    
    # 今日のデータ
    X_latest = X.iloc[-1:] 
    
    print("--- 3. AIモデル学習中 ---")
    # チャッピーのアドバイス通り、まだLSTMなどは使わずランダムフォレストを使う
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    print("--- 4. 予測実行 ---")
    prediction = model.predict(X_latest)[0]
    probability = model.predict_proba(X_latest)[0]
    
    result = "上がる (Up)" if prediction == 1 else "下がる (Down)"
    confidence = probability[prediction] * 100
    
    print(f"\n{'='*40}")
    print(f"【5日後の予測】AAPL 株価は... {result}")
    print(f" AIの自信度(確率): {confidence:.1f}%")
    print(f"{'='*40}\n")
    print("※この確率は『キャリブレーション（確率の正直さ）』が重要です")

if __name__ == "__main__":
    main()
