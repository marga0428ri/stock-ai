
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from data.fetch_data import get_data

def add_features(df):
    """
    株価データから特徴量（AIの判断材料）を作る
    """
    df = df.copy()
    
    # 終値の変化率
    df["Return"] = df["Close"].pct_change()
    
    # 移動平均線（5日、25日）
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_25"] = df["Close"].rolling(window=25).mean()
    
    # 移動平均からの乖離率
    df["Deviation"] = (df["SMA_5"] - df["SMA_25"]) / df["SMA_25"]
    
    # ★ここを「1日後（明日）」に戻しました★
    # 明日の終値が今日より高ければ 1、そうでなければ 0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    
    df.dropna(inplace=True)
    return df

def main():
    print("--- 1. データ取得 ---")
    df = get_data("AAPL")
    
    print("--- 2. データ加工 ---")
    df_processed = add_features(df)
    
    # 学習に使う項目
    features = ["Return", "Deviation"]
    target = "Target"
    
    # データをセット
    X = df_processed[features]
    y = df_processed[target]
    
    # 最後の1日（今日）以外で学習
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    
    # 今日のデータ（明日を予測するため）
    X_latest = X.iloc[-1:] 
    
    print("--- 3. AI学習 ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("--- 4. 明日の予測 ---")
    prediction = model.predict(X_latest)[0]
    probability = model.predict_proba(X_latest)[0]
    
    result = "上がる (Up)" if prediction == 1 else "下がる (Down)"
    confidence = probability[prediction] * 100
    
    print(f"==========================================")
    print(f"【予測】明日の AAPL 株価は... {result}")
    print(f"AIの自信度: {confidence:.1f}%")
    print(f"==========================================")

if __name__ == "__main__":
    main()
