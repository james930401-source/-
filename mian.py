# main.py


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def load_data(path: str = "StudentsPerformance.csv") -> pd.DataFrame:
    """
    讀取 Kaggle Students Performance in Exams 資料集
    檔名預設為 StudentsPerformance.csv
    """
    df = pd.read_csv(path)
    return df


def basic_eda(df: pd.DataFrame):
    """
    簡單資料探索：顯示欄位、前幾筆與描述統計
    """
    print("===== 資料前 5 筆 =====")
    print(df.head())
    print("\n===== 資料欄位 =====")
    print(df.columns)
    print("\n===== 數值型欄位描述統計 =====")
    print(df.describe())


def add_avg_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    建立三科平均分數與高分標籤：
    高分定義：三科平均 >= 75 → 1；否則 0
    """
    df = df.copy()
    df["avg_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
    df["high_score"] = (df["avg_score"] >= 75).astype(int)
    return df


def preprocess_for_supervised(df: pd.DataFrame):
    """
    監督式學習前處理：
    - 目標：high_score
    - 特徵：所有數值欄位 + one-hot 之後的類別欄位（不含 avg_score）
    """
    df = df.copy()

    # 目標變數
    y = df["high_score"]

    # 刪除不需要的欄位（如果有）
    drop_cols = ["high_score"]  # 目標欄位本身不要當特徵
    # avg_score 可留可不留，看你 PPT 怎麼寫；這裡先保留當特徵
    # drop_cols.append("avg_score")

    X = df.drop(columns=drop_cols)

    # 類別變數做 one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def supervised_random_forest(X, y):
    """
    使用 Random Forest 做二元分類，預測高分/非高分
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== 監督式學習：Random Forest 分類結果 =====")
    print(f"Accuracy 準確率：{acc:.4f}")
    print("\nConfusion Matrix 混淆矩陣：")
    print(cm)
    print("\nClassification Report：")
    print(classification_report(y_test, y_pred, digits=4))

    # 顯示前幾個重要特徵
    feature_importances = pd.Series(
        clf.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    print("\n前 10 個重要特徵：")
    print(feature_importances.head(10))

    return clf


def unsupervised_kmeans(df: pd.DataFrame, n_clusters: int = 3):
    """
    使用 K-Means 針對三科成績做分群，並搭配 PCA 做 2D 表示
    """
    # 只拿數值成績
    features = df[["math score", "reading score", "writing score"]].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["cluster"] = cluster_labels

    print("\n===== 非監督式學習：K-Means 分群結果 =====")
    print(df_clustered.groupby("cluster")[["math score", "reading score", "writing score"]].mean())
    print("\n各群學生人數：")
    print(df_clustered["cluster"].value_counts())

    # PCA 做 2D 示意（如果要畫圖，可以再補）
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df_clustered["pca1"] = X_pca[:, 0]
    df_clustered["pca2"] = X_pca[:, 1]

    # 這裡先不畫圖，避免需要圖形介面
    # 想畫可以自行加上 matplotlib / seaborn

    return df_clustered


def main():
    # 1. 讀取資料
    df = load_data("StudentsPerformance.csv")

    # 2. 簡單 EDA
    basic_eda(df)

    # 3. 加上平均分數與高分標籤
    df = add_avg_and_label(df)

    # 4. 監督式學習：Random Forest 分類
    X, y = preprocess_for_supervised(df)
    _ = supervised_random_forest(X, y)

    # 5. 非監督式學習：K-Means 分群
    _ = unsupervised_kmeans(df, n_clusters=3)


if __name__ == "__main__":
    main()
