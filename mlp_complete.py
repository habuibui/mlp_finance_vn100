# -------------------------------------------------------------
# 1. IMPORTS & GLOBAL SETTINGS
# -------------------------------------------------------------
"""
MLP Stock Prediction Model - Complete Pipeline
-----------------------------------------------
This module implements a Multi-Layer Perceptron (MLP) classifier for stock 
price movement prediction using technical indicators, fundamental data, 
and sentiment analysis features.

Features:
- Ablation study comparing different feature combinations
- Time-series cross-validation
- SHAP-based feature importance
- Comprehensive visualization suite

Author: [Your Name]
Course: [Course Name]
Date: 2026
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy import stats  # For statistical tests


from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import TimeSeriesSplit

# optional imbalanced‑learn for SMOTE
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    warnings.warn("imbalanced-learn not installed – SMOTE will be skipped.")

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 12
RANDOM_STATE = 42

# -------------------------------------------------------------
# 2. PATHS
# -------------------------------------------------------------
DATA_DIR = "data"
OHLC_PATH = os.path.join(DATA_DIR, "ohlc.csv")
FUND_PATH = os.path.join(DATA_DIR, "fundamental.csv")
NEWS_RAW_PATH = os.path.join(DATA_DIR, "news.csv")
NEWS_SENT_PATH = os.path.join(DATA_DIR, "news_with_sentiment.csv")
VIS_DIR = "visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 3. HELPER: clean column names (strip whitespace, lower‑case)
# -------------------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df

# -------------------------------------------------------------
# 4. LOAD DATA
# -------------------------------------------------------------
print("=" * 60)
print("Loading OHLC, fundamental & news data")
print("=" * 60)

ohlc = clean_columns(pd.read_csv(OHLC_PATH, parse_dates=["date"]))
fundamental = clean_columns(pd.read_csv(FUND_PATH))
# Load sentiment – if pre‑processed file exists we use it, otherwise raw news
if os.path.exists(NEWS_SENT_PATH):
    news = clean_columns(pd.read_csv(NEWS_SENT_PATH, parse_dates=["date"]))
    print("✓ Loaded pre‑processed sentiment data")
else:
    news = clean_columns(pd.read_csv(NEWS_RAW_PATH, parse_dates=["date"]))
    print("⚠️ Sentiment not pre‑processed – using raw news (no sentiment columns).")

# -------------------------------------------------------------
# 5. TECHNICAL INDICATORS (enhanced but still lightweight)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("Computing technical indicators & Momentum")
print("=" * 60)

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Simple moving averages
for w in [5, 10, 20, 50]:
    ohlc[f"sma_{w}"] = ohlc.groupby("mack")["close"].transform(lambda x: x.rolling(w).mean())

# Exponential moving averages
for w in [5, 10, 20, 50]:
    ohlc[f"ema_{w}"] = ohlc.groupby("mack")["close"].transform(lambda x: x.ewm(span=w, adjust=False).mean())

# RSI
ohlc["rsi"] = ohlc.groupby("mack")["close"].transform(compute_rsi)

# MACD
ema_12 = ohlc.groupby("mack")["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
ema_26 = ohlc.groupby("mack")["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
ohlc["macd"] = ema_12 - ema_26
ohlc["macd_signal"] = ohlc.groupby("mack")["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
ohlc["macd_hist"] = ohlc["macd"] - ohlc["macd_signal"]

# Bollinger Bands (20‑day)
ohlc["bb_mid"] = ohlc["sma_20"]
ohlc["bb_std"] = ohlc.groupby("mack")["close"].transform(lambda x: x.rolling(20).std())
ohlc["bb_upper"] = ohlc["bb_mid"] + 2 * ohlc["bb_std"]
ohlc["bb_lower"] = ohlc["bb_mid"] - 2 * ohlc["bb_std"]

# Volume ratio
vol = ohlc.groupby("mack")["volume"]
ohlc["vol_ma_20"] = vol.transform(lambda x: x.rolling(20).mean())
ohlc["volume_ratio"] = ohlc["volume"] / ohlc["vol_ma_20"]

# Momentum / Lagged Features (CRITICAL for MLPs)
for lag in [1, 3, 5, 10]:
    ohlc[f"ret_{lag}d"] = ohlc.groupby("mack")["close"].transform(lambda x: x.pct_change(lag))

# Volatility (Standard Deviation of returns over 20 days)
ohlc["volatility_20d"] = ohlc.groupby("mack")["close"].transform(lambda x: x.pct_change().rolling(20).std())

# -------------------------------------------------------------
# 6. FUNDAMENTAL DATA – DAILY RESAMPLE
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("Processing fundamental data")
print("=" * 60)

# Convert year/quarter to month‑end date
fundamental["date"] = pd.to_datetime(
    dict(year=fundamental["nam"], month=fundamental["quy"] * 3, day=1)
) + pd.offsets.MonthEnd(0)

fund_features = [
    "eps",
    "roe",
    "roa",
    "pb",
    "pe",
    "lnst_yoy",
    "nophaitra_vcsh",
    "vonhoa_tts",
]

fund_daily = (
    fundamental.sort_values(["mack", "date"])
    .set_index("date")
    .groupby("mack")[fund_features]
    .resample("D")
    .ffill()
    .reset_index()
)

# -------------------------------------------------------------
# 7. SENTIMENT AGGREGATION (daily) – safe guard if columns missing
# -------------------------------------------------------------
if "sent_pos" in news.columns:
    print("Aggregating daily sentiment statistics")
    sent_daily = (
        news.groupby(["mack", "date"])
        .agg(
            {
                "sent_pos": ["mean", "std", "max"],
                "sent_neu": "mean",
                "sent_neg": ["mean", "std", "max"],
                "sent_score": ["mean", "std", "min", "max"],
            }
        )
    )
    sent_daily.columns = ["_".join(c) for c in sent_daily.columns]
    sent_daily = sent_daily.reset_index()
    # news count per day
    news_cnt = news.groupby(["mack", "date"]).size().reset_index(name="news_count")
    sent_daily = sent_daily.merge(news_cnt, on=["mack", "date"], how="left")
else:
    sent_daily = pd.DataFrame()
    print("⚠️ No sentiment columns – skipping sentiment aggregation.")

# -------------------------------------------------------------
# 8. MERGE ALL DATA (robust column handling)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("Merging datasets")
print("=" * 60)

# Ensure the key columns exist in each dataframe
for df, name in [(ohlc, "ohlc"), (fund_daily, "fund_daily"), (sent_daily, "sent_daily")]:
    if name == "sent_daily" and df.empty:
        continue
    for col in ["mack", "date"]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing in {name}")

df = ohlc.merge(fund_daily, on=["mack", "date"], how="left")
if not sent_daily.empty:
    df = df.merge(sent_daily, on=["mack", "date"], how="left")

# Debug: show shape after merge
print(f"Dataframe shape after merge: {df.shape}")

# -------------------------------------------------------------
# 9. MISSING VALUE HANDLING
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("Handling missing values")
print("=" * 60)

# Forward‑fill fundamental & sentiment per ticker
fund_cols = [c for c in fund_features if c in df.columns]
sent_cols = [c for c in df.columns if c.startswith("sent_")]

if fund_cols:
    df[fund_cols] = df.groupby("mack", group_keys=False)[fund_cols].ffill()
if sent_cols:
    df[sent_cols] = df.groupby("mack", group_keys=False)[sent_cols].ffill(limit=7)

# Fill remaining NaNs with neutral values
for col in sent_cols:
    if "score" in col:
        df[col] = df[col].fillna(0)
    elif "pos" in col or "neg" in col:
        df[col] = df[col].fillna(0.33)
    else:
        df[col] = df[col].fillna(0)

if "news_count" in df.columns:
    df["news_count"] = df["news_count"].fillna(0)

# Drop rows missing critical price columns
critical = ["close", "sma_20", "rsi"]
before_rows = df.shape[0]
df = df.dropna(subset=critical).reset_index(drop=True)
after_rows = df.shape[0]
print(f"Rows before NA drop: {before_rows:,}, after: {after_rows:,}")

# -------------------------------------------------------------
# 10. SECTOR MAPPING (static dictionary)
# -------------------------------------------------------------
sector_map = {
    # Dầu khí
    "HPG": "Dầu khí", "HSG": "Dầu khí", "NKG": "Dầu khí", "PVT": "Dầu khí", "PVD": "Dầu khí",
    # Nguyên vật liệu
    "GVR": "Nguyên vật liệu", "PHR": "Nguyên vật liệu", "DGC": "Nguyên vật liệu",
    "DCM": "Nguyên vật liệu", "DPM": "Nguyên vật liệu", "BMP": "Nguyên vật liệu",
    "VGC": "Nguyên vật liệu", "HT1": "Nguyên vật liệu",
    # Công nghiệp
    "GEE": "Công nghiệp", "GEX": "Công nghiệp", "HDG": "Công nghiệp", "PC1": "Công nghiệp",
    "REE": "Công nghiệp", "MSN": "Công nghiệp", "VIC": "Công nghiệp", "GMD": "Công nghiệp",
    "VSC": "Công nghiệp", "SCS": "Công nghiệp", "VJC": "Công nghiệp", "VCG": "Công nghiệp",
    "CII": "Công nghiệp", "HHV": "Công nghiệp", "PTB": "Công nghiệp", "CTD": "Công nghiệp",
    "VTP": "Công nghiệp",
    # Hàng tiêu dùng
    "TLG": "Hàng tiêu dùng", "VNM": "Hàng tiêu dùng", "HAG": "Hàng tiêu dùng",
    "SBT": "Hàng tiêu dùng", "KDC": "Hàng tiêu dùng", "VHC": "Hàng tiêu dùng",
    "DBC": "Hàng tiêu dùng", "ANV": "Hàng tiêu dùng", "PAN": "Hàng tiêu dùng",
    "SAB": "Hàng tiêu dùng", "PNJ": "Hàng tiêu dùng",
    # Dược phẩm & Y tế
    "IMP": "Dược phẩm và Y tế",
    # Dịch vụ tiêu dùng
    "MWG": "Dịch vụ Tiêu dùng", "FRT": "Dịch vụ Tiêu dùng", "PLX": "Dịch vụ Tiêu dùng",
    "TCH": "Dịch vụ Tiêu dùng", "DGW": "Dịch vụ Tiêu dùng",
    # Viễn thông
    "CTR": "Viễn thông",
    # Tiện ích cộng đồng
    "POW": "Tiện ích cộng đồng", "NT2": "Tiện ích cộng đồng",
    "PPC": "Tiện ích cộng đồng", "GAS": "Tiện ích cộng đồng", "BWE": "Tiện ích cộng đồng",
    # Tài chính
    "HDC": "Tài chính", "VHM": "Tài chính", "VRE": "Tài chính", "BCM": "Tài chính",
    "KDH": "Tài chính", "KBC": "Tài chính", "DXG": "Tài chính", "PDR": "Tài chính",
    "VPI": "Tài chính", "SJS": "Tài chính", "NLG": "Tài chính", "DIG": "Tài chính",
    "SIP": "Tài chính", "KOS": "Tài chính", "DXS": "Tài chính", "SZC": "Tài chính",
    "SSI": "Tài chính", "VIX": "Tài chính", "VND": "Tài chính", "VCI": "Tài chính",
    "HCM": "Tài chính", "FTS": "Tài chính", "BSI": "Tài chính", "DSE": "Tài chính",
    "CTS": "Tài chính", "BVH": "Tài chính", "EVF": "Tài chính",
    # Ngân hàng
    "VCB": "Ngân hàng", "CTG": "Ngân hàng", "BID": "Ngân hàng", "TCB": "Ngân hàng",
    "VPB": "Ngân hàng", "MBB": "Ngân hàng", "HDB": "Ngân hàng", "LPB": "Ngân hàng",
    "ACB": "Ngân hàng", "STB": "Ngân hàng", "SHB": "Ngân hàng", "VIB": "Ngân hàng",
    "SSB": "Ngân hàng", "TPB": "Ngân hàng", "EIB": "Ngân hàng", "MSB": "Ngân hàng",
    "OCB": "Ngân hàng", "NAB": "Ngân hàng",
    # Công nghệ thông tin
    "FPT": "Công nghệ thông tin", "CMG": "Công nghệ thông tin"
}
df["sector"] = df["mack"].map(sector_map)
df = df.dropna(subset=["sector"]).reset_index(drop=True)
print(f"Sectors used: {df['sector'].nunique()}")

# -------------------------------------------------------------
# TARGET: Quantile-based binary classification
# -------------------------------------------------------------

df["future_return_5d"] = (
    df.groupby("mack")["close"]
      .pct_change(5)
      .shift(-5)
)

q_low = df["future_return_5d"].quantile(0.2)
q_high = df["future_return_5d"].quantile(0.8)


df["target"] = np.where(
    df["future_return_5d"] >= q_high, 1,
    np.where(df["future_return_5d"] <= q_low, 0, np.nan)
)

df = df.dropna(subset=["target"]).reset_index(drop=True)
df["target"] = df["target"].astype(int)


# -------------------------------------------------------------
# 12. FEATURE GROUPS
# -------------------------------------------------------------
fundamental_features = [c for c in fund_features if c in df.columns]

technical_features = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "sma_5", "sma_20", "sma_50",
    "ema_5", "ema_20",
    "bb_upper", "bb_lower", "volume_ratio",
    "volatility_20d"
]
# add lagged returns dynamically
for lag in [1, 3, 5, 10]:
    technical_features.append(f"ret_{lag}d")
    
technical_features = [c for c in technical_features if c in df.columns]

sentiment_features = [c for c in df.columns if c.startswith("sent_")]

feature_groups = {
    "Fundamental": fundamental_features,
    "Technical": technical_features,
    "Sentiment": sentiment_features,
    "Fund+Tech": fundamental_features + technical_features,
    "Fund+Sent": fundamental_features + sentiment_features,
    "Tech+Sent": technical_features + sentiment_features,
    "All": fundamental_features + technical_features + sentiment_features,
}

# -------------------------------------------------------------
# 13. MLP TRAINING
# -------------------------------------------------------------
def train_mlp_time_split(
    df,
    feature_cols,
    target_col="target",
    date_col="date",
    split_date="2023-01-01"
):
    """
    Train MLP with proper time-based split, scaling, and optional SMOTE.
    Returns trained model, scaler, predictions, and evaluation metrics.
    """

    feats = [f for f in feature_cols if f in df.columns]
    if len(feats) < 3:
        return None

    data = df[[date_col] + feats + [target_col]].dropna()
    if data[target_col].nunique() < 2 or len(data) < 200:
        return None

    split_dt = pd.to_datetime(split_date)
    train_mask = data[date_col] < split_dt
    test_mask = ~train_mask

    if test_mask.sum() < 30:
        return None

    X_train = data.loc[train_mask, feats].values
    y_train = data.loc[train_mask, target_col].values
    X_test  = data.loc[test_mask, feats].values
    y_test  = data.loc[test_mask, target_col].values

    # --------------------
    # Scaling (train only)
    # --------------------
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --------------------
    # SMOTE (train only)
    # --------------------
    if IMBLEARN_AVAILABLE and len(np.unique(y_train)) == 2:
        minority = np.min(np.bincount(y_train))
        if minority > 5:
            sm = SMOTE(random_state=RANDOM_STATE)
            X_train_s, y_train = sm.fit_resample(X_train_s, y_train)

    # --------------------
    # MLP model
    # --------------------
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        batch_size=64,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
        verbose=False
    )

    model.fit(X_train_s, y_train)

    # --------------------
    # Evaluation
    # --------------------
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AP": average_precision_score(y_test, y_prob),
    }

    return {
        "model": model,
        "scaler": scaler,
        "features": feats,
        "X_test": X_test_s,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics
    }



# -------------------------------------------------------------
# 14. MAIN LOOP – Ablation Study
# -------------------------------------------------------------

results = []

for name, feats in feature_groups.items():
    res = train_mlp_time_split(df, feats)
    if res is None:
        continue

    results.append({
        "Model": name,
        **res["metrics"]
    })

results_df = pd.DataFrame(results)
results_df.to_csv("model_results.csv", index=False)

print("\nAblation Study Results:")
print(results_df)


# Ablation Study Visualization
plt.figure(figsize=(10, 5))
sns.barplot(
    data=results_df.sort_values("AUC", ascending=False),
    x="Model",
    y="AUC",
    palette="viridis"
)
plt.title("Ablation Study – AUC Comparison Across Feature Sets")
plt.ylabel("AUC")
plt.xlabel("Feature Set")
plt.xticks(rotation=30)
plt.ylim(0.45, 0.75)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "ablation_auc_bar.png"))
plt.show()

# -------------------------------------------------------------
# 15. Rolling Timeseries AUC
# -------------------------------------------------------------

def rolling_timeseries_auc_with_plot(df, features, n_splits=5):
    data = df[features + ["target"]].dropna()
    X = data[features].values
    y = data["target"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []

    for i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        model = MLPClassifier(
            hidden_layer_sizes=(256,128,64,32),
            max_iter=500,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y[train_idx])

        y_prob = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y[test_idx], y_prob)
        aucs.append(auc)

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, n_splits+1), aucs, marker="o")
    plt.xlabel("Fold")
    plt.ylabel("AUC")
    plt.title("Rolling TimeSeriesSplit – AUC per Fold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "rolling_auc.png"))
    plt.show()

    return np.mean(aucs)

mean_auc_rolling = rolling_timeseries_auc_with_plot(
    df, feature_groups["All"]
)
print(f"\nRolling TimeSeriesSplit Mean AUC: {mean_auc_rolling:.3f}")

# -------------------------------------------------------------
# 16. BEST MODEL SELECTION
# -------------------------------------------------------------
# Utility: Permutation Importance (AUC-based)
# -------------------------------------------------------------
from sklearn.inspection import permutation_importance

def permutation_importance_auc(model, X, y, feature_names):
    result = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return (
        pd.DataFrame({
            "Feature": feature_names,
            "Importance": result.importances_mean
        })
        .sort_values("Importance", ascending=False)
    )
# -------------------------------------------------------------
best_row = results_df.sort_values("AUC", ascending=False).iloc[0]
best_model_name = best_row["Model"]
best_features = feature_groups[best_model_name]

print(f"\nBest model selected: {best_model_name}")

best_res = train_mlp_time_split(df, best_features)


# -------------------------------------------------------------
# 16.1 Precision–Recall Curve (Best Model)
# -------------------------------------------------------------
precision, recall, _ = precision_recall_curve(
    best_res["y_test"],
    best_res["y_prob"]
)

plt.figure(figsize=(7, 6))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – Best Model")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "precision_recall_curve.png"))
plt.show()


# -------------------------------------------------------------
# 16.2 Confusion Matrix (Best Model)
# -------------------------------------------------------------
cm = confusion_matrix(
    best_res["y_test"],
    best_res["y_pred"]
)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – Best Model")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "confusion_matrix.png"))
plt.show()


# -------------------------------------------------------------
# 16.3 ROC Curve (Best Model)
# -------------------------------------------------------------
fpr, tpr, _ = roc_curve(best_res["y_test"], best_res["y_prob"])
auc_val = roc_auc_score(best_res["y_test"], best_res["y_prob"])

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"MLP (AUC = {auc_val:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Best Performing Model")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "roc_curve_best_model.png"))
plt.show()


# -------------------------------------------------------------
# 16.4 SHAP Summary Plot (Best Model)
# -------------------------------------------------------------
print("\n📊 16.4 Generating SHAP Analysis...")

# Prepare data for SHAP
X_all = df[best_res["features"]].dropna()
split_idx = int(len(X_all) * 0.8)

X_train_shap = best_res["scaler"].transform(X_all.iloc[:split_idx])
X_test_shap = best_res["scaler"].transform(X_all.iloc[split_idx:split_idx + 200])

# Create background sample and explainer
try:
    background = shap.sample(X_train_shap, 100, random_state=RANDOM_STATE)
    explainer = shap.KernelExplainer(
        best_res["model"].predict_proba,
        background
    )
    
    shap_values = explainer.shap_values(X_test_shap)
    
    # Create DataFrame for plotting
    X_test_df = pd.DataFrame(
        X_test_shap,
        columns=best_res["features"]
    )
    
    # KernelExplainer returns list for classifiers
    if isinstance(shap_values, list):
        shap_vals_plot = shap_values[1]   # class = 1
    else:
        shap_vals_plot = shap_values
    
    # Ensure shape consistency
    shap_vals_plot = shap_vals_plot[:, :X_test_df.shape[1]]
    
    shap.summary_plot(
        shap_vals_plot,
        X_test_df,
        plot_type="bar",
        show=True
    )
    print("✓ SHAP analysis completed")
except Exception as e:
    print(f"⚠️ SHAP analysis failed: {e}")
    shap_values = None
    X_test_shap = None


# -------------------------------------------------------------
# 16.5 Permutation Importance (Best Model)
# -------------------------------------------------------------
# IMPORTANT: dùng đúng tập test của best model
X_test_pi = best_res["scaler"].transform(
    X_all.iloc[split_idx:split_idx + len(best_res["y_test"])].values
)
y_test_pi = best_res["y_test"]

perm_df = permutation_importance_auc(
    best_res["model"],
    best_res["X_test"],     
    best_res["y_test"],     
    best_res["features"]
)

plt.figure(figsize=(8, 6))
sns.barplot(
    data=perm_df.head(15),
    x="Importance",
    y="Feature"
)
plt.title("Permutation Importance (AUC-based) – Best Model")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "permutation_importance.png"))
plt.show()

print("\n✅ Section 16 completed – Permutation Importance ready.")


# =============================================================
# 17. ADDITIONAL ACADEMIC VISUALIZATIONS
# =============================================================
print("\n" + "=" * 60)
print("SECTION 17: Additional Academic Visualizations")
print("=" * 60)

# -------------------------------------------------------------
# 17.1 Feature Correlation Heatmap
# -------------------------------------------------------------
print("\n📊 17.1 Generating Feature Correlation Heatmap...")

all_features = feature_groups["All"]
corr_features = [f for f in all_features if f in df.columns]
corr_matrix = df[corr_features].corr()

plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=False,
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Correlation"}
)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "feature_correlation.png"), dpi=150)
plt.show()
print(f"✓ Saved: {VIS_DIR}/feature_correlation.png")


# -------------------------------------------------------------
# 17.2 Learning Curve (Training Loss)
# -------------------------------------------------------------
print("\n📊 17.2 Generating Learning Curve...")

if hasattr(best_res["model"], "loss_curve_"):
    loss_curve = best_res["model"].loss_curve_
    
    plt.figure(figsize=(10, 6))
    plt.plot(loss_curve, linewidth=2, color="#3498db")
    plt.fill_between(range(len(loss_curve)), loss_curve, alpha=0.3, color="#3498db")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)
    plt.title("MLP Training Loss Curve – Best Model", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    
    # Mark convergence point
    if len(loss_curve) > 10:
        # Find where loss stabilizes (derivative near zero)
        loss_diff = np.diff(loss_curve)
        convergence_idx = len(loss_curve) - 1
        for i in range(len(loss_diff) - 10, -1, -1):
            if abs(loss_diff[i]) > 0.001:
                convergence_idx = i + 10
                break
        plt.axvline(convergence_idx, color="red", linestyle="--", alpha=0.7, 
                    label=f"Convergence ~epoch {convergence_idx}")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "learning_curve.png"), dpi=150)
    plt.show()
    print(f"✓ Saved: {VIS_DIR}/learning_curve.png")
else:
    print("⚠️ Loss curve not available (model may not have converged)")


# -------------------------------------------------------------
# 17.3 Radar Chart – Model Comparison
# -------------------------------------------------------------
print("\n📊 17.3 Generating Radar Chart for Model Comparison...")

def make_radar_chart(df_results, metrics, title, filename):
    """Create radar chart comparing models across multiple metrics."""
    
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(df_results)))
    
    for idx, row in df_results.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row["Model"], color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    
    plt.title(title, size=14, fontweight="bold", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.show()

radar_metrics = ["Accuracy", "AUC", "Precision", "Recall", "F1", "AP"]
make_radar_chart(
    results_df,
    radar_metrics,
    "Model Performance Comparison – Radar Chart",
    "radar_metrics.png"
)
print(f"✓ Saved: {VIS_DIR}/radar_metrics.png")


# -------------------------------------------------------------
# 17.4 Sector-wise Performance Analysis
# -------------------------------------------------------------
print("\n📊 17.4 Generating Sector Performance Analysis...")

sector_results = []
best_features_list = feature_groups[best_model_name]

for sector in df["sector"].unique():
    sector_df = df[df["sector"] == sector]
    if len(sector_df) < 300:
        continue
    
    res = train_mlp_time_split(sector_df, best_features_list)
    if res is not None:
        sector_results.append({
            "Sector": sector,
            "AUC": res["metrics"]["AUC"],
            "Accuracy": res["metrics"]["Accuracy"],
            "F1": res["metrics"]["F1"],
            "Samples": len(sector_df)
        })

if sector_results:
    sector_df_results = pd.DataFrame(sector_results).sort_values("AUC", ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sector_df_results)))
    
    bars = ax.barh(sector_df_results["Sector"], sector_df_results["AUC"], color=colors)
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.7, label="Random (AUC=0.5)")
    ax.axvline(sector_df_results["AUC"].mean(), color="blue", linestyle="--", alpha=0.7, 
               label=f"Mean AUC={sector_df_results['AUC'].mean():.3f}")
    
    # Add value labels
    for bar, auc in zip(bars, sector_df_results["AUC"]):
        ax.text(auc + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{auc:.3f}", va="center", fontsize=9)
    
    ax.set_xlabel("AUC Score", fontsize=12)
    ax.set_title("Model Performance by Sector", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(0.35, 0.85)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "sector_performance.png"), dpi=150)
    plt.show()
    print(f"✓ Saved: {VIS_DIR}/sector_performance.png")
    
    # Save sector results to CSV
    sector_df_results.to_csv("sector_results.csv", index=False)
    print("✓ Saved: sector_results.csv")
else:
    print("⚠️ Not enough data for sector analysis")


# -------------------------------------------------------------
# 17.5 Save SHAP Summary Plot
# -------------------------------------------------------------
print("\n📊 17.5 Saving SHAP Summary Plot...")

if shap_values is not None and X_test_shap is not None:
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_vals_plot,
        X_test_df,
        feature_names=best_res["features"],
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Feature Importance – Best Model", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved: {VIS_DIR}/shap_summary.png")
else:
    print("⚠️ SHAP plot not available (previous SHAP analysis failed)")


# =============================================================
# 18. STATISTICAL ANALYSIS
# =============================================================
print("\n" + "=" * 60)
print("SECTION 18: Statistical Analysis")
print("=" * 60)

# -------------------------------------------------------------
# 18.1 Bootstrap Confidence Intervals
# -------------------------------------------------------------
print("\n📊 18.1 Computing Bootstrap Confidence Intervals for AUC...")

def bootstrap_ci(y_true, y_prob, n_bootstraps=1000, ci=0.95, random_state=42):
    """
    Compute bootstrap confidence interval for AUC.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities for positive class
    n_bootstraps : int
        Number of bootstrap iterations
    ci : float
        Confidence level (e.g., 0.95 for 95% CI)
    
    Returns
    -------
    tuple
        (mean_auc, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(random_state)
    aucs = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        auc = roc_auc_score(y_true[indices], y_prob[indices])
        aucs.append(auc)
    
    aucs = np.array(aucs)
    alpha = (1 - ci) / 2
    lower = np.percentile(aucs, alpha * 100)
    upper = np.percentile(aucs, (1 - alpha) * 100)
    
    return np.mean(aucs), lower, upper

# Compute CI for best model
mean_auc, ci_lower, ci_upper = bootstrap_ci(
    best_res["y_test"],
    best_res["y_prob"]
)

print(f"\nBest Model ({best_model_name}) AUC with 95% CI:")
print(f"   AUC = {mean_auc:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

# Compute CI for all models
ci_results = []
for name, feats in feature_groups.items():
    res = train_mlp_time_split(df, feats)
    if res is None:
        continue
    mean_auc, ci_lower, ci_upper = bootstrap_ci(
        res["y_test"],
        res["y_prob"]
    )
    ci_results.append({
        "Model": name,
        "AUC": mean_auc,
        "CI_Lower": ci_lower,
        "CI_Upper": ci_upper
    })

ci_df = pd.DataFrame(ci_results).sort_values("AUC", ascending=False)
print("\n--- All Models with 95% CI ---")
print(ci_df.to_string(index=False))

# Visualization: AUC with CI
plt.figure(figsize=(10, 6))
x_pos = range(len(ci_df))
plt.bar(x_pos, ci_df["AUC"], yerr=[ci_df["AUC"] - ci_df["CI_Lower"], 
                                     ci_df["CI_Upper"] - ci_df["AUC"]],
        capsize=5, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(ci_df))),
        edgecolor="black", linewidth=1)
plt.xticks(x_pos, ci_df["Model"], rotation=30, ha="right")
plt.ylabel("AUC Score")
plt.title("Model AUC Comparison with 95% Confidence Intervals", fontsize=12, fontweight="bold")
plt.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "auc_confidence_intervals.png"), dpi=150)
plt.show()
print(f"✓ Saved: {VIS_DIR}/auc_confidence_intervals.png")


# -------------------------------------------------------------
# 18.2 Wilcoxon Signed-Rank Test
# -------------------------------------------------------------
print("\n📊 18.2 Performing Wilcoxon Signed-Rank Tests...")

def paired_cv_comparison(df, features1, features2, n_splits=5):
    """
    Perform paired cross-validation for statistical comparison.
    Returns paired AUC scores for each fold.
    """
    all_features = list(set(features1 + features2))
    data = df[all_features + ["target"]].dropna()
    X = data[all_features]
    y = data["target"].values
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs1, aucs2 = [], []
    
    for train_idx, test_idx in tscv.split(X):
        # Model 1
        X1 = data[features1].values
        scaler1 = RobustScaler()
        X1_train = scaler1.fit_transform(X1[train_idx])
        X1_test = scaler1.transform(X1[test_idx])
        
        model1 = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32),
                               max_iter=500, random_state=RANDOM_STATE)
        model1.fit(X1_train, y[train_idx])
        prob1 = model1.predict_proba(X1_test)[:, 1]
        aucs1.append(roc_auc_score(y[test_idx], prob1))
        
        # Model 2
        X2 = data[features2].values
        scaler2 = RobustScaler()
        X2_train = scaler2.fit_transform(X2[train_idx])
        X2_test = scaler2.transform(X2[test_idx])
        
        model2 = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32),
                               max_iter=500, random_state=RANDOM_STATE)
        model2.fit(X2_train, y[train_idx])
        prob2 = model2.predict_proba(X2_test)[:, 1]
        aucs2.append(roc_auc_score(y[test_idx], prob2))
    
    return np.array(aucs1), np.array(aucs2)


# Compare best model vs others
print(f"\nStatistical comparison: {best_model_name} vs other models")
print("-" * 60)

stat_results = []
for name, feats in feature_groups.items():
    if name == best_model_name:
        continue
    
    try:
        aucs_best, aucs_other = paired_cv_comparison(
            df, 
            feature_groups[best_model_name], 
            feats,
            n_splits=5
        )
        
        # Wilcoxon test (two-sided)
        stat, p_value = stats.wilcoxon(aucs_best, aucs_other)
        
        # Effect size (mean difference)
        mean_diff = np.mean(aucs_best - aucs_other)
        
        stat_results.append({
            "Comparison": f"{best_model_name} vs {name}",
            "Best_Mean": np.mean(aucs_best),
            "Other_Mean": np.mean(aucs_other),
            "Difference": mean_diff,
            "p-value": p_value,
            "Significant": "Yes" if p_value < 0.05 else "No"
        })
        
        significance = "✓" if p_value < 0.05 else "✗"
        print(f"{best_model_name} vs {name}: Δ={mean_diff:+.4f}, p={p_value:.4f} {significance}")
        
    except Exception as e:
        print(f"Could not compare with {name}: {e}")

if stat_results:
    stat_df = pd.DataFrame(stat_results)
    stat_df.to_csv("statistical_tests.csv", index=False)
    print(f"\n✓ Saved: statistical_tests.csv")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(stat_df))
    colors = ["#2ecc71" if p < 0.05 else "#e74c3c" for p in stat_df["p-value"]]
    
    ax.bar(x, stat_df["Difference"], color=colors, edgecolor="black", linewidth=1)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(f"{best_model_name} vs ", "") for c in stat_df["Comparison"]], 
                       rotation=30, ha="right")
    ax.set_ylabel("AUC Difference")
    ax.set_title(f"Statistical Comparison: {best_model_name} vs Others\n(Green = p<0.05, Red = p≥0.05)", 
                 fontsize=12, fontweight="bold")
    
    # Add p-value labels
    for i, (diff, p) in enumerate(zip(stat_df["Difference"], stat_df["p-value"])):
        ax.text(i, diff + 0.002, f"p={p:.3f}", ha="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "statistical_comparison.png"), dpi=150)
    plt.show()
    print(f"✓ Saved: {VIS_DIR}/statistical_comparison.png")


# -------------------------------------------------------------
# 18.3 Feature Combination Analysis (ABLATION STUDY)
# -------------------------------------------------------------
print("\n📊 18.3 Feature Combination Analysis...")
print("-" * 60)

# Sort by AUC for ranking
results_sorted = results_df.sort_values("AUC", ascending=False).reset_index(drop=True)
results_sorted["Rank"] = range(1, len(results_sorted) + 1)

print("\n╔════════════════════════════════════════════════════════════════════════╗")
print("║             ABLATION STUDY - FEATURE COMBINATION ANALYSIS             ║")
print("╠════════════════════════════════════════════════════════════════════════╣")
print(f"║ {'Rank':<4} │ {'Model':<15} │ {'Accuracy':<10} │ {'AUC':<10} │ {'F1':<10} ║")
print("╠════════════════════════════════════════════════════════════════════════╣")

for _, row in results_sorted.iterrows():
    print(f"║ {int(row['Rank']):<4} │ {row['Model']:<15} │ {row['Accuracy']:<10.4f} │ {row['AUC']:<10.4f} │ {row['F1']:<10.4f} ║")

print("╚════════════════════════════════════════════════════════════════════════╝")

# Detailed interpretation
print("\n" + "=" * 60)
print("ACADEMIC INTERPRETATION")
print("=" * 60)

# Get metrics for each model type
fundamental_only = results_df[results_df["Model"] == "Fundamental"].iloc[0] if "Fundamental" in results_df["Model"].values else None
technical_only = results_df[results_df["Model"] == "Technical"].iloc[0] if "Technical" in results_df["Model"].values else None
sentiment_only = results_df[results_df["Model"] == "Sentiment"].iloc[0] if "Sentiment" in results_df["Model"].values else None
fund_tech = results_df[results_df["Model"] == "Fund+Tech"].iloc[0] if "Fund+Tech" in results_df["Model"].values else None
fund_sent = results_df[results_df["Model"] == "Fund+Sent"].iloc[0] if "Fund+Sent" in results_df["Model"].values else None
tech_sent = results_df[results_df["Model"] == "Tech+Sent"].iloc[0] if "Tech+Sent" in results_df["Model"].values else None
all_features = results_df[results_df["Model"] == "All"].iloc[0] if "All" in results_df["Model"].values else None

print("\n┌──────────────────────────────────────────────────────────────────────┐")
print("│ 1. SINGLE FEATURE GROUP PERFORMANCE                                 │")
print("└──────────────────────────────────────────────────────────────────────┘")

if fundamental_only is not None:
    print(f"   • Fundamental: AUC={fundamental_only['AUC']:.4f}, Acc={fundamental_only['Accuracy']:.4f}")
if technical_only is not None:
    print(f"   • Technical:   AUC={technical_only['AUC']:.4f}, Acc={technical_only['Accuracy']:.4f}")
if sentiment_only is not None:
    print(f"   • Sentiment:   AUC={sentiment_only['AUC']:.4f}, Acc={sentiment_only['Accuracy']:.4f}")

# Find best single feature group
single_models = results_df[results_df["Model"].isin(["Fundamental", "Technical", "Sentiment"])]
if len(single_models) > 0:
    best_single = single_models.loc[single_models["AUC"].idxmax()]
    print(f"\n   → Best single feature group: {best_single['Model']} (AUC={best_single['AUC']:.4f})")

print("\n┌──────────────────────────────────────────────────────────────────────┐")
print("│ 2. COMBINATION EFFECTS                                              │")
print("└──────────────────────────────────────────────────────────────────────┘")

# Analyze combination effects
if fund_tech is not None and fundamental_only is not None and technical_only is not None:
    baseline = max(fundamental_only['AUC'], technical_only['AUC'])
    improvement = fund_tech['AUC'] - baseline
    effect = "SYNERGY ↑" if improvement > 0.005 else ("NEUTRAL =" if abs(improvement) <= 0.005 else "INTERFERENCE ↓")
    print(f"   • Fund+Tech:   AUC={fund_tech['AUC']:.4f} | vs best single: {improvement:+.4f} ({effect})")

if fund_sent is not None and fundamental_only is not None and sentiment_only is not None:
    baseline = max(fundamental_only['AUC'], sentiment_only['AUC'])
    improvement = fund_sent['AUC'] - baseline
    effect = "SYNERGY ↑" if improvement > 0.005 else ("NEUTRAL =" if abs(improvement) <= 0.005 else "INTERFERENCE ↓")
    print(f"   • Fund+Sent:   AUC={fund_sent['AUC']:.4f} | vs best single: {improvement:+.4f} ({effect})")

if tech_sent is not None and technical_only is not None and sentiment_only is not None:
    baseline = max(technical_only['AUC'], sentiment_only['AUC'])
    improvement = tech_sent['AUC'] - baseline
    effect = "SYNERGY ↑" if improvement > 0.005 else ("NEUTRAL =" if abs(improvement) <= 0.005 else "INTERFERENCE ↓")
    print(f"   • Tech+Sent:   AUC={tech_sent['AUC']:.4f} | vs best single: {improvement:+.4f} ({effect})")

print("\n┌──────────────────────────────────────────────────────────────────────┐")
print("│ 3. ALL FEATURES PERFORMANCE                                         │")
print("└──────────────────────────────────────────────────────────────────────┘")

if all_features is not None:
    best_combination = results_df[results_df["Model"].isin(["Fund+Tech", "Fund+Sent", "Tech+Sent"])]
    if len(best_combination) > 0:
        best_combo = best_combination.loc[best_combination["AUC"].idxmax()]
        improvement = all_features['AUC'] - best_combo['AUC']
        print(f"   • All Features: AUC={all_features['AUC']:.4f}, Acc={all_features['Accuracy']:.4f}")
        print(f"   • vs Best Pair ({best_combo['Model']}): {improvement:+.4f}")
        
        if improvement > 0.005:
            print("   → Adding all features provides ADDITIONAL PREDICTIVE POWER")
        elif improvement < -0.005:
            print("   → Adding all features causes OVERFITTING or NOISE INTERFERENCE")
        else:
            print("   → All features provide MARGINAL IMPROVEMENT")

print("\n┌──────────────────────────────────────────────────────────────────────┐")
print("│ 4. KEY FINDINGS & CONCLUSIONS                                       │")
print("└──────────────────────────────────────────────────────────────────────┘")

best_overall = results_sorted.iloc[0]
worst_overall = results_sorted.iloc[-1]

print(f"""
   ★ BEST MODEL: {best_overall['Model']}
     - AUC: {best_overall['AUC']:.4f}
     - Accuracy: {best_overall['Accuracy']:.4f}
     - F1 Score: {best_overall['F1']:.4f}

   ✗ WORST MODEL: {worst_overall['Model']}
     - AUC: {worst_overall['AUC']:.4f}
     - Performance gap: {best_overall['AUC'] - worst_overall['AUC']:.4f}
""")

# Determine conclusion based on results
if best_overall['AUC'] > 0.55:
    conclusion = "Models show MODERATE predictive power above random baseline."
elif best_overall['AUC'] > 0.52:
    conclusion = "Models show SLIGHT predictive edge over random baseline."
else:
    conclusion = "Models perform NEAR RANDOM - features may not contain strong predictive signals."

print(f"   📌 CONCLUSION: {conclusion}")
print()

# Save detailed analysis to CSV
analysis_df = results_sorted[["Rank", "Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "AP"]]
analysis_df.to_csv("feature_combination_analysis.csv", index=False)
print("✓ Saved: feature_combination_analysis.csv")

# Create comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: AUC Comparison
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(results_sorted)))[::-1]
bars1 = axes[0].barh(results_sorted["Model"], results_sorted["AUC"], color=colors)
axes[0].axvline(0.5, color="red", linestyle="--", alpha=0.7, label="Random (0.5)")
axes[0].set_xlabel("AUC Score")
axes[0].set_title("Feature Combination AUC Comparison", fontweight="bold")
axes[0].legend()
# Add value labels
for bar, val in zip(bars1, results_sorted["AUC"]):
    axes[0].text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                 f"{val:.3f}", va="center", fontsize=9)

# Plot 2: Accuracy Comparison
bars2 = axes[1].barh(results_sorted["Model"], results_sorted["Accuracy"], color=colors)
axes[1].axvline(0.5, color="red", linestyle="--", alpha=0.7, label="Random (0.5)")
axes[1].set_xlabel("Accuracy Score")
axes[1].set_title("Feature Combination Accuracy Comparison", fontweight="bold")
axes[1].legend()
# Add value labels
for bar, val in zip(bars2, results_sorted["Accuracy"]):
    axes[1].text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                 f"{val:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "feature_combination_comparison.png"), dpi=150)
plt.show()
print(f"✓ Saved: {VIS_DIR}/feature_combination_comparison.png")


# =============================================================
# 19. FINAL SUMMARY
# =============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    MLP STOCK PREDICTION                       ║
║                    ANALYSIS COMPLETE                          ║
╠══════════════════════════════════════════════════════════════╣
║  Best Model: {best_model_name:<20}                            ║
║  AUC Score:  {best_row['AUC']:.4f} [{ci_df[ci_df['Model']==best_model_name]['CI_Lower'].values[0]:.4f}, {ci_df[ci_df['Model']==best_model_name]['CI_Upper'].values[0]:.4f}]              ║
║  Accuracy:   {best_row['Accuracy']:.4f}                                      ║
║  F1 Score:   {best_row['F1']:.4f}                                      ║
╠══════════════════════════════════════════════════════════════╣
║  Visualizations saved to: {VIS_DIR}/                        ║
║  - ablation_auc_bar.png                                       ║
║  - rolling_auc.png                                            ║
║  - precision_recall_curve.png                                 ║
║  - confusion_matrix.png                                       ║
║  - roc_curve_best_model.png                                   ║
║  - permutation_importance.png                                 ║
║  - feature_correlation.png                                    ║
║  - learning_curve.png                                         ║
║  - radar_metrics.png                                          ║
║  - sector_performance.png                                     ║
║  - shap_summary.png                                           ║
║  - auc_confidence_intervals.png                               ║
║  - statistical_comparison.png                                 ║
╠══════════════════════════════════════════════════════════════╣
║  CSV Reports:                                                 ║
║  - model_results.csv                                          ║
║  - sector_results.csv                                         ║
║  - statistical_tests.csv                                      ║
╚══════════════════════════════════════════════════════════════╝
""")

print("✅ All analysis completed successfully!")
