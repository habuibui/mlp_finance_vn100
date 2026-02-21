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
from sklearn.neural_network import MLPClassifier, MLPRegressor
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
    mean_squared_error,
    mean_absolute_error,
    r2_score,
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

# --- NO LEAKAGE: all indicators use .shift(1) so at row t we only use info up to end of day t-1 ---
# Simple moving averages: rolling(w).mean().shift(1)
for w in [5, 10, 20, 50]:
    ohlc[f"sma_{w}"] = ohlc.groupby("mack")["close"].transform(lambda x: x.rolling(w).mean().shift(1))

# Exponential moving averages
for w in [5, 10, 20, 50]:
    ohlc[f"ema_{w}"] = ohlc.groupby("mack")["close"].transform(lambda x: x.ewm(span=w, adjust=False).mean().shift(1))

# RSI (computed then shifted so we don't use close_t)
ohlc["rsi"] = ohlc.groupby("mack")["close"].transform(compute_rsi)
ohlc["rsi"] = ohlc.groupby("mack")["rsi"].shift(1)

# MACD (all shifted)
ema_12 = ohlc.groupby("mack")["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean().shift(1))
ema_26 = ohlc.groupby("mack")["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean().shift(1))
ohlc["macd"] = ema_12 - ema_26
ohlc["macd_signal"] = ohlc.groupby("mack")["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean().shift(1))
ohlc["macd_hist"] = ohlc["macd"] - ohlc["macd_signal"]

# Bollinger Bands (20‑day): use shifted values
_bb_mid = ohlc.groupby("mack")["close"].transform(lambda x: x.rolling(20).mean().shift(1))
_bb_std = ohlc.groupby("mack")["close"].transform(lambda x: x.rolling(20).std().shift(1))
ohlc["bb_mid"] = _bb_mid
ohlc["bb_std"] = _bb_std
ohlc["bb_upper"] = ohlc["bb_mid"] + 2 * ohlc["bb_std"]
ohlc["bb_lower"] = ohlc["bb_mid"] - 2 * ohlc["bb_std"]

# Volume ratio (shifted)
vol = ohlc.groupby("mack")["volume"]
ohlc["vol_ma_20"] = vol.transform(lambda x: x.rolling(20).mean().shift(1))
ohlc["volume_ratio"] = ohlc["volume"] / ohlc["vol_ma_20"].replace(0, np.nan)

# Momentum / Lagged returns: use past returns only (shift(1) so no close_t in feature)
for lag in [1, 3, 5, 10]:
    ohlc[f"ret_{lag}d"] = ohlc.groupby("mack")["close"].transform(lambda x: x.pct_change(lag).shift(1))

# Volatility (past 20d only)
ohlc["volatility_20d"] = ohlc.groupby("mack")["close"].transform(lambda x: x.pct_change().rolling(20).std().shift(1))

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
# TARGET: Future log return (khuyến nghị) | price | direction | classification
# -------------------------------------------------------------
# "return" = regression future log return (ý nghĩa kinh tế, tránh R² thổi phồng từ price).
# Horizon: 1, 5, 10 ngày.
TARGET_MODE = "return"  # "return" | "price" | "direction" | "quantile" | "binary" | "fixed_pct"
HORIZON_DAYS = 1       # 1, 5, hoặc 10 (cho return/price/direction)
TARGET_QUANTILE_LOW = 0.3
TARGET_QUANTILE_HIGH = 0.7
TARGET_FIXED_THRESHOLD_PCT = 0.01
DIRECTION_FLAT_THRESHOLD = 1e-6
SIGNAL_THRESHOLD_PCT = 0.0

# Giá ngày mai (cho price mode)
df["target_price"] = df.groupby("mack")["close"].shift(-1)
# Lợi suất tương lai (simple return)
df["future_return_1d"] = df.groupby("mack")["close"].pct_change(1).shift(-1)
df["future_return_5d"] = df.groupby("mack")["close"].pct_change(5).shift(-5)
df["future_return_10d"] = df.groupby("mack")["close"].pct_change(10).shift(-10)
# Log return (khuyến nghị): log(P_{t+h}/P_t), không dùng Close_t trong target
df["future_log_return_1d"] = np.log(df.groupby("mack")["close"].shift(-1) / df["close"])
df["future_log_return_5d"] = np.log(df.groupby("mack")["close"].shift(-5) / df["close"])
df["future_log_return_10d"] = np.log(df.groupby("mack")["close"].shift(-10) / df["close"])
df_before_target_drop = df.copy()  # Để dùng cho multi-horizon (1d, 5d, 10d)

rows_before_target = len(df)

if TARGET_MODE == "return":
    # Target = future log return (h ngày). Có ý nghĩa kinh tế, đánh giá đúng khả năng dự báo.
    col_log = f"future_log_return_{HORIZON_DAYS}d"
    df["target"] = df[col_log]
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    rows_used = len(df)
    print(f"Target mode: {TARGET_MODE} (Regression: future log return {HORIZON_DAYS}d)")
    print(f"  Rows used: {len(df):,} | target mean={df['target'].mean():.6f}, std={df['target'].std():.6f}")
    USE_REGRESSION = True
    REGRESSION_TARGET = "return"
elif TARGET_MODE == "price":
    df = df.dropna(subset=["target_price"]).reset_index(drop=True)
    df["target"] = df["target_price"]
    rows_used = len(df)
    print(f"Target mode: {TARGET_MODE} (Regression: giá ngày mai)")
    print(f"  Rows used: {len(df):,}")
    USE_REGRESSION = True
    REGRESSION_TARGET = "price"
elif TARGET_MODE == "direction":
    ret_col = f"future_return_{HORIZON_DAYS}d" if f"future_return_{HORIZON_DAYS}d" in df.columns else "future_return_1d"
    ret = df[ret_col]
    df["target"] = np.where(ret > DIRECTION_FLAT_THRESHOLD, 1, np.where(ret < -DIRECTION_FLAT_THRESHOLD, -1, 0))
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    print(f"Target mode: {TARGET_MODE} (regression: -1/0/1)")
    print(f"  Rows used: {len(df):,} | -1={(df['target']==-1).sum():,}, 0={(df['target']==0).sum():,}, 1={(df['target']==1).sum():,}")
    USE_REGRESSION = True
    REGRESSION_TARGET = "direction"
else:
    ret_col = "future_return_5d"
    rows_before_target = len(df.dropna(subset=[ret_col]))
    if TARGET_MODE == "quantile":
        q_low = df[ret_col].quantile(TARGET_QUANTILE_LOW)
        q_high = df[ret_col].quantile(TARGET_QUANTILE_HIGH)
        df["target"] = np.where(df[ret_col] >= q_high, 1, np.where(df[ret_col] <= q_low, 0, np.nan))
    elif TARGET_MODE == "binary":
        df["target"] = (df[ret_col] > TARGET_FIXED_THRESHOLD_PCT).astype(float)
        df.loc[df[ret_col].isna(), "target"] = np.nan
    elif TARGET_MODE == "fixed_pct":
        df["target"] = np.where(
            df[ret_col] > TARGET_FIXED_THRESHOLD_PCT, 1,
            np.where(df[ret_col] < -TARGET_FIXED_THRESHOLD_PCT, 0, np.nan)
        )
    else:
        raise ValueError(f"Unknown TARGET_MODE: {TARGET_MODE}")
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    print(f"Target mode: {TARGET_MODE} | Rows used: {len(df):,}")
    print(f"  Class distribution: 0={(df['target']==0).sum():,}, 1={(df['target']==1).sum():,}")
    USE_REGRESSION = False
    REGRESSION_TARGET = None

# Cột target khi train
TARGET_COL = "target_price" if (USE_REGRESSION and REGRESSION_TARGET == "price") else "target"


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
# 13. MLP TRAINING (Classifier hoặc Regressor theo TARGET_MODE)
# -------------------------------------------------------------
def _round_to_direction(y_continuous):
    """Làm tròn dự báo liên tục về lớp -1, 0, 1 (gần nhất)."""
    y = np.clip(y_continuous, -1, 1)
    return np.where(y < -0.5, -1, np.where(y > 0.5, 1, 0)).astype(int)


def train_mlp_time_split(
    df,
    feature_cols,
    target_col="target",
    date_col="date",
    split_date="2023-01-01",
    use_regression=False,
    regression_target="return",  # "return" | "direction" | "price"
    signal_threshold_pct=0.0,
):
    """
    Train MLP với time-based split. Scaler chỉ fit trên X_train (không leakage).
    - regression_target="return": target = future log return → R2/MAE/RMSE + Direction Acc/Precision/Recall/F1/AUC.
    - regression_target="direction": target -1/0/1.
    - regression_target="price": target = giá ngày mai.
    """

    feats = [f for f in feature_cols if f in df.columns]
    if len(feats) < 3:
        return None

    extra_cols = ["close"] if (use_regression and regression_target == "price") else []
    data = df[[date_col] + feats + [target_col] + extra_cols].dropna()
    if len(data) < 200:
        return None
    if not use_regression and data[target_col].nunique() < 2:
        return None

    split_dt = pd.to_datetime(split_date)
    train_mask = data[date_col] < split_dt
    test_mask = ~train_mask
    if test_mask.sum() < 30:
        return None

    X_train = data.loc[train_mask, feats].values
    y_train = data.loc[train_mask, target_col].values
    X_test = data.loc[test_mask, feats].values
    y_test = data.loc[test_mask, target_col].values

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)  # Chỉ fit trên train, không fit toàn bộ
    X_test_s = scaler.transform(X_test)

    if use_regression:
        model = MLPRegressor(
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
            verbose=False,
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s).ravel()
        y_prob = None

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Direction từ return: 1 nếu > 0, 0 nếu <= 0 (binary cho Precision/Recall/F1/AUC)
        pred_return = true_return = None
        if regression_target in ("return", "price"):
            if regression_target == "price":
                close_today = data.loc[test_mask, "close"].values.astype(float)
                pred_return = (y_pred - close_today) / np.maximum(close_today, 1e-8)
                true_return = (y_test - close_today) / np.maximum(close_today, 1e-8)
            else:
                pred_return = y_pred
                true_return = y_test
            signal_pred = (pred_return > signal_threshold_pct).astype(int)
            signal_true = (true_return > signal_threshold_pct).astype(int)
            y_pred_class = signal_pred
            y_test_class = signal_true
            acc_signal = accuracy_score(signal_true, signal_pred)
            if signal_true.sum() > 0 and signal_true.sum() < len(signal_true):
                prec = precision_score(signal_true, signal_pred, zero_division=0)
                rec = recall_score(signal_true, signal_pred, zero_division=0)
                f1 = f1_score(signal_true, signal_pred, zero_division=0)
                try:
                    auc_dir = roc_auc_score(signal_true, pred_return)
                except Exception:
                    auc_dir = 0.5
            else:
                prec = rec = f1 = 0.0
                auc_dir = 0.5
        else:
            y_pred_class = _round_to_direction(y_pred)
            y_test_class = y_test
            acc_signal = accuracy_score(y_test, y_pred_class)
            prec = rec = f1 = auc_dir = np.nan

        metrics = {
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "Accuracy_direction": acc_signal,
        }
        if regression_target == "return":
            metrics["Precision"] = prec
            metrics["Recall"] = rec
            metrics["F1"] = f1
            metrics["AUC_direction"] = auc_dir
        return {
            "model": model,
            "scaler": scaler,
            "features": feats,
            "X_test": X_test_s,
            "y_test": y_test,
            "y_test_class": y_test_class,
            "y_pred": y_pred,
            "y_pred_class": y_pred_class,
            "y_prob": y_prob,
            "metrics": metrics,
            "date_test": data.loc[test_mask, date_col].values if date_col in data.columns else None,
            "pred_return": pred_return,
            "true_return": true_return,
        }
    else:
        # --------------------
        # MLP Classifier (binary 0/1)
        # --------------------
        if IMBLEARN_AVAILABLE and len(np.unique(y_train)) == 2:
            minority = np.min(np.bincount(y_train.astype(int)))
            if minority > 5:
                sm = SMOTE(random_state=RANDOM_STATE)
                X_train_s, y_train = sm.fit_resample(X_train_s, y_train)

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
            verbose=False,
        )
        model.fit(X_train_s, y_train)
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
            "metrics": metrics,
        }



# -------------------------------------------------------------
# 14. MAIN LOOP – Ablation Study
# -------------------------------------------------------------

results = []
for name, feats in feature_groups.items():
    res = train_mlp_time_split(
        df, feats,
        target_col=TARGET_COL,
        use_regression=USE_REGRESSION,
        regression_target=REGRESSION_TARGET if USE_REGRESSION else "direction",
        signal_threshold_pct=SIGNAL_THRESHOLD_PCT,
    )
    if res is None:
        continue
    results.append({"Model": name, **res["metrics"]})

# -------------------------------------------------------------
# 14.1 BASELINES (Random Walk, AR(1), Naive return=0) – bắt buộc so sánh
# -------------------------------------------------------------
def run_baselines(df, target_col="target", date_col="date", split_date="2023-01-01", signal_threshold_pct=0.0):
    """Baseline dự báo trên cùng train/test split. Scaler không dùng cho baseline (chỉ so sánh target)."""
    data = df[[date_col, target_col]].copy()
    if "ret_1d" in df.columns:
        data["ret_1d"] = df["ret_1d"].values
    data = data.dropna()
    if len(data) < 200:
        return []
    split_dt = pd.to_datetime(split_date)
    train_mask = data[date_col] < split_dt
    test_mask = ~train_mask
    if test_mask.sum() < 30:
        return []
    y_train = data.loc[train_mask, target_col].values
    y_test = data.loc[test_mask, target_col].values

    sig_true = (y_test > signal_threshold_pct).astype(int)

    # Random Walk / Naive: P_{t+1} = P_t → predicted return = 0
    y_pred_rw = np.zeros_like(y_test)
    r2_rw = r2_score(y_test, y_pred_rw)
    rmse_rw = np.sqrt(mean_squared_error(y_test, y_pred_rw))
    mae_rw = mean_absolute_error(y_test, y_pred_rw)
    sig_rw = (y_pred_rw > signal_threshold_pct).astype(int)
    acc_rw = accuracy_score(sig_true, sig_rw)
    prec_rw = precision_score(sig_true, sig_rw, zero_division=0)
    rec_rw = recall_score(sig_true, sig_rw, zero_division=0)
    f1_rw = f1_score(sig_true, sig_rw, zero_division=0)
    try:
        auc_rw = roc_auc_score(sig_true, y_pred_rw)
    except Exception:
        auc_rw = 0.5
    baseline_rows = [
        {"Model": "Random Walk", "R2": r2_rw, "RMSE": rmse_rw, "MAE": mae_rw, "Accuracy_direction": acc_rw,
         "Precision": prec_rw, "Recall": rec_rw, "F1": f1_rw, "AUC_direction": auc_rw},
    ]

    # AR(1): r_{t+1} = phi * r_t, fit phi on train (r_t = ret_1d at t)
    if "ret_1d" in data.columns:
        r_train = data.loc[train_mask, "ret_1d"].values
        r_test = data.loc[test_mask, "ret_1d"].values
        phi = np.cov(y_train, r_train)[0, 1] / (np.var(r_train) + 1e-12)
        y_pred_ar1 = phi * r_test
        r2_ar1 = r2_score(y_test, y_pred_ar1)
        rmse_ar1 = np.sqrt(mean_squared_error(y_test, y_pred_ar1))
        mae_ar1 = mean_absolute_error(y_test, y_pred_ar1)
        sig_ar1 = (y_pred_ar1 > signal_threshold_pct).astype(int)
        acc_ar1 = accuracy_score(sig_true, sig_ar1)
        prec_ar1 = precision_score(sig_true, sig_ar1, zero_division=0)
        rec_ar1 = recall_score(sig_true, sig_ar1, zero_division=0)
        f1_ar1 = f1_score(sig_true, sig_ar1, zero_division=0)
        try:
            auc_ar1 = roc_auc_score(sig_true, y_pred_ar1)
        except Exception:
            auc_ar1 = 0.5
        baseline_rows.append({
            "Model": "AR(1)", "R2": r2_ar1, "RMSE": rmse_ar1, "MAE": mae_ar1, "Accuracy_direction": acc_ar1,
            "Precision": prec_ar1, "Recall": rec_ar1, "F1": f1_ar1, "AUC_direction": auc_ar1,
        })
    return baseline_rows

if USE_REGRESSION:
    baseline_results = run_baselines(df, target_col=TARGET_COL, split_date="2023-01-01", signal_threshold_pct=SIGNAL_THRESHOLD_PCT)
    for row in baseline_results:
        results.append(row)
    # Ensure all result rows have same keys (fill missing with nan)
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    for r in results:
        for k in all_keys:
            if k not in r:
                r[k] = np.nan

results_df = pd.DataFrame(results)
results_df.to_csv("model_results.csv", index=False)

print("\nAblation Study + Baselines:")
print(results_df)

# Ablation Visualization
if USE_REGRESSION:
    sort_col = "R2"
    results_sorted = results_df.sort_values(sort_col, ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_sorted, x="Model", y="R2", palette="viridis")
    plt.title("Ablation Study – R² Comparison Across Feature Sets (Direction Regression)")
    plt.ylabel("R²")
    plt.xlabel("Feature Set")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "ablation_r2_bar.png"))
    plt.show()
else:
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=results_df.sort_values("AUC", ascending=False),
        x="Model", y="AUC", palette="viridis"
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
# 15. Rolling Timeseries (AUC hoặc R2 theo task)
# -------------------------------------------------------------

def rolling_timeseries_metric_with_plot(df, features, n_splits=5, use_regression=False):
    data = df[features + ["target"]].dropna()
    X = data[features].values
    y = data["target"].values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, test_idx in tscv.split(X):
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        if use_regression:
            model = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64, 32),
                max_iter=500, random_state=RANDOM_STATE,
            )
            model.fit(X_train, y[train_idx])
            y_pred = model.predict(X_test)
            scores.append(r2_score(y[test_idx], y_pred))
        else:
            model = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64, 32),
                max_iter=500, random_state=RANDOM_STATE,
            )
            model.fit(X_train, y[train_idx])
            y_prob = model.predict_proba(X_test)[:, 1]
            scores.append(roc_auc_score(y[test_idx], y_prob))

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, n_splits + 1), scores, marker="o")
    plt.xlabel("Fold")
    plt.ylabel("R²" if use_regression else "AUC")
    plt.title("Rolling TimeSeriesSplit – " + ("R²" if use_regression else "AUC") + " per Fold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "rolling_r2.png" if use_regression else "rolling_auc.png"))
    plt.show()
    return np.mean(scores)

mean_rolling = rolling_timeseries_metric_with_plot(
    df, feature_groups["All"], use_regression=USE_REGRESSION
)
print(f"\nRolling TimeSeriesSplit Mean {'R²' if USE_REGRESSION else 'AUC'}: {mean_rolling:.3f}")

# -------------------------------------------------------------
# 16. BEST MODEL SELECTION
# -------------------------------------------------------------
from sklearn.inspection import permutation_importance

def permutation_importance_scoring(model, X, y, feature_names, use_regression=False):
    scoring = "r2" if use_regression else "roc_auc"
    result = permutation_importance(
        model, X, y, scoring=scoring,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    return (
        pd.DataFrame({"Feature": feature_names, "Importance": result.importances_mean})
        .sort_values("Importance", ascending=False)
    )

sort_col = "R2" if USE_REGRESSION else "AUC"
# Best ML model (exclude baselines for training)
ml_models = [m for m in results_df["Model"] if m in feature_groups]
results_ml = results_df[results_df["Model"].isin(ml_models)]
best_row = results_ml.sort_values(sort_col, ascending=(not USE_REGRESSION)).iloc[0]
best_model_name = best_row["Model"]
best_features = feature_groups[best_model_name]
print(f"\nBest ML model selected: {best_model_name}")

best_res = train_mlp_time_split(
    df, best_features,
    target_col=TARGET_COL,
    use_regression=USE_REGRESSION,
    regression_target=REGRESSION_TARGET if USE_REGRESSION else "direction",
    signal_threshold_pct=SIGNAL_THRESHOLD_PCT,
)

# ---------- 16.1 & 16.2 & 16.3: Regression => scatter + residuals + confusion (rounded); Classification => PR + CM + ROC
if USE_REGRESSION:
    # Scatter: Predicted vs Actual (giá hoặc direction)
    plt.figure(figsize=(7, 6))
    plt.scatter(best_res["y_test"], best_res["y_pred"], alpha=0.4, s=15)
    mn, mx = best_res["y_test"].min(), best_res["y_test"].max()
    plt.plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect")
    if REGRESSION_TARGET == "price":
        plt.xlabel("Actual price (next day)")
        plt.ylabel("Predicted price (next day)")
        plt.title(f"Best Model – P_pred vs P_actual (R²={best_res['metrics']['R2']:.4f})")
    elif REGRESSION_TARGET == "return":
        plt.xlabel("Actual future log return")
        plt.ylabel("Predicted log return")
        plt.title(f"Best Model – Predicted vs Actual return (R²={best_res['metrics']['R2']:.4f})")
    else:
        plt.xlabel("Actual (direction -1/0/1)")
        plt.ylabel("Predicted (continuous)")
        plt.title(f"Best Model – Predicted vs Actual (R²={best_res['metrics']['R2']:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "regression_scatter.png"))
    plt.show()
    # Residuals
    plt.figure(figsize=(7, 5))
    residuals = best_res["y_test"] - best_res["y_pred"]
    plt.hist(residuals, bins=50, edgecolor="white", color="steelblue")
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Residuals – Best Model")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "regression_residuals.png"))
    plt.show()
    # Confusion matrix: tín hiệu (price: 0/1 Mua-Bán; direction: -1/0/1)
    y_pred_class = best_res.get("y_pred_class", _round_to_direction(best_res["y_pred"]))
    y_test_class = best_res.get("y_test_class", best_res["y_test"])
    cm = confusion_matrix(y_test_class, y_pred_class)
    plt.figure(figsize=(5, 4))
    if REGRESSION_TARGET == "price":
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Bán/Đứng", "Mua"], yticklabels=["Bán/Đứng", "Mua"])
        plt.title("Confusion Matrix (tín hiệu từ P_pred vs P_today) – Best Model")
    elif REGRESSION_TARGET == "return":
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Không mua", "Mua"], yticklabels=["Không mua", "Mua"])
        plt.title("Confusion Matrix (tín hiệu từ predicted return > 0) – Best Model")
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[-1, 0, 1], yticklabels=[-1, 0, 1])
        plt.title("Confusion Matrix (rounded prediction) – Best Model")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "confusion_matrix.png"))
    plt.show()
else:
    precision, recall, _ = precision_recall_curve(best_res["y_test"], best_res["y_prob"])
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve – Best Model")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "precision_recall_curve.png"))
    plt.show()
    cm = confusion_matrix(best_res["y_test"], best_res["y_pred"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label"); plt.ylabel("True Label")
    plt.title("Confusion Matrix – Best Model")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "confusion_matrix.png"))
    plt.show()
    fpr, tpr, _ = roc_curve(best_res["y_test"], best_res["y_prob"])
    auc_val = roc_auc_score(best_res["y_test"], best_res["y_prob"])
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"MLP (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Best Performing Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "roc_curve_best_model.png"))
    plt.show()

# ---------- 16.4 SHAP
print("\n📊 16.4 Generating SHAP Analysis...")
X_all = df[best_res["features"]].dropna()
split_idx = int(len(X_all) * 0.8)
X_train_shap = best_res["scaler"].transform(X_all.iloc[:split_idx])
X_test_shap = best_res["scaler"].transform(X_all.iloc[split_idx:split_idx + 200])

try:
    background = shap.sample(X_train_shap, 100, random_state=RANDOM_STATE)
    predict_fn = best_res["model"].predict if USE_REGRESSION else best_res["model"].predict_proba
    if not USE_REGRESSION:
        predict_fn = lambda x: predict_fn(x)[:, 1]
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_test_shap)
    X_test_df = pd.DataFrame(X_test_shap, columns=best_res["features"])
    shap_vals_plot = shap_values[1] if (isinstance(shap_values, list) and not USE_REGRESSION) else shap_values
    if np.ndim(shap_vals_plot) == 3:
        shap_vals_plot = shap_vals_plot[1]
    shap_vals_plot = np.asarray(shap_vals_plot)[:, :X_test_df.shape[1]]
    shap.summary_plot(shap_vals_plot, X_test_df, plot_type="bar", show=True)
    print("✓ SHAP analysis completed")
except Exception as e:
    print(f"⚠️ SHAP analysis failed: {e}")
    shap_values = None
    X_test_shap = None

# ---------- 16.5 Permutation Importance
perm_df = permutation_importance_scoring(
    best_res["model"], best_res["X_test"], best_res["y_test"], best_res["features"],
    use_regression=USE_REGRESSION
)
plt.figure(figsize=(8, 6))
sns.barplot(data=perm_df.head(15), x="Importance", y="Feature")
plt.title("Permutation Importance (" + ("R²" if USE_REGRESSION else "AUC") + ") – Best Model")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "permutation_importance.png"))
plt.show()

# ---------- 16.6 BACKTEST: Chiến lược predicted_return > 0 → Mua; so sánh Buy & Hold, Random Walk
def backtest_metrics(true_returns, pred_returns):
    """pred_returns: predicted log return. Strategy: earn true_returns[t] if pred_returns[t] > 0 else 0."""
    strategy_ret = np.where(pred_returns > 0, true_returns, 0.0)
    cum_strategy = np.exp(np.cumsum(strategy_ret)) - 1  # cumulative simple return
    cum_bh = np.exp(np.cumsum(true_returns)) - 1  # Buy & Hold
    rw_ret = np.zeros_like(true_returns)  # Random Walk: never buy → 0
    ann = 252
    def sharpe(r):
        if np.std(r) < 1e-12:
            return 0.0
        return np.mean(r) / np.std(r) * np.sqrt(ann)
    def max_dd(cum):
        peak = np.maximum.accumulate(cum)
        return np.min((cum - peak) / (peak + 1e-12))
    return {
        "Cumulative_return": cum_strategy[-1] if len(cum_strategy) else 0,
        "Sharpe_ratio": sharpe(strategy_ret),
        "Max_drawdown": max_dd(cum_strategy + 1),
        "B&H_cumulative": cum_bh[-1] if len(cum_bh) else 0,
        "B&H_Sharpe": sharpe(true_returns),
        "B&H_MaxDD": max_dd(cum_bh + 1),
    }

if USE_REGRESSION and best_res.get("true_return") is not None and best_res.get("pred_return") is not None:
    print("\n📊 16.6 Backtest (Strategy: Mua khi predicted return > 0)")
    bt = backtest_metrics(best_res["true_return"], best_res["pred_return"])
    print(f"   Strategy: Cumulative return = {bt['Cumulative_return']:.4f}, Sharpe = {bt['Sharpe_ratio']:.3f}, MaxDD = {bt['Max_drawdown']:.4f}")
    print(f"   Buy & Hold: Cumulative = {bt['B&H_cumulative']:.4f}, Sharpe = {bt['B&H_Sharpe']:.3f}, MaxDD = {bt['B&H_MaxDD']:.4f}")
    if bt["Sharpe_ratio"] <= 0 and bt["B&H_Sharpe"] > 0:
        print("   ⚠️ MLP strategy không vượt Buy & Hold – cần thảo luận hạn chế.")
    pd.DataFrame([bt]).to_csv(os.path.join(VIS_DIR, "backtest_metrics.csv"), index=False)

# ---------- 16.7 Diebold–Mariano Test (so sánh sai số dự báo: MLP vs Random Walk vs AR(1))
def diebold_mariano(e1, e2, loss="squared"):
    """e1, e2: forecast errors (y_true - y_pred). H0: equal predictive ability."""
    if loss == "squared":
        d = (e1 ** 2) - (e2 ** 2)
    else:
        d = np.abs(e1) - np.abs(e2)
    n = len(d)
    if n < 2:
        return np.nan, np.nan
    dm = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(n) + 1e-12)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm)))
    return dm, p_value

def get_baseline_predictions_same_split(df, features, target_col, date_col="date", split_date="2023-01-01"):
    """Cùng split với MLP (data = df[features+target].dropna()). Trả về (y_test, y_pred_rw, y_pred_ar1)."""
    need = list(set(features + [target_col, date_col]))
    if "ret_1d" in df.columns:
        need.append("ret_1d")
    data = df[[c for c in need if c in df.columns]].dropna()
    if len(data) < 200:
        return None, None, None
    split_dt = pd.to_datetime(split_date)
    train_mask = data[date_col] < split_dt
    test_mask = ~train_mask
    if test_mask.sum() < 30:
        return None, None, None
    y_test = data.loc[test_mask, target_col].values
    y_pred_rw = np.zeros_like(y_test)
    y_pred_ar1 = None
    if "ret_1d" in data.columns:
        r_train = data.loc[train_mask, "ret_1d"].values
        r_test = data.loc[test_mask, "ret_1d"].values
        y_train = data.loc[train_mask, target_col].values
        phi = np.cov(y_train, r_train)[0, 1] / (np.var(r_train) + 1e-12)
        y_pred_ar1 = phi * r_test
    return y_test, y_pred_rw, y_pred_ar1

if USE_REGRESSION and REGRESSION_TARGET == "return":
    y_te, y_rw, y_ar1 = get_baseline_predictions_same_split(df, best_features, TARGET_COL, split_date="2023-01-01")
    if y_te is not None and len(y_te) == len(best_res["y_test"]):
        e_mlp = best_res["y_test"] - best_res["y_pred"]
        e_rw = best_res["y_test"] - y_rw
        dm_rw, p_rw = diebold_mariano(e_mlp, e_rw)
        print("\n📊 16.7 Diebold–Mariano Test (sai số bình phương)")
        print(f"   MLP vs Random Walk: DM = {dm_rw:.4f}, p-value = {p_rw:.4f}" + (" (có ý nghĩa p<0.05)" if p_rw < 0.05 else " (không khác biệt)"))
        if y_ar1 is not None:
            e_ar1 = best_res["y_test"] - y_ar1
            dm_ar1, p_ar1 = diebold_mariano(e_mlp, e_ar1)
            print(f"   MLP vs AR(1): DM = {dm_ar1:.4f}, p-value = {p_ar1:.4f}" + (" (có ý nghĩa p<0.05)" if p_ar1 < 0.05 else " (không khác biệt)"))

# ---------- 16.8 Multi-horizon (1d, 5d, 10d)
print("\n📊 16.8 Multi-horizon (1d, 5d, 10d)")
horizon_results = []
for h in [1, 5, 10]:
    col = f"future_log_return_{h}d"
    if col not in df_before_target_drop.columns:
        continue
    df_h = df_before_target_drop.copy()
    df_h["target"] = df_h[col]
    df_h = df_h.dropna(subset=["target"]).reset_index(drop=True)
    feats = [f for f in feature_groups["All"] if f in df_h.columns]
    if len(feats) < 3:
        continue
    res_h = train_mlp_time_split(
        df_h, feats, target_col="target", use_regression=True,
        regression_target="return", signal_threshold_pct=SIGNAL_THRESHOLD_PCT,
    )
    if res_h is not None:
        horizon_results.append({
            "Horizon_days": h,
            "R2": res_h["metrics"]["R2"],
            "MAE": res_h["metrics"]["MAE"],
            "RMSE": res_h["metrics"]["RMSE"],
            "Accuracy_direction": res_h["metrics"].get("Accuracy_direction", np.nan),
        })
if horizon_results:
    horizon_df = pd.DataFrame(horizon_results)
    print(horizon_df.to_string(index=False))
    horizon_df.to_csv(os.path.join(VIS_DIR, "multi_horizon_results.csv"), index=False)

print("\n✅ Section 16 completed.")


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

if USE_REGRESSION:
    # Chuẩn hóa R2/RMSE/MAE vào [0,1] cho radar: R2 có thể âm -> (R2+1)/2; RMSE, MAE -> 1/(1+x)
    rdf = results_df.copy()
    rdf["R2_norm"] = (rdf["R2"] + 1) / 2
    rdf["RMSE_norm"] = 1 / (1 + rdf["RMSE"])
    rdf["MAE_norm"] = 1 / (1 + rdf["MAE"])
    radar_metrics = ["R2_norm", "RMSE_norm", "MAE_norm", "Accuracy_direction"]
    make_radar_chart(rdf, radar_metrics, "Model Comparison – R² / RMSE / MAE / Acc (Direction)", "radar_metrics.png")
else:
    radar_metrics = ["Accuracy", "AUC", "Precision", "Recall", "F1", "AP"]
    make_radar_chart(results_df, radar_metrics, "Model Performance Comparison – Radar Chart", "radar_metrics.png")
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
    res = train_mlp_time_split(
        sector_df, best_features_list,
        target_col=TARGET_COL,
        use_regression=USE_REGRESSION,
        regression_target=REGRESSION_TARGET if USE_REGRESSION else "direction",
        signal_threshold_pct=SIGNAL_THRESHOLD_PCT,
    )
    if res is not None:
        if USE_REGRESSION:
            sector_results.append({
                "Sector": sector,
                "R2": res["metrics"]["R2"],
                "RMSE": res["metrics"]["RMSE"],
                "MAE": res["metrics"]["MAE"],
                "Accuracy_direction": res["metrics"]["Accuracy_direction"],
                "Samples": len(sector_df),
            })
        else:
            sector_results.append({
                "Sector": sector,
                "AUC": res["metrics"]["AUC"],
                "Accuracy": res["metrics"]["Accuracy"],
                "F1": res["metrics"]["F1"],
                "Samples": len(sector_df),
            })

if sector_results:
    if USE_REGRESSION:
        sector_df_results = pd.DataFrame(sector_results).sort_values("R2", ascending=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sector_df_results)))
        bars = ax.barh(sector_df_results["Sector"], sector_df_results["R2"], color=colors)
        ax.axvline(0, color="red", linestyle="--", alpha=0.7, label="Baseline (R²=0)")
        ax.axvline(sector_df_results["R2"].mean(), color="blue", linestyle="--", alpha=0.7,
                   label=f"Mean R²={sector_df_results['R2'].mean():.3f}")
        for bar, val in zip(bars, sector_df_results["R2"]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
        ax.set_xlabel("R² Score", fontsize=12)
        ax.set_title("Model Performance by Sector (Direction Regression)", fontsize=14, fontweight="bold")
    else:
        sector_df_results = pd.DataFrame(sector_results).sort_values("AUC", ascending=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sector_df_results)))
        bars = ax.barh(sector_df_results["Sector"], sector_df_results["AUC"], color=colors)
        ax.axvline(0.5, color="red", linestyle="--", alpha=0.7, label="Random (AUC=0.5)")
        ax.axvline(sector_df_results["AUC"].mean(), color="blue", linestyle="--", alpha=0.7,
                   label=f"Mean AUC={sector_df_results['AUC'].mean():.3f}")
        for bar, auc in zip(bars, sector_df_results["AUC"]):
            ax.text(auc + 0.01, bar.get_y() + bar.get_height()/2, f"{auc:.3f}", va="center", fontsize=9)
        ax.set_xlabel("AUC Score", fontsize=12)
        ax.set_title("Model Performance by Sector", fontsize=14, fontweight="bold")
        ax.set_xlim(0.35, 0.85)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "sector_performance.png"), dpi=150)
    plt.show()
    print(f"✓ Saved: {VIS_DIR}/sector_performance.png")
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
# 18.1 Bootstrap Confidence Intervals (AUC hoặc R²)
# -------------------------------------------------------------
print("\n📊 18.1 Computing Bootstrap Confidence Intervals...")

def bootstrap_ci_auc(y_true, y_prob, n_bootstraps=1000, ci=0.95, random_state=42):
    rng = np.random.RandomState(random_state)
    aucs = []
    n_samples = len(y_true)
    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[indices], y_prob[indices]))
    aucs = np.array(aucs)
    alpha = (1 - ci) / 2
    return np.mean(aucs), np.percentile(aucs, alpha * 100), np.percentile(aucs, (1 - alpha) * 100)


def bootstrap_ci_r2(y_true, y_pred, n_bootstraps=1000, ci=0.95, random_state=42):
    rng = np.random.RandomState(random_state)
    r2s = []
    n_samples = len(y_true)
    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, n_samples, replace=True)
        r2s.append(r2_score(y_true[indices], y_pred[indices]))
    r2s = np.array(r2s)
    alpha = (1 - ci) / 2
    return np.mean(r2s), np.percentile(r2s, alpha * 100), np.percentile(r2s, (1 - alpha) * 100)

if USE_REGRESSION:
    mean_metric, ci_lower, ci_upper = bootstrap_ci_r2(best_res["y_test"], best_res["y_pred"])
    print(f"\nBest Model ({best_model_name}) R² with 95% CI:")
    print(f"   R² = {mean_metric:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    ci_results = []
    for name, feats in feature_groups.items():
        res = train_mlp_time_split(
            df, feats, target_col=TARGET_COL, use_regression=True,
            regression_target=REGRESSION_TARGET or "direction",
            signal_threshold_pct=SIGNAL_THRESHOLD_PCT,
        )
        if res is None:
            continue
        m, lo, hi = bootstrap_ci_r2(res["y_test"], res["y_pred"])
        ci_results.append({"Model": name, "R2": m, "CI_Lower": lo, "CI_Upper": hi})
    ci_df = pd.DataFrame(ci_results).sort_values("R2", ascending=False)
    print("\n--- All Models with 95% CI (R²) ---")
    print(ci_df.to_string(index=False))
    plt.figure(figsize=(10, 6))
    x_pos = range(len(ci_df))
    plt.bar(x_pos, ci_df["R2"], yerr=[ci_df["R2"] - ci_df["CI_Lower"], ci_df["CI_Upper"] - ci_df["R2"]],
            capsize=5, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(ci_df))), edgecolor="black", linewidth=1)
    plt.xticks(x_pos, ci_df["Model"], rotation=30, ha="right")
    plt.ylabel("R² Score")
    plt.title("Model R² Comparison with 95% Confidence Intervals", fontsize=12, fontweight="bold")
    plt.axhline(0, color="red", linestyle="--", alpha=0.5, label="Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "r2_confidence_intervals.png"), dpi=150)
    plt.show()
    print(f"✓ Saved: {VIS_DIR}/r2_confidence_intervals.png")
else:
    mean_auc, ci_lower, ci_upper = bootstrap_ci_auc(best_res["y_test"], best_res["y_prob"])
    print(f"\nBest Model ({best_model_name}) AUC with 95% CI:")
    print(f"   AUC = {mean_auc:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    ci_results = []
    for name, feats in feature_groups.items():
        res = train_mlp_time_split(df, feats, target_col=TARGET_COL, use_regression=False)
        if res is None:
            continue
        m, lo, hi = bootstrap_ci_auc(res["y_test"], res["y_prob"])
        ci_results.append({"Model": name, "AUC": m, "CI_Lower": lo, "CI_Upper": hi})
    ci_df = pd.DataFrame(ci_results).sort_values("AUC", ascending=False)
    print("\n--- All Models with 95% CI ---")
    print(ci_df.to_string(index=False))
    plt.figure(figsize=(10, 6))
    x_pos = range(len(ci_df))
    plt.bar(x_pos, ci_df["AUC"], yerr=[ci_df["AUC"] - ci_df["CI_Lower"], ci_df["CI_Upper"] - ci_df["AUC"]],
            capsize=5, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(ci_df))), edgecolor="black", linewidth=1)
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
# 18.2 Wilcoxon Signed-Rank Test (per-fold R², p-value, effect size)
# -------------------------------------------------------------
print("\n📊 18.2 Performing Wilcoxon Signed-Rank Tests...")

def cv_r2_per_fold_with_baselines(df, features, target_col, n_splits=5):
    """TimeSeriesSplit: trả về R² theo từng fold cho MLP, Random Walk, AR(1). Scaler chỉ fit trên train."""
    cols = features + [target_col]
    if "ret_1d" in df.columns:
        cols = cols + ["ret_1d"]
    data = df[cols].dropna()
    X = data[features].values
    y = data[target_col].values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    s_mlp, s_rw, s_ar1 = [], [], []
    for train_idx, test_idx in tscv.split(X):
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X[train_idx])
        X_test_s = scaler.transform(X[test_idx])
        model = MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32), max_iter=500, random_state=RANDOM_STATE)
        model.fit(X_train_s, y[train_idx])
        y_pred_mlp = model.predict(X_test_s)
        s_mlp.append(r2_score(y[test_idx], y_pred_mlp))
        y_pred_rw = np.zeros_like(y[test_idx])
        s_rw.append(r2_score(y[test_idx], y_pred_rw))
        if "ret_1d" in data.columns:
            r_train = data["ret_1d"].values[train_idx]
            r_test = data["ret_1d"].values[test_idx]
            phi = np.cov(y[train_idx], r_train)[0, 1] / (np.var(r_train) + 1e-12)
            y_pred_ar1 = phi * r_test
            s_ar1.append(r2_score(y[test_idx], y_pred_ar1))
    return np.array(s_mlp), np.array(s_rw), np.array(s_ar1) if s_ar1 else None

def wilcoxon_effect_size(s1, s2):
    """Wilcoxon signed-rank: trả về stat, p_value, effect size r = Z/sqrt(N)."""
    stat, p = stats.wilcoxon(s1, s2)
    n = len(s1)
    # Z từ Wilcoxon T: Z = (T - n(n+1)/4) / sqrt(n(n+1)(2n+1)/24)
    E_T = n * (n + 1) / 4
    std_T = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (stat - E_T) / (std_T + 1e-12)
    r = z / np.sqrt(n)  # effect size
    return stat, p, r

def paired_cv_comparison(df, features1, features2, n_splits=5, use_regression=False):
    """Paired CV: returns (scores1, scores2) per fold — AUC or R²."""
    all_features = list(set(features1 + features2))
    data = df[all_features + ["target"]].dropna()
    X = data[all_features]
    y = data["target"].values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    s1, s2 = [], []
    for train_idx, test_idx in tscv.split(X):
        X1 = data[features1].values
        scaler1 = RobustScaler()
        X1_train = scaler1.fit_transform(X1[train_idx])
        X1_test = scaler1.transform(X1[test_idx])
        X2 = data[features2].values
        scaler2 = RobustScaler()
        X2_train = scaler2.fit_transform(X2[train_idx])
        X2_test = scaler2.transform(X2[test_idx])
        if use_regression:
            model1 = MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32), max_iter=500, random_state=RANDOM_STATE)
            model1.fit(X1_train, y[train_idx])
            s1.append(r2_score(y[test_idx], model1.predict(X1_test)))
            model2 = MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32), max_iter=500, random_state=RANDOM_STATE)
            model2.fit(X2_train, y[train_idx])
            s2.append(r2_score(y[test_idx], model2.predict(X2_test)))
        else:
            model1 = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), max_iter=500, random_state=RANDOM_STATE)
            model1.fit(X1_train, y[train_idx])
            s1.append(roc_auc_score(y[test_idx], model1.predict_proba(X1_test)[:, 1]))
            model2 = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), max_iter=500, random_state=RANDOM_STATE)
            model2.fit(X2_train, y[train_idx])
            s2.append(roc_auc_score(y[test_idx], model2.predict_proba(X2_test)[:, 1]))
    return np.array(s1), np.array(s2)

# Wilcoxon vs Baselines: R² theo từng fold, p-value và effect size
if USE_REGRESSION and best_model_name in feature_groups:
    try:
        s_mlp, s_rw, s_ar1 = cv_r2_per_fold_with_baselines(df, best_features, TARGET_COL, n_splits=5)
        print("\nR² per fold (Best MLP vs Baselines):")
        print(f"  MLP (best): {s_mlp}  mean={s_mlp.mean():.4f}")
        print(f"  Random Walk: {s_rw}  mean={s_rw.mean():.4f}")
        if s_ar1 is not None:
            print(f"  AR(1): {s_ar1}  mean={s_ar1.mean():.4f}")
        stat_rw, p_rw, r_rw = wilcoxon_effect_size(s_mlp, s_rw)
        print(f"\nWilcoxon MLP vs Random Walk: p-value={p_rw:.4f}, effect size r={r_rw:.4f}")
        if s_ar1 is not None:
            stat_ar1, p_ar1, r_ar1 = wilcoxon_effect_size(s_mlp, s_ar1)
            print(f"Wilcoxon MLP vs AR(1): p-value={p_ar1:.4f}, effect size r={r_ar1:.4f}")
        wilcoxon_baselines_df = pd.DataFrame({
            "Fold": range(1, len(s_mlp) + 1),
            "R2_MLP": s_mlp,
            "R2_RandomWalk": s_rw,
            **({"R2_AR1": s_ar1} if s_ar1 is not None else {}),
        })
        wilcoxon_baselines_df.to_csv(os.path.join(VIS_DIR, "wilcoxon_per_fold_r2.csv"), index=False)
    except Exception as e:
        print(f"Wilcoxon vs baselines skipped: {e}")

# Compare best model vs other feature sets
print(f"\nStatistical comparison: {best_model_name} vs other models")
print("-" * 60)

stat_results = []
for name, feats in feature_groups.items():
    if name == best_model_name:
        continue
    
    try:
        scores_best, scores_other = paired_cv_comparison(
            df, feature_groups[best_model_name], feats, n_splits=5, use_regression=USE_REGRESSION
        )
        stat, p_value = stats.wilcoxon(scores_best, scores_other)
        mean_diff = np.mean(scores_best - scores_other)
        stat_results.append({
            "Comparison": f"{best_model_name} vs {name}",
            "Best_Mean": np.mean(scores_best),
            "Other_Mean": np.mean(scores_other),
            "Difference": mean_diff,
            "p-value": p_value,
            "Significant": "Yes" if p_value < 0.05 else "No",
        })
        significance = "✓" if p_value < 0.05 else "✗"
        metric_name = "R²" if USE_REGRESSION else "AUC"
        print(f"{best_model_name} vs {name}: Δ{metric_name}={mean_diff:+.4f}, p={p_value:.4f} {significance}")
        
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
    ax.set_ylabel("R² Difference" if USE_REGRESSION else "AUC Difference")
    ax.set_title(f"Statistical Comparison: {best_model_name} vs Others\n(Green = p<0.05, Red = p≥0.05)", fontsize=12, fontweight="bold")
    
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

# Sort by R2 or AUC for ranking
rank_col = "R2" if USE_REGRESSION else "AUC"
results_sorted = results_df.sort_values(rank_col, ascending=(not USE_REGRESSION)).reset_index(drop=True)
results_sorted["Rank"] = range(1, len(results_sorted) + 1)

print("\n╔════════════════════════════════════════════════════════════════════════╗")
print("║             ABLATION STUDY - FEATURE COMBINATION ANALYSIS             ║")
print("╠════════════════════════════════════════════════════════════════════════╣")
if USE_REGRESSION:
    print(f"║ {'Rank':<4} │ {'Model':<15} │ {'R2':<10} │ {'RMSE':<10} │ {'MAE':<10} ║")
    print("╠════════════════════════════════════════════════════════════════════════╣")
    for _, row in results_sorted.iterrows():
        print(f"║ {int(row['Rank']):<4} │ {row['Model']:<15} │ {row['R2']:<10.4f} │ {row['RMSE']:<10.4f} │ {row['MAE']:<10.4f} ║")
else:
    print(f"║ {'Rank':<4} │ {'Model':<15} │ {'Accuracy':<10} │ {'AUC':<10} │ {'F1':<10} ║")
    print("╠════════════════════════════════════════════════════════════════════════╣")
    for _, row in results_sorted.iterrows():
        print(f"║ {int(row['Rank']):<4} │ {row['Model']:<15} │ {row['Accuracy']:<10.4f} │ {row['AUC']:<10.4f} │ {row['F1']:<10.4f} ║")
print("╚════════════════════════════════════════════════════════════════════════╝")

# Detailed interpretation
print("\n" + "=" * 60)
print("ACADEMIC INTERPRETATION")
print("=" * 60)

def _get_row(df, name):
    if name not in df["Model"].values:
        return None
    return df[df["Model"] == name].iloc[0]

fundamental_only = _get_row(results_df, "Fundamental")
technical_only = _get_row(results_df, "Technical")
sentiment_only = _get_row(results_df, "Sentiment")
fund_tech = _get_row(results_df, "Fund+Tech")
fund_sent = _get_row(results_df, "Fund+Sent")
tech_sent = _get_row(results_df, "Tech+Sent")
all_features_row = _get_row(results_df, "All")

if USE_REGRESSION:
    print("\n┌──────────────────────────────────────────────────────────────────────┐")
    print("│ 1. SINGLE FEATURE GROUP PERFORMANCE (R² / RMSE)                      │")
    print("└──────────────────────────────────────────────────────────────────────┘")
    for label, row in [("Fundamental", fundamental_only), ("Technical", technical_only), ("Sentiment", sentiment_only)]:
        if row is not None:
            print(f"   • {label:12s}: R²={row['R2']:.4f}, RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}")
    single_models = results_df[results_df["Model"].isin(["Fundamental", "Technical", "Sentiment"])]
    if len(single_models) > 0:
        best_single = single_models.loc[single_models["R2"].idxmax()]
        print(f"\n   → Best single: {best_single['Model']} (R²={best_single['R2']:.4f})")
    print("\n┌──────────────────────────────────────────────────────────────────────┐")
    print("│ 2. COMBINATION EFFECTS (R²)                                          │")
    print("└──────────────────────────────────────────────────────────────────────┘")
    for (label, row), (b1, b2) in [
        (("Fund+Tech", fund_tech), (fundamental_only, technical_only)),
        (("Fund+Sent", fund_sent), (fundamental_only, sentiment_only)),
        (("Tech+Sent", tech_sent), (technical_only, sentiment_only)),
    ]:
        if row is not None and b1 is not None and b2 is not None:
            baseline = max(b1["R2"], b2["R2"])
            improvement = row["R2"] - baseline
            effect = "SYNERGY ↑" if improvement > 0.01 else ("NEUTRAL =" if abs(improvement) <= 0.01 else "INTERFERENCE ↓")
            print(f"   • {label}: R²={row['R2']:.4f} | vs best single: {improvement:+.4f} ({effect})")
    print("\n┌──────────────────────────────────────────────────────────────────────┐")
    print("│ 3. ALL FEATURES & 4. CONCLUSIONS                                    │")
    print("└──────────────────────────────────────────────────────────────────────┘")
    best_overall = results_sorted.iloc[0]
    worst_overall = results_sorted.iloc[-1]
    print(f"   ★ BEST: {best_overall['Model']}  R²={best_overall['R2']:.4f}, RMSE={best_overall['RMSE']:.4f}, MAE={best_overall['MAE']:.4f}")
    print(f"   ✗ WORST: {worst_overall['Model']}  R²={worst_overall['R2']:.4f}")
    if best_overall["R2"] > 0.05:
        conclusion = "Models show MODERATE predictive power (R² > 0.05)."
    elif best_overall["R2"] > 0:
        conclusion = "Models show SLIGHT predictive edge (R² > 0)."
    else:
        conclusion = "Models perform NEAR/ below baseline - features may not contain strong signals."
    print(f"   📌 CONCLUSION: {conclusion}")
    analysis_df = results_sorted[["Rank", "Model", "R2", "RMSE", "MAE", "Accuracy_direction"]].copy()
    analysis_df.to_csv("feature_combination_analysis.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(results_sorted)))[::-1]
    bars1 = axes[0].barh(results_sorted["Model"], results_sorted["R2"], color=colors)
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.7, label="Baseline")
    axes[0].set_xlabel("R²"); axes[0].set_title("Feature Combination R² Comparison", fontweight="bold")
    for bar, val in zip(bars1, results_sorted["R2"]):
        axes[0].text(val + 0.005, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
    bars2 = axes[1].barh(results_sorted["Model"], results_sorted["RMSE"], color=colors)
    axes[1].set_xlabel("RMSE"); axes[1].set_title("Feature Combination RMSE Comparison", fontweight="bold")
    for bar, val in zip(bars2, results_sorted["RMSE"]):
        axes[1].text(val + 0.005, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "feature_combination_comparison.png"), dpi=150)
    plt.show()
else:
    print("\n┌──────────────────────────────────────────────────────────────────────┐")
    print("│ 1. SINGLE FEATURE GROUP PERFORMANCE                                 │")
    print("└──────────────────────────────────────────────────────────────────────┘")
    if fundamental_only is not None:
        print(f"   • Fundamental: AUC={fundamental_only['AUC']:.4f}, Acc={fundamental_only['Accuracy']:.4f}")
    if technical_only is not None:
        print(f"   • Technical:   AUC={technical_only['AUC']:.4f}, Acc={technical_only['Accuracy']:.4f}")
    if sentiment_only is not None:
        print(f"   • Sentiment:   AUC={sentiment_only['AUC']:.4f}, Acc={sentiment_only['Accuracy']:.4f}")
    single_models = results_df[results_df["Model"].isin(["Fundamental", "Technical", "Sentiment"])]
    if len(single_models) > 0:
        best_single = single_models.loc[single_models["AUC"].idxmax()]
        print(f"\n   → Best single feature group: {best_single['Model']} (AUC={best_single['AUC']:.4f})")
    print("\n┌──────────────────────────────────────────────────────────────────────┐")
    print("│ 2. COMBINATION EFFECTS                                              │")
    print("└──────────────────────────────────────────────────────────────────────┘")
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
    if all_features_row is not None:
        best_combination = results_df[results_df["Model"].isin(["Fund+Tech", "Fund+Sent", "Tech+Sent"])]
        if len(best_combination) > 0:
            best_combo = best_combination.loc[best_combination["AUC"].idxmax()]
            improvement = all_features_row['AUC'] - best_combo['AUC']
            print(f"   • All Features: AUC={all_features_row['AUC']:.4f}, Acc={all_features_row['Accuracy']:.4f}")
            print(f"   • vs Best Pair ({best_combo['Model']}): {improvement:+.4f}")
    print("\n┌──────────────────────────────────────────────────────────────────────┐")
    print("│ 4. KEY FINDINGS & CONCLUSIONS                                       │")
    print("└──────────────────────────────────────────────────────────────────────┘")
    best_overall = results_sorted.iloc[0]
    worst_overall = results_sorted.iloc[-1]
    print(f"   ★ BEST MODEL: {best_overall['Model']}  AUC={best_overall['AUC']:.4f}, Acc={best_overall['Accuracy']:.4f}, F1={best_overall['F1']:.4f}")
    print(f"   ✗ WORST MODEL: {worst_overall['Model']}  AUC={worst_overall['AUC']:.4f}")
    if best_overall['AUC'] > 0.55:
        conclusion = "Models show MODERATE predictive power above random baseline."
    elif best_overall['AUC'] > 0.52:
        conclusion = "Models show SLIGHT predictive edge over random baseline."
    else:
        conclusion = "Models perform NEAR RANDOM - features may not contain strong predictive signals."
    print(f"   📌 CONCLUSION: {conclusion}")
    analysis_df = results_sorted[["Rank", "Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "AP"]]
    analysis_df.to_csv("feature_combination_analysis.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(results_sorted)))[::-1]
    bars1 = axes[0].barh(results_sorted["Model"], results_sorted["AUC"], color=colors)
    axes[0].axvline(0.5, color="red", linestyle="--", alpha=0.7, label="Random (0.5)")
    axes[0].set_xlabel("AUC Score"); axes[0].set_title("Feature Combination AUC Comparison", fontweight="bold")
    for bar, val in zip(bars1, results_sorted["AUC"]):
        axes[0].text(val + 0.005, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
    bars2 = axes[1].barh(results_sorted["Model"], results_sorted["Accuracy"], color=colors)
    axes[1].axvline(0.5, color="red", linestyle="--", alpha=0.7, label="Random (0.5)")
    axes[1].set_xlabel("Accuracy"); axes[1].set_title("Feature Combination Accuracy Comparison", fontweight="bold")
    for bar, val in zip(bars2, results_sorted["Accuracy"]):
        axes[1].text(val + 0.005, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "feature_combination_comparison.png"), dpi=150)
    plt.show()
print("✓ Saved: feature_combination_analysis.csv")
print(f"✓ Saved: {VIS_DIR}/feature_combination_comparison.png")


# =============================================================
# 19. THẢO LUẬN & HẠN CHẾ (Discussion & Limitations)
# =============================================================
print("\n" + "=" * 60)
print("SECTION 19: Thảo luận & Hạn chế")
print("=" * 60)
print("""
• Ý nghĩa kinh tế: Target là log return (không phải giá) giúp đánh giá đúng khả năng dự báo
  lợi suất; R² cao trên giá thường do tự tương quan, ít ý nghĩa giao dịch.
• Baseline: Nếu MLP không vượt Random Walk / AR(1) trên R² hoặc Backtest (Sharpe, drawdown),
  mô hình chưa có giá trị thực tế.
• Kiểm định thống kê: Wilcoxon (phân phối R² theo fold) và Diebold-Mariano (sai số dự báo)
  cho biết cải thiện có ý nghĩa thống kê (p < 0.05) hay không.
• Hạn chế: Dữ liệu lịch sử, điều kiện thị trường thay đổi; chi phí giao dịch chưa được mô hình hóa;
  đa horizon (5d, 10d) có thể khác 1d.
""")

# =============================================================
# 20. FINAL SUMMARY
# =============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

ci_row = ci_df[ci_df["Model"] == best_model_name].iloc[0]
if USE_REGRESSION:
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           MLP STOCK PREDICTION (DIRECTION REGRESSION)         ║
║                    ANALYSIS COMPLETE                          ║
╠══════════════════════════════════════════════════════════════╣
║  Best Model: {best_model_name:<20}                            ║
║  R² Score:   {best_row['R2']:.4f} [{ci_row['CI_Lower']:.4f}, {ci_row['CI_Upper']:.4f}]              ║
║  RMSE:       {best_row['RMSE']:.4f}   MAE: {best_row['MAE']:.4f}   Acc(dir): {best_row['Accuracy_direction']:.4f}  ║
╠══════════════════════════════════════════════════════════════╣
║  Visualizations: {VIS_DIR}/  (ablation_r2_bar, rolling_r2,   ║
║  regression_scatter, regression_residuals, confusion_matrix, ║
║  r2_confidence_intervals, statistical_comparison, ...)       ║
╠══════════════════════════════════════════════════════════════╣
║  CSV: model_results.csv, sector_results.csv, statistical_tests.csv  ║
╚══════════════════════════════════════════════════════════════╝
""")
else:
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    MLP STOCK PREDICTION                       ║
║                    ANALYSIS COMPLETE                          ║
╠══════════════════════════════════════════════════════════════╣
║  Best Model: {best_model_name:<20}                            ║
║  AUC Score:  {best_row['AUC']:.4f} [{ci_row['CI_Lower']:.4f}, {ci_row['CI_Upper']:.4f}]              ║
║  Accuracy:   {best_row['Accuracy']:.4f}   F1: {best_row['F1']:.4f}                                      ║
╠══════════════════════════════════════════════════════════════╣
║  Visualizations saved to: {VIS_DIR}/                        ║
║  CSV: model_results.csv, sector_results.csv, statistical_tests.csv  ║
╚══════════════════════════════════════════════════════════════╝
""")

print("✅ All analysis completed successfully!")