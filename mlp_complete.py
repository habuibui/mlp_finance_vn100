# -------------------------------------------------------------
# 1. IMPORTS & GLOBAL SETTINGS
# -------------------------------------------------------------
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

# --- Target: return 5 ngày tới (quantile 20/80) ---
TARGET_RETURN_DAYS = 5

# --- Cấu hình MLP & thống kê (bản gốc, chạy nhanh) ---
# Nếu muốn thử mô hình mạnh hơn thì tăng dần các giá trị này, nhưng nên giữ cấu trúc chung.
MLP_HIDDEN = (64, 32)           # mạng nhỏ (mặc định gốc)
MLP_MAX_ITER = 300              # số epoch tối đa
MLP_BATCH_SIZE = 256           # batch lớn = ít bước/epoch (nhanh hơn)
ROLLING_N_SPLITS = 3           # số fold rolling
WILCOXON_N_SPLITS = 3          # số fold cho Wilcoxon
PERMUTATION_N_REPEATS = 3      # lặp permutation importance
BOOTSTRAP_N = 200              # số lần bootstrap CI (AUC)

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

# Simple moving averages (5,10,20,50 + theo mô tả: 14, 28, 100)
for w in [5, 10, 14, 20, 28, 50, 100]:
    ohlc[f"sma_{w}"] = ohlc.groupby("mack")["close"].transform(lambda x: x.rolling(w).mean())

# Exponential moving averages
for w in [5, 10, 20, 50]:
    ohlc[f"ema_{w}"] = ohlc.groupby("mack")["close"].transform(lambda x: x.ewm(span=w, adjust=False).mean())

# RSI chu kỳ 9, 14, 28 (theo mô tả)
ohlc["rsi"] = ohlc.groupby("mack")["close"].transform(compute_rsi)  # 14
ohlc["rsi_9"] = ohlc.groupby("mack")["close"].transform(lambda x: compute_rsi(x, window=9))
ohlc["rsi_14"] = ohlc["rsi"].copy()
ohlc["rsi_28"] = ohlc.groupby("mack")["close"].transform(lambda x: compute_rsi(x, window=28))

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

# Stochastic oscillator (%K, %D) – chu kỳ 14
low_14 = ohlc.groupby("mack")["low"].transform(lambda x: x.rolling(14).min())
high_14 = ohlc.groupby("mack")["high"].transform(lambda x: x.rolling(14).max())
ohlc["stochastic_k"] = np.where(high_14 > low_14, 100 * (ohlc["close"] - low_14) / (high_14 - low_14), 50)
ohlc["stochastic_d"] = ohlc.groupby("mack")["stochastic_k"].transform(lambda x: x.rolling(3).mean())

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
# 10b. BỔ SUNG CHỈ SỐ KỸ THUẬT (nâng AUC): log volume, MA gap, crossover, rank, interaction
# -------------------------------------------------------------
if "volume" in df.columns:
    df["volume"] = np.log1p(df["volume"].clip(lower=0))
if "sma_20" in df.columns and df["sma_20"].gt(0).any():
    df["sma20_gap"] = np.where(df["sma_20"] > 0, (df["close"] - df["sma_20"]) / df["sma_20"], 0)
if "sma_10" in df.columns and "sma_50" in df.columns:
    denom = df["sma_50"].replace(0, np.nan)
    df["sma_cross"] = ((df["sma_10"] - df["sma_50"]) / denom).fillna(0).replace([np.inf, -np.inf], 0)
if "ret_5d" in df.columns and "date" in df.columns:
    df["ret_5d_rank"] = df.groupby("date")["ret_5d"].rank(pct=True)
if "ret_5d" in df.columns and "volatility_20d" in df.columns:
    df["momentum_vol"] = df["ret_5d"] * df["volatility_20d"]
print("  Đã thêm: log(volume), sma20_gap, sma_cross, ret_5d_rank, momentum_vol")

# -------------------------------------------------------------
# TARGET: Quantile-based binary (return 1d hoặc 5d tùy TARGET_RETURN_DAYS)
# -------------------------------------------------------------
_n = TARGET_RETURN_DAYS
df["future_return"] = (
    df.groupby("mack")["close"]
    .pct_change(_n)
    .shift(-_n)
)

q_low = df["future_return"].quantile(0.2)
q_high = df["future_return"].quantile(0.8)

df["target"] = np.where(
    df["future_return"] >= q_high, 1,
    np.where(df["future_return"] <= q_low, 0, np.nan)
)

df = df.dropna(subset=["target"]).reset_index(drop=True)
df["target"] = df["target"].astype(int)
print(f"  Target: return {_n}d tới (quantile 20/80). Lớp 1: {(df['target']==1).sum():,}, Lớp 0: {(df['target']==0).sum():,}")


# -------------------------------------------------------------
# 12. FEATURE GROUPS
# -------------------------------------------------------------
fundamental_features = [c for c in fund_features if c in df.columns]

# 1. Nhóm Phân tích cơ bản (Fundamental): P/B, EPS, P/E, ROE, ROA, nợ/vốn, ...
# (EV/EBITDA, PEG, Beta không có trong file → bỏ qua; thêm khi có dữ liệu)
fundamental_features = [c for c in fund_features if c in df.columns]

# 3. Nhóm Phân tích kỹ thuật: SMA, RSI, Stochastic + gap/crossover/rank/tương tác
technical_features = [
    "sma_14", "sma_28", "sma_50", "sma_100",
    "rsi_9", "rsi_14", "rsi_28",
    "stochastic_k", "stochastic_d",
    "volume_ratio", "volatility_20d",
    "sma20_gap", "sma_cross", "ret_5d_rank", "momentum_vol",
]
for lag in [1, 3, 5, 10]:
    technical_features.append(f"ret_{lag}d")
technical_features = [c for c in technical_features if c in df.columns]
# Bổ sung nếu thiếu (sma_14/28/100 có thể chưa đủ dữ liệu ở đầu chuỗi)
for w in [5, 10, 20]:
    if f"sma_{w}" in df.columns and f"sma_{w}" not in technical_features:
        technical_features.append(f"sma_{w}")
if "rsi" in df.columns and "rsi_14" not in df.columns:
    technical_features.append("rsi")

# 4. Nhóm Tâm lý thị trường (Sentiment): sức mạnh tâm lý, số lượt nhắc đến
sentiment_features = [c for c in df.columns if c.startswith("sent_")]
if "news_count" in df.columns:
    sentiment_features = sentiment_features + ["news_count"]
sentiment_features = [c for c in sentiment_features if c in df.columns]

feature_groups = {
    "Fundamental": fundamental_features,
    "Technical": technical_features,
    "Sentiment": sentiment_features,
    "Fund+Tech": fundamental_features + technical_features,
    "Fund+Sent": fundamental_features + sentiment_features,
    "Tech+Sent": technical_features + sentiment_features,
    "All": fundamental_features + technical_features + sentiment_features,
}

# Loại biến đa cộng tuyến (|r| > 0.9): giữ 1 trong mỗi cặp
def _drop_high_corr(data, feats, thresh=0.9):
    feats = [f for f in feats if f in data.columns]
    if len(feats) < 2:
        return feats
    corr = data[feats].corr()
    to_drop = set()
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            if abs(corr.iloc[i, j]) > thresh:
                to_drop.add(feats[j])
    return [f for f in feats if f not in to_drop]

_all_before = len(feature_groups["All"])
_kept = _drop_high_corr(df, feature_groups["All"], 0.9)
for _k in feature_groups:
    feature_groups[_k] = [f for f in feature_groups[_k] if f in _kept]
if len(_kept) < _all_before:
    print(f"  Đa cộng tuyến (|r|>0.9): giữ {len(_kept)}/{_all_before} biến.")

# Giữ 1 RSI đại diện (bỏ rsi_9, rsi_28 để giảm đa cộng tuyến)
_rsi_drop = ["rsi_9", "rsi_28"]
if "rsi_14" in feature_groups["Technical"]:
    for _r in _rsi_drop:
        if _r in feature_groups["Technical"]:
            feature_groups["Technical"].remove(_r)
    feature_groups["Fund+Tech"] = feature_groups["Fundamental"] + feature_groups["Technical"]
    feature_groups["Tech+Sent"] = feature_groups["Technical"] + feature_groups["Sentiment"]
    feature_groups["All"] = feature_groups["Fundamental"] + feature_groups["Technical"] + feature_groups["Sentiment"]
    print("  RSI: giữ rsi_14 đại diện (bỏ rsi_9, rsi_28).")

# -------------------------------------------------------------
# 12b. THỐNG KÊ MÔ TẢ: biến phụ thuộc và biến độc lập
# -------------------------------------------------------------
all_feat_cols = [f for f in feature_groups["All"] if f in df.columns]
dep_cols = ["target"]
if "future_return" in df.columns:
    dep_cols.append("future_return")

print("\n" + "=" * 60)
print("Thống kê mô tả: Biến phụ thuộc và Biến độc lập")
print("=" * 60)

# Biến phụ thuộc (dependent)
dep_valid = [c for c in dep_cols if c in df.columns]
if dep_valid:
    desc_dep = df[dep_valid].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc_dep = desc_dep.rename(columns={"50%": "median"})
    desc_dep["missing"] = df[dep_valid].isna().sum()
    if "target" in dep_valid:
        desc_dep.loc["target", "value_counts"] = str(df["target"].value_counts().sort_index().to_dict())
    path_dep = os.path.join(DATA_DIR, "descriptive_stats_dependent.csv")
    desc_dep.to_csv(path_dep, encoding="utf-8-sig")
    print(f"  Biến phụ thuộc: {dep_valid}")
    print(desc_dep.round(4).to_string())
    print(f"  -> Đã lưu: {path_dep}")

# Biến độc lập (independent)
if all_feat_cols:
    desc_ind = df[all_feat_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc_ind = desc_ind.rename(columns={"50%": "median"})
    desc_ind["missing"] = df[all_feat_cols].isna().sum()
    try:
        desc_ind["skew"] = df[all_feat_cols].skew()
    except Exception:
        pass
    path_ind = os.path.join(DATA_DIR, "descriptive_stats_independent.csv")
    desc_ind.to_csv(path_ind, encoding="utf-8-sig")
    print(f"\n  Biến độc lập: {len(all_feat_cols)} biến (xem file để đầy đủ)")
    print(desc_ind[["count", "mean", "std", "min", "median", "max", "missing"]].head(15).round(4).to_string())
    print(f"  -> Đã lưu: {path_ind}")

print("\n  [Đã kiểm tra biến xong. Tiếp tục tiền xử lý dữ liệu...]\n")

# -------------------------------------------------------------
# 12c. TIỀN XỬ LÝ DỮ LIỆU (sau thống kê mô tả)
# -------------------------------------------------------------
# 1. Loại giá trị vô nghĩa (pe, pb > 0)
if "pe" in df.columns:
    before = len(df)
    df = df[df["pe"] > 0].reset_index(drop=True)
    print(f"  Loại pe <= 0: bỏ {before - len(df)} dòng.")
if "pb" in df.columns:
    before = len(df)
    df = df[df["pb"] > 0].reset_index(drop=True)
    print(f"  Loại pb <= 0: bỏ {before - len(df)} dòng.")

# 2. Log-transform giá & SMA (skew > 1) – giảm bias gradient
price_sma_cols = ["close", "open", "high", "low"]
for w in [5, 10, 14, 20, 28, 50, 100]:
    if f"sma_{w}" in df.columns:
        price_sma_cols.append(f"sma_{w}")
for col in price_sma_cols:
    if col in df.columns:
        df[col] = np.log(df[col].clip(lower=1e-8))
print("  Log(price): close, open, high, low, sma_*.")

# 3. Winsorize (1%–99%) + log cho fundamental lệch mạnh
cols_clip = ["pe", "pb", "lnst_yoy", "eps", "nophaitra_vcsh", "volume_ratio", "roa", "roe", "vonhoa_tts"]
for col in cols_clip:
    if col in df.columns:
        q01, q99 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(q01, q99)
print("  Winsorize (0.01, 0.99): pe, pb, eps, roa, roe, vonhoa_tts, nophaitra_vcsh, lnst_yoy.")

# 4. Log(1+x) cho biến tỷ lệ / fundamental
if "pe" in df.columns:
    df["pe"] = np.log1p(df["pe"])
if "pb" in df.columns:
    df["pb"] = np.log1p(df["pb"])
if "eps" in df.columns:
    df["eps"] = np.log1p(df["eps"].clip(lower=0))
if "lnst_yoy" in df.columns:
    df["lnst_yoy"] = np.sign(df["lnst_yoy"]) * np.log1p(np.abs(df["lnst_yoy"]))
for col in ["roa", "roe", "vonhoa_tts", "nophaitra_vcsh"]:
    if col in df.columns:
        df[col] = np.log1p(df[col].clip(lower=0))
print("  Log transform: pe, pb, eps, lnst_yoy, roa, roe, vonhoa_tts, nophaitra_vcsh.")

# 5. Fundamental: within-stock z-score (ROA, ROE) → composite profitability
for _col in ["roa", "roe"]:
    if _col in df.columns:
        _mu = df.groupby("mack")[_col].transform("mean")
        _std = df.groupby("mack")[_col].transform("std")
        df[f"{_col}_z"] = np.where(_std > 1e-8, (df[_col] - _mu) / _std, 0)
if "roa_z" in df.columns and "roe_z" in df.columns:
    df["profitability"] = 0.5 * df["roa_z"] + 0.5 * df["roe_z"]
    print("  Fundamental: within-stock z (roa, roe) → profitability = 0.5*roa_z + 0.5*roe_z.")

# 6. Sentiment: news_dummy (trước khi log), log(1+news_count), sent_roll 5d, sent_shock
if "news_count" in df.columns:
    df["news_dummy"] = (df["news_count"].fillna(0) > 0).astype(int)
    df["news_count"] = np.log1p(df["news_count"].clip(lower=0))
_sent_col = "sent_score_mean" if "sent_score_mean" in df.columns else "sent_score"
if _sent_col in df.columns:
    df["sent_score_roll"] = df.groupby("mack", group_keys=False)[_sent_col].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["sent_shock"] = df[_sent_col] - df["sent_score_roll"]
print("  Sentiment: news_dummy, log(1+news_count), sent_score_roll (5d), sent_shock.")

# 7. Volume ratio: log(1+volume_ratio) giảm skew
if "volume_ratio" in df.columns:
    df["volume_ratio"] = np.log1p(df["volume_ratio"].clip(lower=0))
print("  Volume ratio: log(1+volume_ratio).")

# 8. Feature interaction: momentum_sent, volume_sent (tăng AUC)
if "ret_3d" in df.columns and "sent_score_roll" in df.columns:
    df["momentum_sent"] = df["ret_3d"] * df["sent_score_roll"]
if "volume_ratio" in df.columns and "sent_score_roll" in df.columns:
    df["volume_sent"] = df["volume_ratio"] * df["sent_score_roll"]
print("  Interaction: momentum_sent = ret_3d * sent_roll, volume_sent = vol_ratio * sent_roll.")

# 9. Fill missing (fundamental) bằng median
for col in fundamental_features:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
if "profitability" in df.columns and df["profitability"].isna().any():
    df["profitability"] = df["profitability"].fillna(df["profitability"].median())
print("  Fill missing: fundamental + profitability.")

# 10. Cập nhật feature groups: profitability; sentiment gọn; bỏ stochastic_k; thêm interaction
if "profitability" in df.columns:
    feature_groups["Fundamental"] = [f for f in feature_groups["Fundamental"] if f not in ["roa", "roe"]] + ["profitability"]
    print("  Fundamental: dùng profitability thay roa, roe.")
_sent_keep = [c for c in ["sent_score_roll", "sent_shock", "news_dummy"] if c in df.columns]
if _sent_keep:
    feature_groups["Sentiment"] = _sent_keep
    print("  Sentiment: giữ sent_score_roll, sent_shock, news_dummy.")
if "stochastic_k" in feature_groups["Technical"]:
    feature_groups["Technical"].remove("stochastic_k")
    print("  Technical: bỏ stochastic_k (giữ RSI).")
for _inter in ["momentum_sent", "volume_sent"]:
    if _inter in df.columns and _inter not in feature_groups["Technical"]:
        feature_groups["Technical"].append(_inter)
feature_groups["Fund+Tech"] = feature_groups["Fundamental"] + feature_groups["Technical"]
feature_groups["Fund+Sent"] = feature_groups["Fundamental"] + feature_groups["Sentiment"]
feature_groups["Tech+Sent"] = feature_groups["Technical"] + feature_groups["Sentiment"]
feature_groups["All"] = feature_groups["Fundamental"] + feature_groups["Technical"] + feature_groups["Sentiment"]

# Chuẩn hóa: RobustScaler khi train (có thể tách Robust cho fundamental, Standard cho technical sau)
print("  Chuẩn hóa: RobustScaler khi train từng fold (ret_5d_rank không scale).\n")

# -------------------------------------------------------------
# 12d. THỐNG KÊ MÔ TẢ (sau tiền xử lý) – kiểm tra biến đã đúng chưa
# -------------------------------------------------------------
_all_feat2 = [f for f in feature_groups["All"] if f in df.columns]
_dep_cols2 = ["target"]
if "future_return" in df.columns:
    _dep_cols2.append("future_return")

print("\n" + "=" * 60)
print("Thống kê mô tả (SAU tiền xử lý) – kiểm tra các biến")
print("=" * 60)

_dep_valid2 = [c for c in _dep_cols2 if c in df.columns]
if _dep_valid2:
    _desc_dep2 = df[_dep_valid2].describe(percentiles=[0.25, 0.5, 0.75]).T
    _desc_dep2 = _desc_dep2.rename(columns={"50%": "median"})
    _desc_dep2["missing"] = df[_dep_valid2].isna().sum()
    if "target" in _dep_valid2:
        _desc_dep2.loc["target", "value_counts"] = str(df["target"].value_counts().sort_index().to_dict())
    _path_dep2 = os.path.join(DATA_DIR, "descriptive_stats_dependent_after_preprocess.csv")
    try:
        _desc_dep2.to_csv(_path_dep2, encoding="utf-8-sig")
        print(f"  Biến phụ thuộc: {_dep_valid2}")
        print(_desc_dep2.round(4).to_string())
        print(f"  -> Đã lưu: {_path_dep2}")
    except PermissionError:
        print(f"⚠️ Không ghi được {_path_dep2} (Permission denied). Đóng file nếu đang mở trong Excel rồi chạy lại, hoặc xóa file cũ.")

if _all_feat2:
    _desc_ind2 = df[_all_feat2].describe(percentiles=[0.25, 0.5, 0.75]).T
    _desc_ind2 = _desc_ind2.rename(columns={"50%": "median"})
    _desc_ind2["missing"] = df[_all_feat2].isna().sum()
    try:
        _desc_ind2["skew"] = df[_all_feat2].skew()
    except Exception:
        pass
    _path_ind2 = os.path.join(DATA_DIR, "descriptive_stats_independent_after_preprocess.csv")
    try:
        _desc_ind2.to_csv(_path_ind2, encoding="utf-8-sig")
        print(f"\n  Biến độc lập: {len(_all_feat2)} biến (xem file để đầy đủ)")
        print(_desc_ind2[["count", "mean", "std", "min", "median", "max", "missing"]].head(15).round(4).to_string())
        print(f"  -> Đã lưu: {_path_ind2}")
    except PermissionError:
        print(f"⚠️ Không ghi được {_path_ind2} (Permission denied). Đóng file nếu đang mở trong Excel rồi chạy lại, hoặc xóa file cũ.")

print("\n  [Đã kiểm tra biến sau tiền xử lý. Tiếp tục huấn luyện mô hình...]\n")

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
        hidden_layer_sizes=MLP_HIDDEN,
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        batch_size=MLP_BATCH_SIZE,
        max_iter=MLP_MAX_ITER,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
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

def rolling_timeseries_auc_with_plot(df, features, n_splits=None):
    if n_splits is None:
        n_splits = ROLLING_N_SPLITS
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
            hidden_layer_sizes=MLP_HIDDEN,
            max_iter=MLP_MAX_ITER,
            batch_size=MLP_BATCH_SIZE,
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
        n_repeats=PERMUTATION_N_REPEATS,
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
# 17.4b Heatmap: 7 tổ hợp biến × 11 ngành (AUC)
# -------------------------------------------------------------
print("\n📊 17.4b Generating Heatmap: Feature Combinations × Sectors (AUC)...")

MIN_SECTOR_SAMPLES = 300
heatmap_rows = []
combination_names = [k for k in feature_groups.keys() if k in results_df["Model"].values]

for sector in sorted(df["sector"].unique()):
    sector_df = df[df["sector"] == sector]
    if len(sector_df) < MIN_SECTOR_SAMPLES:
        continue
    for name in combination_names:
        feats = feature_groups[name]
        res = train_mlp_time_split(sector_df, feats)
        if res is not None:
            heatmap_rows.append({
                "Sector": sector,
                "Combination": name,
                "AUC": res["metrics"]["AUC"],
            })

if heatmap_rows:
    heatmap_df = pd.DataFrame(heatmap_rows)
    pivot_auc = heatmap_df.pivot(index="Sector", columns="Combination", values="AUC")
    # Đảm bảo thứ tự cột giống combination_names
    pivot_auc = pivot_auc[[c for c in combination_names if c in pivot_auc.columns]]

    # Lưu CSV ma trận
    pivot_auc.to_csv("heatmap_combination_sector.csv")
    print("✓ Saved: heatmap_combination_sector.csv")

    # Vẽ heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot_auc,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0.5,
        vmin=0.4,
        vmax=0.7,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "AUC"},
    )
    ax.set_title("Hiệu quả mô hình: 7 tổ hợp biến × ngành (AUC)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tổ hợp biến", fontsize=12)
    ax.set_ylabel("Ngành", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "heatmap_combination_sector.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved: {VIS_DIR}/heatmap_combination_sector.png")
else:
    print("⚠️ Not enough data for heatmap (combination × sector)")

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

    # Biểu đồ tương tác SHAP riêng cho SMA_14 và RSI_14 (tránh chồng text)
    if "sma_14" in best_res["features"] and "rsi_14" in best_res["features"]:
        try:
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(
                "sma_14",
                shap_vals_plot,
                X_test_df,
                interaction_index=best_res["features"].index("rsi_14"),
                show=False
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(VIS_DIR, "shap_interaction_sma14_rsi14.png"),
                dpi=150,
                bbox_inches="tight"
            )
            plt.close()
            print("✓ Saved: shap_interaction_sma14_rsi14.png (SMA_14 × RSI_14, không bị chồng text)")
        except Exception as e:
            print(f"⚠️ Không vẽ được SHAP interaction SMA_14 × RSI_14: {e}")
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

def bootstrap_ci(y_true, y_prob, n_bootstraps=None, ci=0.95, random_state=42):
    if n_bootstraps is None:
        n_bootstraps = BOOTSTRAP_N
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
    best_res["y_prob"],
    n_bootstraps=BOOTSTRAP_N
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

def paired_cv_comparison(df, features1, features2, n_splits=None):
    """
    Perform paired cross-validation for statistical comparison.
    Returns paired AUC scores for each fold.
    """
    if n_splits is None:
        n_splits = WILCOXON_N_SPLITS
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

        model1 = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=MLP_MAX_ITER,
                               batch_size=MLP_BATCH_SIZE, random_state=RANDOM_STATE)
        model1.fit(X1_train, y[train_idx])
        prob1 = model1.predict_proba(X1_test)[:, 1]
        aucs1.append(roc_auc_score(y[test_idx], prob1))

        # Model 2
        X2 = data[features2].values
        scaler2 = RobustScaler()
        X2_train = scaler2.fit_transform(X2[train_idx])
        X2_test = scaler2.transform(X2[test_idx])

        model2 = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=MLP_MAX_ITER,
                               batch_size=MLP_BATCH_SIZE, random_state=RANDOM_STATE)
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
            n_splits=WILCOXON_N_SPLITS
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
