# =============================================================
# MLP PROJECT – Clean & Robust Version
# -------------------------------------------------------------
# This script loads market data, computes technical/fundamental
# features, applies pre‑computed sentiment (via sentiment_analysis.py),
# builds target labels, trains several MLP models with SMOTE &
# RobustScaler, and produces a set of visualisations for a complete
# academic‑style report.
# =============================================================

# -------------------------------------------------------------
# 1. IMPORTS & GLOBAL SETTINGS
# -------------------------------------------------------------
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
print("Computing technical indicators")
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

# Volume ratio (today / 20‑day avg)
vol = ohlc.groupby("mack")["volume"]
ohlc["vol_ma_20"] = vol.transform(lambda x: x.rolling(20).mean())
ohlc["volume_ratio"] = ohlc["volume"] / ohlc["vol_ma_20"]

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
    for col in ["mack", "date"]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing in {name}")

df = (
    ohlc.merge(fund_daily, on=["mack", "date"], how="left")
    .merge(sent_daily, on=["mack", "date"], how="left")
)

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
    "HPG": "Dầu khí", "HSG": "Dầu khí", "NKG": "Dầu khí", "PVT": "Dầu khí", "PVD": "Dầu khí",
    "GVR": "Nguyên vật liệu", "PHR": "Nguyên vật liệu", "DGC": "Nguyên vật liệu",
    "DCM": "Nguyên vật liệu", "DPM": "Nguyên vật liệu", "BMP": "Nguyên vật liệu",
    "VGC": "Nguyên vật liệu", "HT1": "Nguyên vật liệu",
    "FPT": "Công nghệ thông tin", "CMG": "Công nghệ thông tin",
    # ... (additional mappings omitted for brevity)
}
df["sector"] = df["mack"].map(sector_map)
df = df.dropna(subset=["sector"]).reset_index(drop=True)
print(f"Sectors used: {df['sector'].nunique()}")

# -------------------------------------------------------------
# 11. TARGET VARIABLE (threshold‑based)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("Creating target variable")
print("=" * 60)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["mack", "date"]).reset_index(drop=True)

# 1‑day forward return
df["future_return"] = df.groupby("mack")["close"].pct_change().shift(-1)

THRESH = 0.003  # 0.3 % threshold
df["target"] = np.where(
    df["future_return"] > THRESH, 1,
    np.where(df["future_return"] < -THRESH, 0, np.nan)
)

before_target = df.shape[0]
df = df.dropna(subset=["target"]).reset_index(drop=True)
after_target = df.shape[0]
print(f"Target rows kept: {after_target}/{before_target}")
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
]
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
# 13. TRAINING FUNCTION (SMOTE + RobustScaler)
# -------------------------------------------------------------
def train_mlp(df_sector, features, split_date="2023-01-01"):
    # keep only available features
    feats = [f for f in features if f in df_sector.columns]
    if len(feats) < 3:
        return None

    X = df_sector[feats].copy()
    y = df_sector["target"].copy()
    dates = df_sector["date"].values

    # drop rows with any NaNs in the selected features
    mask = X.notna().all(axis=1)
    X, y, dates = X[mask], y[mask], dates[mask]
    if len(y) < 200 or y.nunique() < 2:
        return None

    split_dt = pd.to_datetime(split_date)
    train_idx = dates < split_dt
    X_train, X_test = X[train_idx].values, X[~train_idx].values
    y_train, y_test = y[train_idx].values, y[~train_idx].values
    if len(y_test) < 30 or len(np.unique(y_test)) < 2:
        return None

    # scaling
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # SMOTE (if available)
    if IMBLEARN_AVAILABLE:
        try:
            sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, len(y_train) - 1))
            X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
        except Exception as e:
            print("SMOTE error:", e)

    # deeper MLP with early stopping
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=min(64, len(y_train) // 10),
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    try:
        mlp.fit(X_train_s, y_train)
        y_pred = mlp.predict(X_test_s)
        y_prob = mlp.predict_proba(X_test_s)[:, 1]
        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "AP": average_precision_score(y_test, y_prob),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "features": feats,
        }
    except Exception as e:
        print("Training error:", e)
        return None

# -------------------------------------------------------------
# 14. MAIN TRAINING LOOP
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("Training models per sector & feature group")
print("=" * 60)

results = []
all_preds = {}
sectors = sorted(df["sector"].unique())
for idx, sector in enumerate(sectors, 1):
    df_sec = df[df["sector"] == sector].copy()
    if len(df_sec) < 200:
        continue
    for name, feats in feature_groups.items():
        metrics = train_mlp(df_sec, feats)
        if metrics is None:
            continue
        results.append({
            "Sector": sector,
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "AUC": metrics["AUC"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "AP": metrics["AP"],
        })
        all_preds[f"{sector}_{name}"] = {
            "y_test": metrics["y_test"],
            "y_pred": metrics["y_pred"],
            "y_prob": metrics["y_prob"],
        }
    print(f"[{idx}/{len(sectors)}] ✓ {sector}")

results_df = pd.DataFrame(results)
print(f"\nTraining completed – {len(results_df)} model‑sector combos.\n")

# -------------------------------------------------------------
# 15. VISUALISATIONS (selected key plots)
# -------------------------------------------------------------
# 15.1 Performance heatmaps (AUC & F1)
plt.figure(figsize=(20, 8))
auc_tbl = results_df.pivot(index="Sector", columns="Model", values="AUC").round(3)
sns.heatmap(auc_tbl, annot=True, cmap="RdYlGn", fmt=".3f", linewidths=0.5)
plt.title("AUC Heatmap – Sector vs Model")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "auc_heatmap.png"))
plt.close()

plt.figure(figsize=(20, 8))
f1_tbl = results_df.pivot(index="Sector", columns="Model", values="F1").round(3)
sns.heatmap(f1_tbl, annot=True, cmap="RdYlGn", fmt=".3f", linewidths=0.5)
plt.title("F1‑Score Heatmap – Sector vs Model")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "f1_heatmap.png"))
plt.close()

# 15.2 ROC curves (aggregated per model)
plt.figure(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(feature_groups)))
for i, model_name in enumerate(feature_groups.keys()):
    y_true, y_score = [], []
    for key, pred in all_preds.items():
        if key.endswith(f"_{model_name}"):
            y_true.extend(pred["y_test"])
            y_score.extend(pred["y_prob"])
    if y_true:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})", color=colors[i])
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Aggregated per Model")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "roc_curves.png"))
plt.close()

# 15.3 Confusion matrices – best & worst AUC models
if not results_df.empty:
    best_key = results_df.loc[results_df["AUC"].idxmax(), ["Sector", "Model"]]
    worst_key = results_df.loc[results_df["AUC"].idxmin(), ["Sector", "Model"]]
    best_id = f"{best_key['Sector']}_{best_key['Model']}"
    worst_id = f"{worst_key['Sector']}_{worst_key['Model']}"

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for ax, pid, title in zip(axs,
                              [best_id, worst_id],
                              ["Best Model", "Worst Model"]):
        pred = all_preds.get(pid)
        if pred:
            cm = confusion_matrix(pred["y_test"], pred["y_pred"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{title}: {pid}")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "confusion_matrices.png"))
    plt.close()

# -------------------------------------------------------------
# 16. SAVE RESULTS
# -------------------------------------------------------------
results_df.to_csv("model_results.csv", index=False)
print("Results saved to model_results.csv and visualisations in 'visualizations/' folder.")
print("=" * 60)
print("✅ PIPELINE COMPLETED")
print("=" * 60)