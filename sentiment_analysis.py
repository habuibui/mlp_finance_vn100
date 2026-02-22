# =====================================================
# SENTIMENT ANALYSIS - PhoBERT (Continuous Score)
# =====================================================
"""
Phân tích sentiment tin tức bằng PhoBERT.
Đầu vào: cột title (hoặc text). Đầu ra: sent_pos, sent_neu, sent_neg, sent_score.
Dùng cho pipeline dự đoán giá cổ phiếu (mlp_complete.py).

Pipeline gợi ý:
  1. Chạy sentiment_analysis.py (--recompute nếu cần) → data/news_with_sentiment.csv
  2. Chạy mlp_complete.py: target 30/70 quantile, train 70% / test 30%, AUC & accuracy.
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
class SentimentConfig:
    """Cấu hình pipeline sentiment (PhoBERT)."""
    DATA_DIR = "data"
    INPUT_FILE = os.path.join(DATA_DIR, "news.csv")
    OUTPUT_FILE = os.path.join(DATA_DIR, "news_with_sentiment.csv")
    VIS_DIR = "visualizations"
    MODEL_NAME = "vinai/phobert-base"
    SENTIMENT_MODEL = "wonrax/phobert-base-vietnamese-sentiment"
    MAX_LENGTH = 256
    BATCH_SIZE_CUDA = 64
    BATCH_SIZE_CPU = 16
    # Cột nguồn cho sentiment: 'title' (tiêu đề bài viết, chuẩn cho luận văn)
    SOURCE_TEXT_COL = "title"
    # Cột sau khi chuẩn hóa (đưa vào model)
    TEXT_COL = "text"


def get_device() -> torch.device:
    """Chọn device: CUDA nếu có, ngược lại CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_batch_size(device: torch.device) -> int:
    """Batch size theo device."""
    return (
        SentimentConfig.BATCH_SIZE_CUDA
        if device.type == "cuda"
        else SentimentConfig.BATCH_SIZE_CPU
    )


def load_news(path: str) -> pd.DataFrame:
    """Đọc file news, sort theo mack, date."""
    news = pd.read_csv(path, parse_dates=["date"])
    news = news.sort_values(["mack", "date"]).reset_index(drop=True)
    return news


def prepare_text(
    news: pd.DataFrame,
    text_col: Optional[str] = None,
) -> pd.DataFrame:
    """Chuẩn hóa cột text từ title: fillna, strip, gán vào cột 'text'."""
    text_col = text_col or SentimentConfig.SOURCE_TEXT_COL
    df = news.copy()
    if text_col not in df.columns:
        df[SentimentConfig.TEXT_COL] = ""
        return df
    df[SentimentConfig.TEXT_COL] = (
        df[text_col].fillna("").astype(str).str.replace("\n", " ").str.strip()
    )
    return df


@torch.no_grad()
def batch_phobert_sentiment(
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    texts: list,
    max_length: int = 256,
) -> np.ndarray:
    """Chạy PhoBERT sentiment cho một batch, trả về xác suất (pos, neu, neg)."""
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        outputs = model(**encodings)

    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy().astype(np.float32)


def run_sentiment_pipeline(
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    news: pd.DataFrame,
    batch_size: int,
    max_length: int = 256,
) -> pd.DataFrame:
    """
    Thêm các cột sentiment: sent_pos, sent_neu, sent_neg, sent_score.
    sent_score = pos*1 + neu*0 + neg*(-1) ∈ [-1, 1].
    """
    texts = news[SentimentConfig.TEXT_COL].fillna("").astype(str).tolist()
    n = len(texts)
    all_probs = np.empty((n, 3), dtype=np.float32)

    for i in tqdm(range(0, n, batch_size), desc="PhoBERT Sentiment"):
        batch = texts[i : i + batch_size]
        probs = batch_phobert_sentiment(
            tokenizer, model, device, batch, max_length
        )
        all_probs[i : i + len(batch)] = probs

    news = news.copy()
    news["sent_pos"] = all_probs[:, 0]
    news["sent_neu"] = all_probs[:, 1]
    news["sent_neg"] = all_probs[:, 2]
    news["sent_score"] = (
        news["sent_pos"] * 1 + news["sent_neu"] * 0 + news["sent_neg"] * (-1)
    )
    return news


def plot_sentiment_distribution(news: pd.DataFrame, vis_dir: str) -> None:
    """Vẽ phân phối sentiment score và xác suất pos/neu/neg."""
    os.makedirs(vis_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(news["sent_score"], bins=50, color="#3498db", alpha=0.85)
    axes[0].axvline(
        news["sent_score"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean = {news['sent_score'].mean():.3f}",
    )
    axes[0].set_title("Sentiment Score Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Sentiment Score")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[1].hist(news["sent_pos"], bins=30, alpha=0.6, label="Positive", color="green")
    axes[1].hist(news["sent_neu"], bins=30, alpha=0.6, label="Neutral", color="gray")
    axes[1].hist(news["sent_neg"], bins=30, alpha=0.6, label="Negative", color="red")
    axes[1].set_title("Probability Distributions", fontsize=12, fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(vis_dir, "sentiment_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {out_path}")


def news_stats_table(news: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo bảng thống kê tin tức: tổng số news, số news theo từng doanh nghiệp (mack).
    Trả về DataFrame với cột [mack, news_count], có thêm dòng tổng (mack = 'TỔNG').
    """
    if "mack" not in news.columns:
        total = len(news)
        return pd.DataFrame({"mack": ["TỔNG"], "news_count": [total]})

    per_mack = (
        news.groupby("mack", as_index=False).size().rename(columns={"size": "news_count"})
    )
    total_row = pd.DataFrame({"mack": ["TỔNG"], "news_count": [per_mack["news_count"].sum()]})
    return pd.concat([per_mack, total_row], ignore_index=True)


def print_and_save_news_stats(news: pd.DataFrame, vis_dir: str) -> None:
    """In và lưu bảng thống kê tin tức (tổng news, news theo từng doanh nghiệp)."""
    stats = news_stats_table(news)
    os.makedirs(vis_dir, exist_ok=True)

    total_news = int(stats.loc[stats["mack"] == "TỔNG", "news_count"].iloc[0])
    n_companies = len(stats) - 1

    print("\n" + "=" * 60)
    print("THỐNG KÊ TIN TỨC (NEWS)")
    print("=" * 60)
    print(f"  Tổng số tin (news): {total_news:,}")
    print(f"  Số doanh nghiệp (mack): {n_companies}")
    print("\n  Bảng số tin theo doanh nghiệp (mack) — sắp xếp giảm dần:")
    print("-" * 60)
    # Bảng đầy đủ: từng mack (sort giảm dần) + dòng TỔNG ở cuối
    tbl = stats[stats["mack"] != "TỔNG"].sort_values("news_count", ascending=False)
    tbl_display = pd.concat([tbl, stats[stats["mack"] == "TỔNG"]], ignore_index=True)
    print(tbl_display.to_string(index=False))
    print("-" * 60)
    # Lưu file CSV
    os.makedirs(SentimentConfig.DATA_DIR, exist_ok=True)
    out_csv = os.path.join(SentimentConfig.DATA_DIR, "news_stats.csv")
    stats.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✓ Đã lưu bảng thống kê: {out_csv}")


def print_summary(news: pd.DataFrame) -> None:
    """In tóm tắt thống kê sentiment."""
    print("\n" + "=" * 50)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total articles processed: {len(news):,}")
    print(f"Mean sentiment score: {news['sent_score'].mean():.4f}")
    print(f"Std sentiment score:  {news['sent_score'].std():.4f}")
    print(f"Min sentiment score:  {news['sent_score'].min():.4f}")
    print(f"Max sentiment score:  {news['sent_score'].max():.4f}")


def load_model_and_tokenizer(
    device: torch.device,
) -> Tuple[object, torch.nn.Module]:
    """Load tokenizer và model PhoBERT sentiment."""
    tokenizer = AutoTokenizer.from_pretrained(SentimentConfig.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        SentimentConfig.SENTIMENT_MODEL
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model


def run_sentiment_analysis(
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Chạy toàn bộ pipeline sentiment.
    Nếu đã có file output và không force_recompute thì đọc từ cache.
    """
    cfg = SentimentConfig
    os.makedirs(cfg.VIS_DIR, exist_ok=True)

    if os.path.exists(cfg.OUTPUT_FILE) and not force_recompute:
        print(f"✅ Sentiment cache found: {cfg.OUTPUT_FILE}")
        news = pd.read_csv(cfg.OUTPUT_FILE)
        if "sent_score" in news.columns:
            plot_sentiment_distribution(news, cfg.VIS_DIR)
        print_and_save_news_stats(news, cfg.VIS_DIR)
        return news

    device = get_device()
    print(f"Using device: {device}")

    tokenizer, model = load_model_and_tokenizer(device)
    print("✓ PhoBERT loaded successfully")

    news = load_news(cfg.INPUT_FILE)
    print(f"Total news: {len(news)}")

    news = prepare_text(news)
    batch_size = get_batch_size(device)

    print("\nRunning sentiment analysis...")
    news = run_sentiment_pipeline(
        tokenizer,
        model,
        device,
        news,
        batch_size=batch_size,
        max_length=cfg.MAX_LENGTH,
    )

    print("\nGenerating visualizations...")
    plot_sentiment_distribution(news, cfg.VIS_DIR)

    print(f"\nSaving to {cfg.OUTPUT_FILE}...")
    news.to_csv(cfg.OUTPUT_FILE, index=False)

    print("✓ Sentiment analysis completed!")
    print_summary(news)
    print_and_save_news_stats(news, cfg.VIS_DIR)
    return news


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PhoBERT Sentiment Analysis")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Bỏ qua cache, chạy lại sentiment từ đầu",
    )
    args = parser.parse_args()
    run_sentiment_analysis(force_recompute=args.recompute)
