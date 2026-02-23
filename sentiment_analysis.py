# =====================================================
# SENTIMENT ANALYSIS - PHOBERT
# Run ONCE to create sentiment cache
# =====================================================


import os
import pandas as pd
import numpy as np
import torch
try:
    import requests
    from bs4 import BeautifulSoup
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    CRAWL_AVAILABLE = True
except ImportError:
    CRAWL_AVAILABLE = False
    requests = None
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ==============================
# CONFIG
# ==============================
INPUT_FILE = "data/news.csv"
OUTPUT_FILE = "data/news_with_sentiment.csv"
VIS_DIR = "visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

# Cột nguồn ưu tiên: content > summary > title
SOURCE_TEXT_COLS = ["content", "summary", "title"]
TEXT_COL = "text"
MAX_LENGTH = 384  # Dùng content nên tăng (trước 256)
SENTIMENT_MODEL = "wonrax/phobert-base-vietnamese-sentiment"

# ==============================
# 🚨 CACHE CHECK
# ==============================
if os.path.exists(OUTPUT_FILE):
    print(f"✅ Sentiment cache found: {OUTPUT_FILE}")
    print("➡️ Skipping sentiment analysis.")
    
    # Still generate visualization from cached data
    print("\n📊 Generating sentiment distribution from cache...")
    news = pd.read_csv(OUTPUT_FILE)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Sentiment Label Distribution
    label_counts = news["sentiment_label"].value_counts().sort_index()
    label_names = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    colors = ["#e74c3c", "#95a5a6", "#2ecc71"]
    
    axes[0].bar(
        [label_names[i] for i in label_counts.index],
        label_counts.values,
        color=colors
    )
    axes[0].set_title("Sentiment Label Distribution", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(label_counts.values):
        axes[0].text(i, v + 50, str(v), ha="center", fontsize=10)
    
    # 2. Sentiment Score Distribution
    axes[1].hist(news["sent_score"], bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    axes[1].axvline(news["sent_score"].mean(), color="red", linestyle="--", label=f"Mean: {news['sent_score'].mean():.3f}")
    axes[1].set_title("Sentiment Score Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Sentiment Score (Positive - Negative)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    
    # 3. Probability Distributions
    axes[2].hist(news["sent_pos"], bins=30, alpha=0.6, label="Positive", color="#2ecc71")
    axes[2].hist(news["sent_neu"], bins=30, alpha=0.6, label="Neutral", color="#95a5a6")
    axes[2].hist(news["sent_neg"], bins=30, alpha=0.6, label="Negative", color="#e74c3c")
    axes[2].set_title("Probability Distributions", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Probability")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "sentiment_distribution.png"), dpi=150)
    plt.show()
    print(f"✓ Saved: {VIS_DIR}/sentiment_distribution.png")
    
    exit(0)

# ==============================
# CUDA & PHOBERT LOADING
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
    model = model.to(device)
    model.eval()
    print("✓ PhoBERT loaded successfully")
except Exception as e:
    print(f"❌ Error loading PhoBERT model: {e}")
    print("Please ensure you have internet connection and transformers library installed.")
    exit(1)


# ==============================
# LOAD DATA
# ==============================
print("\nLoading news data...")
try:
    news = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    news = news.sort_values(["mack", "date"])
    print(f"Total news: {len(news)}")
except FileNotFoundError:
    print(f"❌ File not found: {INPUT_FILE}")
    exit(1)


# ==============================
# CRAWL NỘI DUNG BÀI VIẾT (nếu chưa có content)
# Cần: pip install requests beautifulsoup4 lxml
# ==============================
def create_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5)
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def fetch_article_content(url: str, session: requests.Session) -> str:
    """
    Crawl nội dung bài viết từ URL.
    Trả về text đã làm sạch.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text(strip=True) for p in paragraphs])
        return text.strip()
    except Exception:
        return ""


def enrich_with_content(news: pd.DataFrame) -> pd.DataFrame:
    """
    Nếu chưa có cột content thì crawl từ url.
    """
    if "url" not in news.columns:
        print("⚠️ Không có cột url → dùng title, không crawl.")
        news["content"] = ""
        return news
    session = create_session()
    contents = []
    print("\nCrawling article content...")
    for url in tqdm(news["url"].fillna(""), desc="Fetching content"):
        if not url:
            contents.append("")
        else:
            contents.append(fetch_article_content(url, session))
    news["content"] = contents
    return news


# Crawl content nếu chưa có (cần: pip install requests beautifulsoup4 lxml)
if "content" not in news.columns or news["content"].isna().all():
    if CRAWL_AVAILABLE:
        news = enrich_with_content(news)
    else:
        print("  ⚠️ Thiếu thư viện crawl (requests/beautifulsoup4/lxml) → dùng title. Chạy: pip install requests beautifulsoup4 lxml")
        news["content"] = ""
else:
    print("  Đã có cột content, bỏ qua crawl.")


# ==============================
# PREPARE TEXT (ưu tiên content → fallback title)
# ==============================
def prepare_text(news: pd.DataFrame) -> pd.DataFrame:
    """
    Ưu tiên content → nếu rỗng/ngắn thì fallback về title.
    """
    df = news.copy()
    content = df.get("content", pd.Series("", index=df.index)).fillna("").astype(str)
    title = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
    text = np.where(content.str.len() > 50, content, title)
    df[TEXT_COL] = (
        pd.Series(text, index=df.index)
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df


news = prepare_text(news)
# Bỏ dòng không có nội dung đủ dài (tùy chọn: giữ lại thì gán sent = 0 sau)
min_text_len = 10
before = len(news)
news = news[news[TEXT_COL].str.len() >= min_text_len].reset_index(drop=True)
if before > len(news):
    print(f"  Bỏ {before - len(news)} tin có text < {min_text_len} ký tự. Còn {len(news):,} tin.")


# ==============================
# BATCH SENTIMENT FUNCTION
# ==============================
@torch.no_grad()
def batch_phobert_sentiment(texts, max_length=None):
    """
    Batch sentiment analysis using PhoBERT.
    
    Parameters
    ----------
    texts : list[str]
        List of Vietnamese text strings to analyze.
    max_length : int, optional
        Maximum token length for truncation. Default from config (384 when using content).
        
    Returns
    -------
    np.ndarray
        Shape (batch, 3) with probabilities for [positive, neutral, negative].
    """
    if max_length is None:
        max_length = MAX_LENGTH
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}
    
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        outputs = model(**encodings)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    
    return probs.cpu().numpy().astype(np.float32)


# ==============================
# APPLY SENTIMENT (BATCH + PROGRESS)
# ==============================
print("\nRunning sentiment analysis...")

BATCH_SIZE = 64 if device.type == "cuda" else 16

texts = news[TEXT_COL].fillna("").astype(str).tolist()
n = len(texts)

all_probs = np.empty((n, 3), dtype=np.float32)

for i in tqdm(range(0, n, BATCH_SIZE), desc="PhoBERT Sentiment"):
    batch_texts = texts[i:i + BATCH_SIZE]
    probs = batch_phobert_sentiment(batch_texts, max_length=MAX_LENGTH)
    all_probs[i:i + len(batch_texts)] = probs

news["sent_pos"] = all_probs[:, 0]
news["sent_neu"] = all_probs[:, 1]
news["sent_neg"] = all_probs[:, 2]

# Continuous score [-1, 1]
news["sent_score"] = news["sent_pos"] - news["sent_neg"]

# Có tin thực sự không?
news["has_news"] = (news[TEXT_COL].str.len() > 20).astype(int)
# Cường độ tin tức
news["sent_intensity"] = news["sent_score"] * news["has_news"]

# Sentiment shock (bất ngờ thông tin) – thường predictive hơn level
news["sent_score_rolling3"] = (
    news.groupby("mack")["sent_score"]
    .transform(lambda x: x.rolling(3, min_periods=1).mean())
)
news["sent_shock"] = news["sent_score"] - news["sent_score_rolling3"]

# Lag sentiment (phù hợp target 5d)
news["sent_score_lag1"] = news.groupby("mack")["sent_score"].shift(1)
news["sent_score_lag2"] = news.groupby("mack")["sent_score"].shift(2)

# Map to {-1, 0, 1}
sent_label_idx = all_probs.argmax(axis=1)
news["sentiment_label"] = np.select(
    [sent_label_idx == 0, sent_label_idx == 1, sent_label_idx == 2],
    [1, 0, -1]  # positive, neutral, negative
)


# ==============================
# VISUALIZATION
# ==============================
print("\n📊 Generating sentiment visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Sentiment Label Distribution
label_counts = news["sentiment_label"].value_counts().sort_index()
label_names = {-1: "Negative", 0: "Neutral", 1: "Positive"}
colors = ["#e74c3c", "#95a5a6", "#2ecc71"]

axes[0].bar(
    [label_names[i] for i in label_counts.index],
    label_counts.values,
    color=colors
)
axes[0].set_title("Sentiment Label Distribution", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Count")
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 50, str(v), ha="center", fontsize=10)

# 2. Sentiment Score Distribution
axes[1].hist(news["sent_score"], bins=50, color="#3498db", edgecolor="white", alpha=0.8)
axes[1].axvline(news["sent_score"].mean(), color="red", linestyle="--", label=f"Mean: {news['sent_score'].mean():.3f}")
axes[1].set_title("Sentiment Score Distribution", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Sentiment Score (Positive - Negative)")
axes[1].set_ylabel("Frequency")
axes[1].legend()

# 3. Probability Distributions
axes[2].hist(news["sent_pos"], bins=30, alpha=0.6, label="Positive", color="#2ecc71")
axes[2].hist(news["sent_neu"], bins=30, alpha=0.6, label="Neutral", color="#95a5a6")
axes[2].hist(news["sent_neg"], bins=30, alpha=0.6, label="Negative", color="#e74c3c")
axes[2].set_title("Probability Distributions", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Probability")
axes[2].set_ylabel("Frequency")
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "sentiment_distribution.png"), dpi=150)
plt.show()
print(f"✓ Saved: {VIS_DIR}/sentiment_distribution.png")


# ==============================
# SAVE RESULTS
# ==============================
# Giữ đủ cột (sent_pos, sent_neu, sent_neg, sent_score, has_news, sent_intensity,
# sent_shock, sent_score_lag1, sent_score_lag2, ...) để tương thích pipeline & viz.
# Nếu muốn giảm nhiễu có thể chỉ lưu: sent_score, sent_shock, sent_score_lag1, sent_intensity, has_news
print(f"\nSaving to {OUTPUT_FILE}...")
news.to_csv(OUTPUT_FILE, index=False)
print("✓ Sentiment analysis completed!")

# Summary
print("\n" + "=" * 50)
print("SENTIMENT ANALYSIS SUMMARY")
print("=" * 50)
print(f"Total articles processed: {len(news):,}")
print(f"\n--- Sentiment Distribution ---")
for label in sorted(news["sentiment_label"].unique()):
    count = (news["sentiment_label"] == label).sum()
    pct = count / len(news) * 100
    name = label_names[label]
    print(f"  {name:10s}: {count:6,} ({pct:5.1f}%)")
print(f"\nMean sentiment score: {news['sent_score'].mean():.4f}")
print(f"Std sentiment score:  {news['sent_score'].std():.4f}")
