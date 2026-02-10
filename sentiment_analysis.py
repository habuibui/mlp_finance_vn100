# =====================================================
# SENTIMENT ANALYSIS - PHOBERT
# Run ONCE to create sentiment cache
# =====================================================
"""
Sentiment Analysis Module using PhoBERT
----------------------------------------
This module processes Vietnamese news headlines and assigns sentiment scores
using the pre-trained PhoBERT model fine-tuned for Vietnamese sentiment analysis.

Author: [Your Name]
Date: 2026
"""

import os
import pandas as pd
import numpy as np
import torch
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
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "wonrax/phobert-base-vietnamese-sentiment"
    )
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
# PREPARE TEXT
# ==============================
news["text"] = news["title"].fillna("").astype(str)
news["text"] = news["text"].str.replace("\n", " ").str.strip()
news = news.reset_index(drop=True)


# ==============================
# BATCH SENTIMENT FUNCTION
# ==============================
@torch.no_grad()
def batch_phobert_sentiment(texts, max_length=256):
    """
    Batch sentiment analysis using PhoBERT.
    
    Parameters
    ----------
    texts : list[str]
        List of Vietnamese text strings to analyze.
    max_length : int, optional
        Maximum token length for truncation. Default is 256.
        
    Returns
    -------
    np.ndarray
        Shape (batch, 3) with probabilities for [positive, neutral, negative].
    """
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
MAX_LEN = 256
TEXT_COL = "text"

texts = news[TEXT_COL].fillna("").astype(str).tolist()
n = len(texts)

all_probs = np.empty((n, 3), dtype=np.float32)

for i in tqdm(range(0, n, BATCH_SIZE), desc="PhoBERT Sentiment"):
    batch_texts = texts[i:i + BATCH_SIZE]
    probs = batch_phobert_sentiment(batch_texts, max_length=MAX_LEN)
    all_probs[i:i + len(batch_texts)] = probs

news["sent_pos"] = all_probs[:, 0]
news["sent_neu"] = all_probs[:, 1]
news["sent_neg"] = all_probs[:, 2]

# Sentiment score (continuous)
news["sent_score"] = news["sent_pos"] - news["sent_neg"]

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
