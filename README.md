# Dự đoán biến động giá cổ phiếu bằng MLP và Sentiment

Pipeline học máy dự đoán xu hướng giá cổ phiếu (lên/xuống) dựa trên chỉ báo kỹ thuật, dữ liệu cơ bản và sentiment tin tức (PhoBERT).

---

## Yêu cầu

- **Python 3.8+**
- Các thư viện chính: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`, `scipy`
- Cho sentiment: `torch`, `transformers`, `tqdm`
- Tùy chọn: `imbalanced-learn` (SMOTE)

Cài đặt ví dụ:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap scipy
pip install torch transformers tqdm
pip install imbalanced-learn
```

---

## Dữ liệu

Đặt các file sau trong thư mục `data/`:

| File | Mô tả |
|------|--------|
| `ohlc.csv` | Giá OHLC + volume theo `mack`, `date` |
| `fundamental.csv` | Chỉ số cơ bản (eps, roe, pe, …) theo `mack`, `nam`, `quy` |
| `news.csv` | Tin tức với cột `mack`, `date`, `title` (hoặc `text`) |

Sau khi chạy sentiment sẽ có thêm: `news_with_sentiment.csv`, `news_stats.csv`.  
Sau khi chạy MLP sẽ có thêm: `descriptive_stats.csv`.

---

## Cách chạy

### 1. Phân tích sentiment (PhoBERT)

Tạo điểm sentiment cho tin tức, lưu vào `data/news_with_sentiment.csv`:

```bash
python sentiment_analysis.py
```

Tính lại từ đầu (bỏ cache):

```bash
python sentiment_analysis.py --recompute
```

**Đầu ra:** `data/news_with_sentiment.csv`, `data/news_stats.csv`, `visualizations/sentiment_distribution.png`

---

### 2. Huấn luyện và đánh giá mô hình MLP

Chạy toàn bộ pipeline MLP (load dữ liệu, tạo feature, train, đánh giá, vẽ đồ thị):

```bash
python mlp_complete.py
```

**Cấu hình chính trong code:**

- **Target:** Nhị phân theo quantile 30/70 (return ≤ q30 → 0, return ≥ q70 → 1).
- **Chia dữ liệu:** 70% thời gian train, 30% test (time-based).
- **Sentiment:** Chỉ dùng điểm `sent_score` (đã gộp theo ngày), không dùng nhãn pos/neu/neg.
- **Đa cộng tuyến:** Loại bỏ cặp biến có |tương quan| > 0.8 trước khi train.

**Đầu ra:**

- `model_results.csv` — Kết quả ablation (AUC, Accuracy, F1, …) theo từng nhóm feature.
- `data/descriptive_stats.csv` — Thống kê mô tả các biến.
- `sector_results.csv`, `statistical_tests.csv`, `feature_combination_analysis.csv` (nếu có).
- Thư mục `visualizations/`: ROC, confusion matrix, learning curve, SHAP, correlation heatmap, rolling AUC, v.v.

---

### 3. Chạy một lần (Sentiment + MLP)

Trên Windows có thể dùng:

```bash
run_all.bat
```

(Sửa đường dẫn Python trong file nếu cần.)

---

## Cấu trúc pipeline MLP (tóm tắt)

| Phase | Nội dung |
|-------|----------|
| 1 | Load OHLC, fundamental, news (có sentiment nếu đã chạy bước 1) |
| 2 | Chỉ báo kỹ thuật (RSI, MACD, SMA, EMA, Bollinger, volume, momentum) |
| 3 | Fundamental resample theo ngày |
| 4 | Gộp sentiment theo ngày (chỉ sent_score) |
| 5 | Merge, xử lý missing, sector |
| 6 | Tạo target 30/70, split 70/30 theo thời gian |
| 7 | Nhóm feature (Fundamental, Technical, Sentiment, All) |
| 7b | Loại đa cộng tuyến (|r| > 0.8) |
| 7c | Thống kê mô tả các biến → `descriptive_stats.csv` |
| 8 | Scale (RobustScaler), SMOTE, train MLP, ablation |
| 9 | TimeSeriesSplit, chọn best model, AUC/Accuracy/F1 |
| 10 | Trực quan: SHAP, permutation importance, ROC, heatmap, learning curve, rolling AUC |

---

## Cấu trúc thư mục gợi ý

```
K224141657/
├── README.md
├── sentiment_analysis.py    # PhoBERT sentiment
├── mlp_complete.py          # Pipeline MLP đầy đủ
├── run_all.bat              # Chạy sentiment + MLP (Windows)
├── data/
│   ├── ohlc.csv
│   ├── fundamental.csv
│   ├── news.csv
│   ├── news_with_sentiment.csv   # Sau sentiment
│   ├── news_stats.csv
│   └── descriptive_stats.csv     # Sau MLP
├── visualizations/               # Đồ thị từ mlp_complete & sentiment
├── model_results.csv
├── sector_results.csv
├── statistical_tests.csv
└── feature_combination_analysis.csv
```

---

## Cải thiện AUC

Nếu mô hình cho AUC xấp xỉ 0.5 (gần đoán ngẫu nhiên), xem file **[IMPROVE_AUC.md](IMPROVE_AUC.md)** để biết hướng kiểm tra dữ liệu, target, feature và mô hình. Trong code đã có **Phase 7d** in tương quan feature–target để chẩn đoán nhanh.

## Ghi chú

- Sentiment dùng mô hình `wonrax/phobert-base-vietnamese-sentiment` (transformers).
- MLP: `RobustScaler`, SMOTE (nếu cài imbalanced-learn), early stopping.
- Heatmap tương quan vẽ trên bộ feature **sau khi** đã bỏ đa cộng tuyến.
