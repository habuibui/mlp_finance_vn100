# Dự án dự báo cổ phiếu: MLP

Pipeline gồm **hai bước**: (1) phân tích cảm xúc tin tức bằng PhoBERT, (2) gộp dữ liệu giá – cơ bản – sentiment và huấn luyện mô hình MLP để dự báo xu hướng lợi nhuận (phân loại nhị phân theo quantile return).

---

## Cấu trúc file cần thiết

Toàn bộ cấu trúc thư mục và file của dự án, gồm **file cần có sẵn** (đầu vào / cấu hình) và **file do pipeline tạo ra** (đầu ra).

### Tổng quan cây thư mục

```
K224141657/
├── README.md                          # Mô tả dự án, cấu trúc file và cách chạy
├── requirements.txt                   # Thư viện Python (numpy, pandas, torch, transformers, ...)
├── run_all.bat                        # Script chạy toàn bộ pipeline (Windows)
│
├── sentiment_analysis.py              # Bước 1: Phân tích cảm xúc PhoBERT → news_with_sentiment.csv
├── mlp_complete.py                    # Bước 2: Gộp dữ liệu, feature, train MLP, đánh giá, vẽ đồ thị
│
├── data/                              # Dữ liệu đầu vào (bắt buộc) + file trung gian/đầu ra
│   ├── ohlc.csv                       # [ĐẦU VÀO] Giá OHLC + volume theo ngày, theo mã (mack)
│   ├── fundamental.csv                # [ĐẦU VÀO] Chỉ số cơ bản theo quý/năm
│   ├── news.csv                       # [ĐẦU VÀO] Tin tức thô (mack, date, title/content/summary)
│   ├── news_with_sentiment.csv        # [ĐẦU RA bước 1] Tin đã gán sentiment (sentiment_analysis.py)
│   ├── descriptive_stats.csv          # [ĐẦU RA] Thống kê mô tả tổng hợp (mlp_complete.py)
│   ├── descriptive_stats_dependent.csv
│   ├── descriptive_stats_independent.csv
│   ├── descriptive_stats_dependent_after_preprocess.csv
│   └── descriptive_stats_independent_after_preprocess.csv
│
├── visualizations/                    # Biểu đồ do pipeline sinh ra khi chạy
│   ├── sentiment_distribution.png     # Phân phối sentiment (bước 1)
│   ├── ablation_auc_bar.png
│   ├── rolling_auc.png
│   ├── precision_recall_curve.png
│   ├── confusion_matrix.png
│   ├── roc_curve_best_model.png
│   ├── permutation_importance.png
│   ├── feature_correlation.png
│   ├── learning_curve.png
│   ├── radar_metrics.png
│   ├── sector_performance.png
│   ├── heatmap_combination_sector.png
│   ├── shap_summary.png
│   ├── shap_interaction_sma14_rsi14.png
│   ├── auc_confidence_intervals.png
│   ├── statistical_comparison.png
│   └── feature_combination_comparison.png
│
├── model_results.csv                  # [ĐẦU RA] AUC, Accuracy, F1,... theo tổ hợp feature
├── sector_results.csv                 # [ĐẦU RA] Hiệu suất mô hình theo từng ngành
├── statistical_tests.csv              # [ĐẦU RA] Kiểm định Wilcoxon (so sánh mô hình)
├── feature_combination_analysis.csv   # [ĐẦU RA] Phân tích ablation chi tiết
└── heatmap_combination_sector.csv     # [ĐẦU RA] Heatmap tổ hợp feature × sector
```

### Bảng tóm tắt theo vai trò

| Vai trò | File / thư mục | Ghi chú |
|--------|-----------------|--------|
| **Cấu hình & chạy** | `README.md`, `requirements.txt`, `run_all.bat` | Cần có sẵn (trừ output của README). |
| **Script chính** | `sentiment_analysis.py`, `mlp_complete.py` | Bắt buộc để chạy pipeline. |
| **Notebook** | `adidaaphat.ipynb`, `run.ipynb` | Tùy chọn, dùng thử nghiệm / chạy nhanh. |
| **Dữ liệu đầu vào** | `data/ohlc.csv`, `data/fundamental.csv`, `data/news.csv` | Bắt buộc; thiếu thì pipeline báo lỗi. |
| **Đầu ra bước 1** | `data/news_with_sentiment.csv` | Tạo bởi `sentiment_analysis.py`; nếu có sẵn có thể bỏ qua bước 1. |
| **Đầu ra bước 2 (data/)** | `data/descriptive_stats*.csv` | Tạo bởi `mlp_complete.py`. |
| **Đầu ra bước 2 (gốc)** | `model_results.csv`, `sector_results.csv`, `statistical_tests.csv`, `feature_combination_analysis.csv`, `heatmap_combination_sector.csv` | Tạo bởi `mlp_complete.py`. |
| **Biểu đồ** | `visualizations/*.png` | Thư mục tự tạo khi chạy; đủ các file trên sau khi chạy đủ pipeline. |

---

## Dữ liệu đầu vào

| File | Mô tả |
|------|--------|
| **data/ohlc.csv** | Cột: `date`, `mack`, `open`, `high`, `low`, `close`, `volume`. Dùng để tính chỉ báo kỹ thuật và biến mục tiêu (return tương lai). **Giá đóng cửa = cột `close`.** |
| **data/fundamental.csv** | Cột: `mack`, `nam`, `quy`, `eps`, `roe`, `roa`, `pb`, `pe`, `lnst_yoy`, `nophaitra_vcsh`, `vonhoa_tts`. Resample theo ngày (ffill) rồi merge với OHLC. |
| **data/news.csv** | Tin tức theo `mack`, `date`; có ít nhất cột nội dung (vd. `title`, `content`, `summary`) để đưa vào PhoBERT. |
| **data/news_with_sentiment.csv** | Đầu ra của `sentiment_analysis.py`: thêm `sent_pos`, `sent_neu`, `sent_neg`, `sent_score`, `sentiment_label`. Nếu đã có file này, pipeline dùng luôn và có thể bỏ qua bước 1. |

---

## Pipeline tổng quan

### Bước 1: Sentiment Analysis (`sentiment_analysis.py`)

- Đọc **data/news.csv**, lấy cột văn bản (ưu tiên content/summary/title).
- Dùng mô hình **PhoBERT** (`wonrax/phobert-base-vietnamese-sentiment`) để dự đoán xác suất positive / neutral / negative.
- Tạo: `sent_pos`, `sent_neu`, `sent_neg`, `sent_score` (= pos − neg), `sentiment_label` (−1, 0, 1).
- Ghi **data/news_with_sentiment.csv** và vẽ **visualizations/sentiment_distribution.png**.

### Bước 2: MLP Training & Evaluation (`mlp_complete.py`)

Luồng xử lý chính trong code:

| Phase | Nội dung |
|-------|----------|
| **Load** | Đọc OHLC, fundamental, news (hoặc news_with_sentiment). Chuẩn hóa tên cột, ép kiểu số cho giá/volume. |
| **Technical** | Tính SMA, EMA, RSI (9/14/28), MACD, Bollinger Bands, volume ratio, volatility, return trễ (ret_1d, ret_3d, ret_5d, ret_10d), sma20_gap, sma_cross, ret_5d_rank, momentum_vol. |
| **Fundamental** | Chuyển fundamental theo quý → ngày (resample D + ffill theo từng mã). |
| **Sentiment** | Gộp sentiment theo (mack, date): mean/std/max của sent_pos, sent_neg, sent_score; news_count. |
| **Merge** | Merge OHLC + fundamental_daily + sentiment_daily trên (mack, date). Chuẩn hóa kiểu cột `date` (datetime) trước khi merge. |
| **Missing** | Ffill fundamental & sentiment theo mã; điền trung tính cho sentiment còn thiếu. |
| **Sector** | Gán ngành (sector) theo mapping `mack` → sector; có thể chỉ giữ mã có trong mapping. |
| **Target** | Tính `future_return` = pct_change(TARGET_RETURN_DAYS).shift(-TARGET_RETURN_DAYS) trên **close**; phân vị 20% và 80% → gán target = 1 (tăng mạnh) / 0 (giảm mạnh), bỏ quan sát ở giữa. |
| **Feature groups** | Định nghĩa 7 tổ hợp: Fundamental, Technical, Sentiment, Fund+Tech, Fund+Sent, Tech+Sent, **All**. Loại đa cộng tuyến (|r| > 0.9), giữ 1 RSI đại diện (rsi_14). |
| **Train/Test** | Chia theo thời gian (split date); huấn luyện MLP (và có thể so sánh mô hình khác) cho từng tổ hợp feature. |
| **Đánh giá** | Accuracy, AUC, Precision, Recall, F1; TimeSeriesSplit rolling AUC; permutation importance; SHAP; phân tích theo sector; bootstrap CI; Wilcoxon. |
| **Xuất kết quả** | Ghi CSV (model_results, sector_results, statistical_tests, feature_combination_analysis, heatmap), vẽ tất cả biểu đồ vào **visualizations/**. |

---

## Biến mục tiêu và nhóm đặc trưng

- **Biến mục tiêu**: Phân loại nhị phân.
  - `future_return` = tỷ suất sinh lợi **TARGET_RETURN_DAYS** ngày tới, tính từ **giá đóng cửa (close)**.
  - Phân vị 20% và 80% của `future_return` → `target = 1` nếu ≥ q_80, `target = 0` nếu ≤ q_20; quan sát giữa bị loại.
- **Nhóm đặc trưng** (ablation):
  - **Fundamental**: eps, roe, roa, pb, pe, lnst_yoy, nophaitra_vcsh, vonhoa_tts (sau khi bỏ đa cộng tuyến).
  - **Technical**: SMA, RSI, Stochastic, volume_ratio, volatility, ret_*d, sma20_gap, sma_cross, ret_5d_rank, momentum_vol, ...
  - **Sentiment**: các cột bắt đầu bằng `sent_` (và có thể news_count).
  - **All** = Fundamental + Technical + Sentiment (sau khi loại đa cộng tuyến).

---

## Kết quả đầu ra

*(Danh sách file đầu ra chi tiết xem mục **Cấu trúc file cần thiết** ở trên.)*

- **CSV (thư mục gốc)**  
  - `model_results.csv`: Accuracy, AUC, Precision, Recall, F1 (và có thể AP) theo từng tổ hợp feature / mô hình.  
  - `sector_results.csv`: Hiệu suất (AUC, Accuracy, F1) theo từng sector.  
  - `statistical_tests.csv`: Kết quả kiểm định Wilcoxon.  
  - `feature_combination_analysis.csv`: Bảng phân tích ablation.  
  - `heatmap_combination_sector.csv`: Heatmap tổ hợp × sector.

- **CSV trong data/**  
  - Các file thống kê mô tả (descriptive_stats_*.csv) và file diagnostic (nếu script có ghi).

- **visualizations/**  
  - Tất cả biểu đồ: sentiment, ablation, ROC/PR, confusion matrix, permutation importance, SHAP, learning curve, sector, bootstrap CI, so sánh thống kê, heatmap.

---

## Cách chạy

**Yêu cầu:** Python 3.x, cài đặt thư viện theo `requirements.txt`:

```bash
pip install -r requirements.txt
# Nếu dùng sentiment (PhoBERT):
# pip install torch transformers tqdm
# (scikit-learn, pandas, matplotlib, seaborn, scipy, shap, imbalanced-learn cho mlp_complete)
```

**Chạy toàn bộ (Windows):**

```bash
run_all.bat
```

**Chạy từng bước:**

```bash
# 1. Tạo sentiment (chỉ cần khi thay đổi news hoặc lần đầu)
python sentiment_analysis.py

# 2. Huấn luyện và đánh giá MLP
python mlp_complete.py
```

Sau khi chạy xong, xem **model_results.csv** và thư mục **visualizations/** để đánh giá kết quả.
