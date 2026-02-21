# Dự đoán biến động giá cổ phiếu bằng MLP và Sentiment (PhoBERT)

Đề tài sử dụng **Multi-Layer Perceptron (MLP)** kết hợp **chỉ báo kỹ thuật**, **dữ liệu cơ bản** và **sentiment từ tin tức (PhoBERT)** để dự đoán xu hướng lợi suất 5 ngày của cổ phiếu Việt Nam.

---

## Cấu trúc dự án

| Thành phần | Mô tả |
|------------|--------|
| **mlp_complete.py** | Pipeline chính: load dữ liệu, tính chỉ báo, gộp fundamental + sentiment, huấn luyện MLP, ablation study, đánh giá và trực quan hóa. |
| **sentiment_analysis.py** | Chạy **một lần** để tạo file sentiment: dùng PhoBERT (wonrax/phobert-base-vietnamese-sentiment) phân tích sentiment tiếng Việt trên tiêu đề tin → xuất `data/news_with_sentiment.csv`. |
| **data/** | `ohlc.csv`, `fundamental.csv`, `news.csv`. Sau khi chạy sentiment: `news_with_sentiment.csv`. |
| **visualizations/** | Các biểu đồ: ablation AUC, ROC, confusion matrix, SHAP, learning curve, sector performance, bootstrap CI, v.v. |
| **model_results.csv** | Kết quả ablation (Accuracy, AUC, F1, …) theo từng nhóm feature. |

---

## Cách chạy

1. **Chuẩn bị dữ liệu**  
   Đặt `ohlc.csv`, `fundamental.csv`, `news.csv` trong thư mục `data/`.

2. **Tạo sentiment (chạy một lần)**  
   ```bash
   python sentiment_analysis.py
   ```  
   Cần: `torch`, `transformers`, `tqdm`. Nếu đã có `data/news_with_sentiment.csv` thì script sẽ bỏ qua inference và chỉ vẽ biểu đồ phân bố sentiment.

3. **Chạy pipeline MLP**  
   ```bash
   python mlp_complete.py
   ```  
   Cần: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`, `scipy`. Tùy chọn: `imbalanced-learn` (SMOTE).

4. **Chạy tất cả (Windows)**  
   ```bash
   run_all.bat
   ```

---

## Tóm tắt hai file chính

### 1. sentiment_analysis.py

- **Input:** `data/news.csv` (cột `title`, `date`, `mack`).
- **Xử lý:** Tokenize bằng PhoBERT, inference batch (GPU/CPU), softmax → xác suất positive / neutral / negative.
- **Output:** `data/news_with_sentiment.csv` với các cột: `sent_pos`, `sent_neu`, `sent_neg`, `sent_score` (= pos − neg), `sentiment_label` ∈ {−1, 0, 1}.
- **Lưu ý:** Chỉ phân tích **toàn bộ** tin (không loại 40% dữ liệu); cache bằng cách kiểm tra tồn tại file output.

### 2. mlp_complete.py

- **Dữ liệu:** Merge OHLC (sau khi thêm chỉ báo kỹ thuật), fundamental (resample theo ngày, ffill), sentiment (aggregate theo mack + date: mean/std/max của sent_*, news_count).
- **Chỉ báo kỹ thuật:** SMA/EMA (5,10,20,50), RSI, MACD, Bollinger Bands, volume_ratio, return lag 1/3/5/10 ngày, volatility_20d.
- **Target:** Cấu hình trong `mlp_complete.py`:
  - **price** (mặc định, theo nghiên cứu): **(1) Hồi quy** – Mô hình dự báo **giá ngày mai** P_pred. **(2) So sánh** – So sánh P_pred với giá hôm nay P_today. **(3) Tín hiệu** – Nếu P_pred > P_today (hoặc tăng > `SIGNAL_THRESHOLD_PCT`, ví dụ 2%) → tín hiệu **Mua (1)**. Lợi ích: *độ chi tiết* (biên độ lợi nhuận), *linh hoạt* (đổi ngưỡng mua/bán không cần train lại). Đánh giá: R², RMSE, MAE trên giá + Accuracy của tín hiệu Mua/Bán.
  - **direction:** Target -1/0/1 theo hướng biến động, regression dự đoán số thực, R²/RMSE/MAE.
  - **quantile** / **binary** / **fixed_pct:** classification 0/1 (AUC, F1, …).
- **Huấn luyện:** Time-based split (train trước 2023-01-01, test sau), RobustScaler, tùy chọn SMOTE, MLP (256-128-64-32, early stopping).
- **Ablation:** So sánh Fundamental, Technical, Sentiment, Fund+Tech, Fund+Sent, Tech+Sent, All.
- **Đánh giá:** Accuracy, AUC, Precision, Recall, F1, AP; ROC, PR curve, confusion matrix; SHAP, permutation importance; bootstrap CI; Wilcoxon test; phân tích theo sector.

---

## Đánh giá: Đề tài đã hiệu quả chưa?

### Kết quả hiện tại (từ model_results.csv)

| Model      | Accuracy | AUC   | F1    |
|-----------|----------|-------|--------|
| Technical | ~0.522   | ~0.525| ~0.522 |
| Fundamental | ~0.514 | ~0.517| ~0.509 |
| Fund+Tech | ~0.509   | ~0.511| ~0.476 |
| Tech+Sent | ~0.515   | ~0.516| ~0.497 |
| All       | ~0.511   | ~0.509| ~0.433 |
| Sentiment | ~0.511   | ~0.500| ~0.225 |

**Nhận xét ngắn gọn:**

- **AUC quanh 0.50–0.52** → sức dự báo **rất yếu**, gần với đoán ngẫu nhiên (0.5). Trong bối cảnh học máy tài chính, AUC > 0.55 thường mới được xem là có tín hiệu nhẹ.
- **Technical** đang tốt nhất trong các cấu hình; **Sentiment** đơn lẻ gần random và F1 rất thấp (lệch lớp / tín hiệu yếu).
- **All** (đủ feature) không tốt hơn Technical, thậm chí thấp hơn → gợi ý **nhiễu hoặc overfitting** khi thêm fundamental + sentiment với cách dùng hiện tại.

**Kết luận:** Đề tài **đã có pipeline đầy đủ, đúng hướng** (dữ liệu, kỹ thuật, fundamental, sentiment, ablation, đánh giá thống kê), nhưng **chưa đạt mức “hiệu quả” về mặt dự báo** (AUC vẫn gần random). Cần chỉnh target, feature, horizon và mô hình thì mới có cơ hội cải thiện rõ rệt.

---

## Nếu kết quả chạy ra không đẹp thì nên làm gì?

### 1. Chỉnh cách tạo target và lượng dữ liệu

- Đang dùng quantile (ví dụ 0.3/0.7) thì **chỉ giữ 60%** mẫu; 40% ở giữa bị bỏ → mất thông tin.
- **Thử:** `TARGET_MODE = "binary"` với `TARGET_FIXED_THRESHOLD_PCT = 0` hoặc `0.005` để dùng **gần 100%** dữ liệu, lớp cân bằng hơn (có thể kết hợp SMOTE hoặc class_weight nếu dùng mô hình hỗ trợ).
- Hoặc **regression**: dự báo `future_return_5d` liên tục thay vì phân lớp; đánh giá bằng RMSE, MAE, correlation với thực tế.

### 2. Horizon ngắn hơn

- **5 ngày** có thể quá xa, tín hiệu bị nhiễu.
- **Thử:** target 1 ngày (`future_return_1d`) để dễ học hơn (có thể sau đó mở rộng lại 3/5 ngày).

### 3. Feature và sentiment

- **Sentiment:** Thêm **lag** (sent_score lag 1–5 ngày), **rolling mean** sentiment theo 3/5 ngày; hoặc tương tác **sector × sentiment**.
- **Aggregation:** Trong `mlp_complete.py` đã gộp sentiment theo ngày; có thể thử cửa sổ 3 ngày, 5 ngày (rolling) thay vì chỉ theo ngày.
- **Fundamental:** Kiểm tra missing, outlier; có thể chỉ giữ vài chỉ số quan trọng (PE, ROE, …) để giảm nhiễu.

### 4. Train/test và validation

- Giữ **time-based split** (không shuffle theo thời gian).
- **Thử expanding window:** train từ đầu đến tăng dần, test trên từng đoạn sau; hoặc TimeSeriesSplit với nhiều fold để ước lượng AUC ổn định hơn.

### 5. Mô hình

- **Grid search** MLP: `hidden_layer_sizes`, `alpha`, `learning_rate_init`, `batch_size`.
- Nếu chuyển sang **PyTorch/Keras**: thêm dropout, batch norm để giảm overfitting.
- So sánh với **baseline đơn giản** (logistic regression, Random Forest) để biết MLP có thực sự đóng góp hay không.

### 6. Đánh giá thực tế

- **Backtest đơn giản:** Nhóm mẫu theo dự báo (ví dụ prob > 0.5 vs ≤ 0.5), so sánh **average return** thực tế giữa hai nhóm; nếu nhóm “dự báo tăng” có return trung bình cao hơn rõ rệt thì mới có giá trị thực tế.
- Đã có **phân tích theo sector**; có thể báo cáo rõ sector nào model hoạt động tương đối tốt / kém.

---

## Đề xuất các hướng đi tiếp theo

| Hướng | Nội dung |
|-------|----------|
| **1. Target & dữ liệu** | Chuyển sang binary với threshold nhỏ hoặc regression; tăng phần dữ liệu dùng (60% → 100%) và so sánh AUC/F1. |
| **2. Horizon** | So sánh target 1d vs 3d vs 5d; có thể báo cáo “model có tín hiệu tốt hơn ở horizon ngắn”. |
| **3. Sentiment nâng cao** | Lag + rolling sentiment; thêm feature tương tác sector–sentiment; có thể thử model sentiment khác (hoặc ensemble) nếu có dữ liệu nhãn. |
| **4. Feature selection** | Dùng permutation importance / SHAP (đã có) để bỏ bớt feature nhiễu; huấn luyện lại chỉ với top-k feature. |
| **5. So sánh mô hình** | Thêm Logistic Regression, Random Forest, XGBoost; so sánh AUC và thời gian huấn luyện; giải thích tại sao chọn MLP (hoặc chuyển sang mô hình tốt hơn). |
| **6. Backtest & ứng dụng** | Viết script backtest đơn giản (theo signal mua/bán hoặc theo xác suất); báo cáo Sharpe ratio hoặc lợi nhuận giả định so với buy-and-hold. |
| **7. Phân tích sector** | Viết phần “kết luận” ngắn: sector nào AUC cao/thấp, có phù hợp với đặc điểm ngành (ví dụ ngân hàng nhạy tin tức hơn). |
| **8. Cải thiện kỹ thuật** | Thêm chỉ báo (ADX, ATR, …) hoặc giảm đa cộng tuyến (loại bỏ feature tương quan quá cao); chuẩn hóa/scale nhất quán. |

---

## Tài liệu thêm

- **IMPROVEMENT_PROPOSALS.md** – Đề xuất cải thiện chi tiết (target mode, class balance, horizon, feature, model).
- **model_results.csv** – Kết quả ablation hiện tại.
- **sector_results.csv** – Hiệu năng theo sector (sau khi chạy mlp_complete.py).

---

## Tóm tắt một dòng

**Đề tài:** Dự đoán xu hướng lợi suất 5 ngày bằng MLP + technical + fundamental + sentiment (PhoBERT). **Hiện trạng:** Pipeline đầy đủ, nhưng AUC ~0.51 (gần random). **Hướng đi:** Chỉnh target (binary/regression), tăng dữ liệu dùng, thử horizon 1d, cải thiện feature sentiment, so sánh mô hình và backtest đơn giản để đánh giá giá trị thực tế.
